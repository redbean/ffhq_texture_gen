
import os
import psutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.manifold import TSNE

from tqdm import tqdm


from gan.datasetloader import CombinedDataset
from gan.model import Generator, Discriminator
from gan.functions import preprocess_data, data_loader


def init_train():
    # 모델 초기화
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.BCEWithLogitsLoss()
    learning_rate = 1e-4

    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # 학습 루프
    num_epochs = 25
    writer = SummaryWriter()

    # 체크포인트 디렉토리 설정
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return device, netG, netD,criterion, optimizerG, optimizerD, num_epochs, writer, checkpoint_dir
    
def train():
    input_dir = os.path.join(os.getcwd(), 'Data_preprocessed_one')
    target_dir = os.path.join(os.getcwd(), 'data_prep_9wh')
    
    train_d, val_d = data_loader(in_dir=input_dir, tar_dir=target_dir, num_samples=10000)
    
    device,netG, netD, criterion, optimizerG, optimizerD, num_epochs, writer, chkpt = init_train()
    

    scaler = GradScaler()
    
    for epoch in range(0, num_epochs):
        netG.train()
        netD.train()
        
        running_lossD = 0.0
        running_lossG = 0.0

        for i, data in enumerate(tqdm(train_d, desc=f"Epoch {epoch+1}/{num_epochs}"), 0):
            # 실제 데이터로 판별자 학습
            netD.zero_grad()
            real_images, target_images = data
            real_images, target_images = real_images.to(device), target_images.to(device)
            batch_size = real_images.size(0)
            
            # output의 배치 크기 확인
            output_real = netD(torch.cat((real_images, target_images), 1))
            label_real = torch.ones_like(output_real, device=device, dtype=torch.float)  # dtype 변경
            label_fake = torch.zeros_like(output_real, device=device, dtype=torch.float)  # dtype 변경
            
            with autocast():
                # 진짜 이미지 학습
                lossD_real = criterion(output_real, label_real)
                
                # 가짜 이미지 학습
                fake_images = netG(real_images)
                output_fake = netD(torch.cat((real_images, fake_images.detach()), 1))
                lossD_fake = criterion(output_fake, label_fake)
                
                lossD = lossD_real + lossD_fake
            
            scaler.scale(lossD).backward()
            scaler.step(optimizerD)
            scaler.update()
            
            # 생성자 학습
            netG.zero_grad()
            with autocast():
                output = netD(torch.cat((real_images, fake_images), 1))
                lossG = criterion(output, label_real)
            
            scaler.scale(lossG).backward()
            scaler.step(optimizerG)
            scaler.update()
            
            running_lossD += lossD.item()
            running_lossG += lossG.item()
            
            # 학습 상태 출력 및 텐서보드 기록
            if i % 100 == 0:
                avg_lossD = running_lossD / 100
                avg_lossG = running_lossG / 100
                print(f"[{epoch}/{num_epochs}][{i}/{len(train_d)}] Loss_D: {avg_lossD}, Loss_G: {avg_lossG}")
                
                writer.add_scalar('Loss/Discriminator', avg_lossD, epoch * len(train_d) + i)
                writer.add_scalar('Loss/Generator', avg_lossG, epoch * len(train_d) + i)
                
                running_lossD = 0.0
                running_lossG = 0.0
                
        # 검증 루프 (선택 사항)
        netG.eval()
        netD.eval()
        with torch.no_grad():
            val_lossD = 0.0
            val_lossG = 0.0
            for i, data in enumerate(val_d, 0):
                real_images, target_images = data
                real_images, target_images = real_images.to(device), target_images.to(device)
                
                output_real = netD(torch.cat((real_images, target_images), 1))
                label_real = torch.ones_like(output_real, device=device, dtype=torch.float)  # dtype 변경
                label_fake = torch.zeros_like(output_real, device=device, dtype=torch.float)  # dtype 변경
                lossD_real = criterion(output_real, label_real)
                
                fake_images = netG(real_images)
                output_fake = netD(torch.cat((real_images, fake_images), 1))
                lossD_fake = criterion(output_fake, label_fake)
                
                lossD = lossD_real + lossD_fake
                val_lossD += lossD.item()
                
                output = netD(torch.cat((real_images, fake_images), 1))
                lossG = criterion(output, label_real)
                val_lossG += lossG.item()
                
            val_lossD /= len(val_d)
            val_lossG /= len(val_d)
            print(f"Validation Loss_D: {val_lossD}, Loss_G: {val_lossG}")
            
            writer.add_scalar('Validation Loss/Discriminator', val_lossD, epoch)
            writer.add_scalar('Validation Loss/Generator', val_lossG, epoch)

        # 체크포인트 저장
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'lossG': lossG,
                'lossD': lossD
            }, os.path.join(chkpt, f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()

def load_checkpoint(checkpoint_path, netG, netD, optimizerG, optimizerD):
    checkpoint = torch.load(checkpoint_path)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    start_epoch = checkpoint['epoch']
    lossG = checkpoint['lossG']
    lossD = checkpoint['lossD']
    return start_epoch, lossG, lossD

if __name__=="__main__":
    train()
    
