
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
import wandb  # W&B 라이브러리 추가

from datagenerator import CombinedDataset
from Vae_Model import VAEStyleModel
from util.latent_field_visualizer import visualize_latent_space

wandb.init(project="1texture2pbrtexture",
           config={ 
                   "learning_rate": 1e-4,
                   })

# 데이터 전처리 함수 정의
def preprocess_data(data):
    data = data.astype(np.float32) / 255.0  # 데이터를 0-1 범위로 정규화
    data = torch.from_numpy(data)  # NumPy 배열을 PyTorch 텐서로 변환
    data = transforms.Resize((512, 512))(data)  # 이미지 크기를 512x512로 조정
    return data


def data_loader(in_dir:str, tar_dir:str):
    combined_dataset = CombinedDataset(input_dir=in_dir, target_dir=tar_dir, transform=preprocess_data)
    # 데이터셋 인덱스 생성 (일부 데이터만 사용)
    num_samples = 20  # 사용할 데이터 샘플 수
    indices = list(range(min(num_samples, len(combined_dataset))))

    # 학습, 검증, 테스트 인덱스 생성
    train_indices = [i for i in indices if i % 10 < 8]
    val_indices = [i for i in indices if i % 10 == 9]
    #test_indices = [i for i in indices if i % 10 == 9]
    batch_size = 8

    # 데이터 로더 생성
    train_loader = DataLoader(Subset(combined_dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(Subset(combined_dataset, val_indices), batch_size=batch_size, shuffle=False,    num_workers=0, pin_memory=True)
    #test_loader = DataLoader(Subset(combined_dataset, test_indices), batch_size=batch_size, shuffle=False,  num_workers=0, pin_memory=True)
    return train_loader, val_loader, None


def init_train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    learning_rate = 1e-4
    vae = VAEStyleModel(img_size=512).to(device)  # 모델을 GPU로 이동
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()
    criterion2 = nn.CrossEntropyLoss()

    return epochs, vae, criterion, optimizer, device, criterion2


def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 체크포인트 저장 디렉토리6
import datetime

# 현재 날짜
today = datetime.date.today()

# yymmdd 형식으로 출력
formatted_date = today.strftime("%y%m%d")
checkpoint_dir = f'checkpoints_{formatted_date}'
os.makedirs(checkpoint_dir, exist_ok=True)

def train():
    input_dir = os.path.join(os.getcwd(), 'Data_preprocessed_one')
    target_dir = os.path.join(os.getcwd(), 'data_prep_9wh')
    
    train_d, val_d, test_d = data_loader(in_dir=input_dir, tar_dir=target_dir)
    epochs, model, criterion, optimizer, device, criterion2 = init_train()

    # W&B 설정
    wandb.watch(model, criterion, log="all", log_freq=10)
        
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for data in tqdm(train_d):
            optimizer.zero_grad()
            input_images, target_images = data 
            input_images = input_images.to(device)
            target_images = target_images.to(device)
            
            # VAE 모델의 순전파
            generated_images, mu, logvar = model(input_images)
            
            # 각 텍스처 맵별 손실 계산
            recon_loss = criterion(generated_images, target_images)

            # KL 다이버전스 손실 계산
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # 최종 손실 계산
            loss = recon_loss + kl_loss

            train_loss += loss.item() * input_images.size(0)
            loss.backward()
            optimizer.step()
        
        train_loss /= len(train_d.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}")

        # W&B에 학습 손실 기록
        wandb.log({"epoch": epoch + 1, "Train Loss": train_loss, "KLDivergance Loss" : kl_loss, "recon loss": recon_loss})
        
        # 잠재 공간 시각화
        #visualize_latent_space(model, train_d, device, epoch + 1, perplexity=min(5, len(train_d.dataset) - 1))
        # 검증 루프
        if val_d is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images in tqdm(val_d):
                    input_images, target_images = images
                    input_images = input_images.to(device)
                    target_images = target_images.to(device)
                    
                    generated_images, mu, logvar = model(input_images)
                    
                    recon_loss = criterion(generated_images, target_images)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kl_loss
                    val_loss += loss.item() * input_images.size(0)
            
            val_loss /= len(val_d.dataset)
            print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss:.4f}")

            # W&B에 검증 손실 기록
            wandb.log({"epoch": epoch + 1, "Val Loss": val_loss})
        
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, checkpoint_path)

if __name__ == "__main__":
    train()
