#!/usr/bin/env python
# coding: utf-8

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
from tqdm import tqdm


from ResNetSD import ResNetSD
from datagenerator import CombinedDataset


# 데이터 전처리 함수 정의
def preprocess_data(data):
    data = data.astype(np.float32) / 255.0  # 데이터를 0-1 범위로 정규화
    data = torch.from_numpy(data)  # NumPy 배열을 PyTorch 텐서로 변환
    data = transforms.Resize((512, 512))(data)  # 이미지 크기를 512x512로 조정
    return data


def data_loader(in_dir:str, tar_dir:str):
    combined_dataset = CombinedDataset(input_dir=in_dir, target_dir=tar_dir, transform=preprocess_data)
    # 데이터셋 인덱스 생성 (일부 데이터만 사용)
    num_samples = 10  # 사용할 데이터 샘플 수
    indices = list(range(min(num_samples, len(combined_dataset))))

    # 학습, 검증, 테스트 인덱스 생성
    train_indices = [i for i in indices if i % 10 < 7]
    val_indices = [i for i in indices if 7 <= i % 10 < 9]
    test_indices = [i for i in indices if i % 10 == 9]
    batch_size = 8

    # 데이터 로더 생성
    train_loader = DataLoader(Subset(combined_dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(Subset(combined_dataset, val_indices), batch_size=batch_size, shuffle=False,    num_workers=0, pin_memory=True)
    test_loader = DataLoader(Subset(combined_dataset, test_indices), batch_size=batch_size, shuffle=False,  num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader


def print_memory_usage(description):
    process = psutil.Process(os.getpid())
    print(f"{description} - Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

def init_train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    learning_rate = 1e-4
    # GPU 사용 가능 여부 확인 및 device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # TensorBoard 작성기 초기화
    writer = SummaryWriter()
    model = ResNetSD(latent_dim=256, num_layers=2).to('cuda')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return epochs, model, writer, criterion, optimizer, device


# 체크포인트 저장 디렉토리
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

def train():
    # 데이터셋 경로 설정
    input_dir = os.path.join(os.getcwd(), 'Data_preprocessed_one')
    target_dir = os.path.join(os.getcwd(), 'data_prep_4chan')
    
    train_d, val_d, test_d = data_loader(in_dir=input_dir, tar_dir=target_dir)
    #return epochs, model, writer, criterion, optimizer

    epochs, model, writer, criterion, optimizer, device = init_train()
        
    for epoch in range(0, epochs):
        model.train()
        train_loss = 0.0
        print("----------")
        print(f"current epoch is {epoch}")
        print("----------")
        for data in tqdm(train_d):
            optimizer.zero_grad()
        
            input_images, target_images = data 
            input_images = input_images.to(device)
            target_images = target_images.to(device)       # 타겟 이미지 (6개 이미지 모두)
            
            timesteps = torch.linspace(1, 0, 4).to(device)  # 4개의 타임스텝 생성
            
            generated_images = model(input_images, timesteps)
                        
            # 각 텍스처 맵별 손실 계산
            loss = 0
            for i in range(4):  # 4개의 텍스처 맵
                loss += criterion(generated_images[:, i], target_images[:, i])

            # 평균 손실 사용
            loss /= 4

            train_loss += loss.item() * input_images.size(0)
            loss.backward()
            optimizer.step()
            
        train_loss /= len(train_d.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}")
            # TensorBoard에 학습 손실 기록
        writer.add_scalar('Train Loss', train_loss, epoch)
        # 검증 루프
        if val_d is not None:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for images in tqdm(val_d):
                    input_images, target_images = images
                    input_images = input_images.to(device)
                    target_images = target_images.to(device)
                    
                    timesteps = torch.linspace(1, 0, 4).to(device)
                    generated_images = model(input_images, timesteps)
                    
                    loss = criterion(generated_images, target_images)
                    val_loss += loss.item() * input_images.size(0)
            
            val_loss /= len(val_d.dataset)
            writer.add_scalar('Val Loss', val_loss, epoch)
        
        if epoch % 10 == 0:
            # 체크포인트 저장
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, checkpoint_path)

    # TensorBoard 작성기 종료
    writer.close()


if __name__ == "__main__":
    train()