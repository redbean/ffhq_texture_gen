
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


from datagenerator import CombinedDataset
from Vae_Model import VAEStyleModel

# 데이터 전처리 함수 정의
def preprocess_data(data):
    data = data.astype(np.float32) / 255.0  # 데이터를 0-1 범위로 정규화
    data = torch.from_numpy(data)  # NumPy 배열을 PyTorch 텐서로 변환
    data = transforms.Resize((512, 512))(data)  # 이미지 크기를 512x512로 조정
    return data


def data_loader(in_dir:str, tar_dir:str):
    combined_dataset = CombinedDataset(input_dir=in_dir, target_dir=tar_dir, transform=preprocess_data)
    # 데이터셋 인덱스 생성 (일부 데이터만 사용)
    num_samples = 2000  # 사용할 데이터 샘플 수
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
    epochs = 1000
    learning_rate = 1e-4
    writer = SummaryWriter('runs')
    vae = VAEStyleModel(img_size=512).to(device)  # 모델을 GPU로 이동
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    return epochs, vae, writer, criterion, optimizer, device


def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def visualize_latent_space(writer, epoch, latents, labels):
    tsne = TSNE(n_components=2)
    latents_2d = tsne.fit_transform(latents)
    metadata = [str(label) for label in labels]
    writer.add_embedding(torch.tensor(latents_2d), metadata=metadata, global_step=epoch, tag='Latent_Space')

# 체크포인트 저장 디렉토리6
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
        
        
        all_latents = []
        all_labels = []
        
        for data in tqdm(train_d):
            optimizer.zero_grad()
            input_images, target_images = data 
            input_images = input_images.to(device)
            target_images = target_images.to(device)
            
            # VAE 모델의 순전파
            generated_images, mu, logvar = model(input_images)
            
            # 각 텍스처 맵별 손실 계산
            recon_loss = 0
            for i in range(4):
                recon_loss += criterion(generated_images[:, i], target_images[:, i])
            recon_loss /= 4

            # KL 다이버전스 손실 계산
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # 최종 손실 계산
            loss = recon_loss + kl_loss

            train_loss += loss.item() * input_images.size(0)
            loss.backward()
            optimizer.step()
            
            # 잠재 벡터 저장
            all_latents.append(mu.cpu().detach().numpy())
            all_labels.append(np.argmax(target_images.cpu().detach().numpy(), axis=1))  # 예시: 원핫 인코딩된 레이블을 인덱스로 변환

        # 잠재 공간 시각화
        all_latents = np.concatenate(all_latents, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        visualize_latent_space(writer, epoch, all_latents, all_labels)
            
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
                    
                    generated_images, mu, logvar = model(input_images)
                    
                    recon_loss = 0
                    for i in range(4):
                        recon_loss += criterion(generated_images[:, i], target_images[:, i])
                    recon_loss /= 4

                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kl_loss
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