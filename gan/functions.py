import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler

from .datasetloader import CombinedDataset

# 데이터 전처리 함수 정의
def preprocess_data(data):
    data = data.astype(np.float32) / 255.0  # 데이터를 0-1 범위로 정규화
    data = torch.from_numpy(data)  # NumPy 배열을 PyTorch 텐서로 변환
    data = transforms.Resize((512, 512))(data)  # 이미지 크기를 512x512로 조정
    return data


def data_loader(in_dir:str, tar_dir:str, num_samples : int= 1000):
    combined_dataset = CombinedDataset(input_dir=in_dir, target_dir=tar_dir, transform=preprocess_data)
    
    # 데이터셋 인덱스 생성 (일부 데이터만 사용)
    indices = list(range(min(num_samples, len(combined_dataset))))

    # 학습, 검증, 테스트 인덱스 생성
    #train_indices = [i for i in indices if i % 10 < 7]
    #val_indices = [i for i in indices if 7 <= i % 10 < 9]
    #test_indices = [i for i in indices if i % 10 == 9]
    train_indices = [i for i in indices if i % 10 < 8]
    val_indices = [i for i in indices if i % 10 == 9]
    batch_size = 8

    # 데이터 로더 생성
    train_loader = DataLoader(Subset(combined_dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(Subset(combined_dataset, val_indices), batch_size=batch_size, shuffle=False,    num_workers=0, pin_memory=True)
    #test_loader = DataLoader(Subset(combined_dataset, test_indices), batch_size=batch_size, shuffle=False,  num_workers=0, pin_memory=True)
    print(f"{len(train_loader)} / {len(train_indices)} loaded")
    
    return train_loader, val_loader, #test_loader
