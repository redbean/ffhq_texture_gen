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
import cv2
from tqdm import tqdm
from PIL import Image


#from ResNetSD import ResNetSD
from NewModel import NewModel
from datagenerator import CombinedDataset

curr_path = os.getcwd()
fullpath = os.path.join(curr_path, "checkpoints")
all_chkts = sorted(os.listdir(fullpath), reverse=True)
ch_full_path = os.path.join(fullpath, all_chkts[0])

model = NewModel(latent_dim=256, num_layers=2).to('cuda')
print(f"current chpt is {ch_full_path}")

checkpoint = torch.load(ch_full_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# CUDA가 가능한 경우 GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("--------------")
print("initializing")

print("input preprocessing")
fullpath = os.path.join(os.getcwd(), '054')
file = sorted(os.listdir(fullpath))[83]
file_path = os.path.join(fullpath, file)
print(file_path)


image = cv2.imread(file_path, cv2.IMREAD_COLOR)
if image is None:
    print("not exists")
    
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = np.transpose(img, (2,0,1))# HWC에서 CHW로 변환
img = img.astype(np.float32) / 255.0
img = torch.from_numpy(img)
img = transforms.Resize((512, 512))(img)

tensor = img.unsqueeze(0)
print(tensor.shape)

input_tensor = tensor   # 예시 데이터 생성, 실제 데이터는 파일에서 로드하거나 입력받아야 함
sample_data = input_tensor.to(device)

print("input preprocessing Done")
print("--------------")

# 모델로부터 예측 수행
with torch.no_grad():  # 그래디언트 계산을 비활성화하여 메모리 사용량을 줄이고 계산 속도를 높임
    timesteps = torch.linspace(1, 0, 4).to(device)
    output = model(sample_data, timesteps)
    output_tensor = output.squeeze(0)  # 배치 차원 제거
    print(output_tensor.shape)

    num_textures = output_tensor.shape[0]
    num_images = num_textures // 3

    for i, image_tensor in enumerate(tqdm(output_tensor)):
        image_numpy = image_tensor.cpu().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # CHW to HWC
        image_numpy = (image_numpy * 255).astype(np.uint8)  # 스케일링
        image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f'output_image_{i}.png', image_numpy)