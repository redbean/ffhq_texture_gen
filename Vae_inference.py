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
from Vae_Model import VAEStyleModel
from datagenerator import CombinedDataset

curr_path = os.getcwd()
fullpath = os.path.join(curr_path, "checkpoints")
all_chkts = sorted(os.listdir(fullpath), reverse=True)
print(all_chkts)
ch_full_path = os.path.join(fullpath, all_chkts[0])

vae = VAEStyleModel(img_size=512)  # 모델을 GPU로 이동
print(f"current chpt is {ch_full_path}")

checkpoint = torch.load(ch_full_path)
vae.load_state_dict(checkpoint['model_state_dict'])

vae.eval()

# CUDA가 가능한 경우 GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)

print("--------------")
print("initializing")

print("input preprocessing")
fullpath = os.path.join(os.getcwd(), '054000.png')
#fullpath = os.path.join(os.getcwd(), '054')
#file = sorted(os.listdir(fullpath))[83]
#file_path = os.path.join(fullpath, file)
print(fullpath)


image = cv2.imread(fullpath, cv2.IMREAD_COLOR)
if image is None:
    print("not exists")
    
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = np.transpose(img, (2, 0, 1))  # HWC에서 CHW로 변환
img = img.astype(np.float32) / 255.0
img = torch.from_numpy(img)
img = transforms.Resize((512, 512))(img)
img = img.unsqueeze(0).to(device)  # 배치 차원 추가 및 GPU로 이동

tensor = img.unsqueeze(0)
print(tensor.shape)

input_tensor = tensor   # 예시 데이터 생성, 실제 데이터는 파일에서 로드하거나 입력받아야 함
sample_data = input_tensor.to(device)

print("input preprocessing Done")
print("--------------")

# 모델로부터 예측 수행
with torch.no_grad():  # 그래디언트 계산을 비활성화하여 메모리 사용량을 줄이고 계산 속도를 높임
    generated_images, mu, logvar = vae(img)

    
    output_tensor = generated_images.squeeze(0)  # 배치 차원 제거
    print(output_tensor.shape)

    num_textures = output_tensor.shape[0]
    num_images = num_textures // 3

    for i, image_tensor in enumerate(tqdm(output_tensor)):
        image_numpy = image_tensor.cpu().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # CHW to HWC
        image_numpy = (image_numpy * 255).astype(np.uint8)  # 스케일링
        image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f'output_image_{i}.png', image_numpy)