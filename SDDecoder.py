import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# class UNetBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, up=True):
#         super(UNetBlock, self).__init__()
#         self.up = up
#         if up:
#             self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
#         else:
#             self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        
#         self.conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         res = x
#         x = self.bn1(self.relu(self.conv1(x)))
#         x = x + res
#         return x
    
# class MyDecoder(nn.Module):
#     def __init__(self, input_dim=256):
#         super(MyDecoder, self).__init__()
#         self.input_dim = input_dim
#         self.fc = nn.Linear(input_dim, 512 * 4 * 4)
#         self.up_layers = nn.Sequential(
#             UNetBlock(512, 512, up=True),
#             UNetBlock(512, 256, up=True),
#             UNetBlock(256, 128, up=True),
#             UNetBlock(128, 64, up=True),
#             UNetBlock(64, 32, up=True)
#         )
#         self.final_up = nn.Sequential(
#             nn.ConvTranspose2d(32, 16, 4, 2, 1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 3, 4, 2, 1),
#             nn.Tanh()
#         )


    
#     def timestep_embedding(self, t):
#         half_dim = self.input_dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
#         emb = t[:, None] * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=1)
#         return emb


#     def forward(self, x, t):
#         타임스텝 임베딩을 생성
#         t_emb = self.timestep_embedding(t)  # 이미 정의된 timestep_embedding 함수를 사용
#         t_emb = t_emb.view(-1, self.input_dim, 1, 1)
        
#         Fully connected layer를 통과시킨 후 4x4로 reshape
#         x = self.fc(x)
#         x = x.view(-1, self.input_dim, 4, 4)
        
#         if t_emb.size(0) != x.size(0):
#             t_emb = t_emb.repeat(x.size(0), 1, 1, 1)  # repeat를 사용하여 크기 조정
#         x = x + t_emb

#         업샘플링 레이어를 통과
#         x = self.up_layers(x)
#         x = self.final_up(x)
#         return x
    
#     def decode(self, x, timesteps):
#         batch_size = x.shape[0]
#         img = torch.zeros((batch_size, timesteps.size(0), 3, 512, 512), device=x.device)

#         for i, t_step in enumerate(range(timesteps.size(0))):
#             t = torch.full((batch_size,), t_step, dtype=torch.float32, device=x.device)
#             img[:, i] = self.forward(x, t)

#         return img

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=True):
        super(UNetBlock, self).__init__()
        self.up = up
        if up:
            self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.up:
            x = self.upconv(x)
        else:
            x = self.conv(x)
            
        res = x  # skip connection을 위해 저장
        x = self.bn1(self.relu(self.conv1(x)))
        x = x + res  # skip connection 추가
        return x
    
class MyDecoder(nn.Module):
    def __init__(self, input_dim=256):
        super(MyDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, 512 * 4 * 4)  # 초기 차원을 4x4 이미지로 변환

        self.up_layers = nn.Sequential(
            UNetBlock(512, 256),
            UNetBlock(256, 128),
            UNetBlock(128, 64),
            UNetBlock(64, 32),
            UNetBlock(32, 16)
        )
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 4, 2, 1),  # 업스케일링
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),   # 최종 채널을 3으로 조정
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)
        x = self.up_layers(x)
        x = self.final_up(x)
        return x

    def decode(self, x, timesteps):
        batch_size = x.shape[0]
        img = torch.zeros((batch_size, timesteps.size(0), 3, 512, 512), device=x.device)
        
        for i, t in enumerate(timesteps):
            noise = torch.randn_like(x)  
            z = x + t * noise 
            img[:, i] = self.forward(z)
        return img