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
        self.fc = nn.Linear(input_dim, 512 * 4 * 4)  # Fully connected layer to expand dimensions

        # Separate upsampling layers for each texture type
        self.diffuse_up_layers = self._create_up_layers()
        self.light_normalized_up_layers = self._create_up_layers()
        self.normal_up_layers = self._create_up_layers()
        self.specular_up_layer = self._create_gray_up_layer()
        self.ao_up_layer = self._create_gray_up_layer()
        self.translucency_up_layer = self._create_gray_up_layer()
    

        # # Separate upsampling layers for each texture type
        # self.rgb_up_layers = nn.Sequential(
        #     UNetBlock(512, 512, up=True),
        #     UNetBlock(512, 512, up=True),
        #     UNetBlock(512, 256, up=True),
        #     UNetBlock(256, 128, up=True),
        #     UNetBlock(128, 64, up=True),
        #     UNetBlock(64, 32, up=True),
        #     nn.ConvTranspose2d(32, 12, 4, 2, 1),  # Output 3 channels for RGB
        #     nn.Tanh()
        # )
        # self.gray_up_layer = nn.Sequential(
        #     UNetBlock(512, 512, up=True),
        #     UNetBlock(512, 512, up=True),
        #     UNetBlock(512, 256, up=True),
        #     UNetBlock(256, 128, up=True),
        #     UNetBlock(128, 64, up=True),
        #     UNetBlock(64, 32, up=True),
        #     nn.ConvTranspose2d(32, 12, 4, 2, 1),  # Output 1 channel for grayscale
        #     nn.Tanh()
        # )
        
    
    def _create_up_layers(self):
        return nn.Sequential(
            UNetBlock(512, 512, up=True),
            UNetBlock(512, 512, up=True),
            UNetBlock(512, 256, up=True),
            UNetBlock(256, 128, up=True),
            UNetBlock(128, 64, up=True),
            UNetBlock(64, 32, up=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),  # Output 3 channels for RGB
            nn.Tanh()
        )

    def _create_gray_up_layer(self):
        return nn.Sequential(
            UNetBlock(512, 512, up=True),
            UNetBlock(512, 512, up=True),
            UNetBlock(512, 256, up=True),
            UNetBlock(256, 128, up=True),
            UNetBlock(128, 64, up=True),
            UNetBlock(64, 32, up=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # Output 1 channel for grayscale
            nn.Tanh()
        )


    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)  # Reshape to a spatial size for convolution

        # RGB textures
        diffuse = self.diffuse_up_layers(x)
        light_normalized = self.light_normalized_up_layers(x)
        normal = self.normal_up_layers(x)

        # Grayscale textures
        specular = self.specular_up_layer(x)
        ao = self.ao_up_layer(x)
        translucency = self.translucency_up_layer(x)

        return diffuse, light_normalized, normal, specular, ao, translucency
        # # RGB textures
        # diffuse = self.rgb_up_layers(x)
        # light_normalized = self.rgb_up_layers(x)
        # normal = self.rgb_up_layers(x)

        # # Grayscale textures, merged into one RGB image
        # specular = self.gray_up_layer(x)
        # ao = self.gray_up_layer(x)
        # translucency = self.gray_up_layer(x)
        # merged_gray_image = torch.cat((specular, ao, translucency), dim=1)  # Merge grayscale images into one RGB image

        #return diffuse, light_normalized, normal, merged_gray_image

    def decode(self, x, timesteps):
        batch_size = x.shape[0]
        output_images = torch.zeros((batch_size, 4, 3, 512, 512), device=x.device)

        diffuse, light_normalized, normal, specular, ao, translucency = self.forward(x)

        output_images[:, 0, :, :, :] = diffuse
        output_images[:, 1, :, :, :] = light_normalized
        output_images[:, 2, :, :, :] = normal

        # Merge grayscale textures into one RGB image
        merged_gray_image = torch.cat((specular, ao, translucency), dim=1)
        output_images[:, 3, :, :, :] = merged_gray_image
        return output_images
    