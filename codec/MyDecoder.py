import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fc = nn.Linear(input_dim, 512 * 4 * 4)

        self.diffuse_up_layers = self._create_up_layers()
        self.light_normalized_up_layers = self._create_up_layers()
        self.normal_up_layers = self._create_up_layers()
        self.specular_up_layer = self._create_gray_up_layer()
        self.ao_up_layer = self._create_gray_up_layer()
        self.translucency_up_layer = self._create_gray_up_layer()

    def _create_up_layers(self):
        return nn.Sequential(
            UNetBlock(512, 512, up=True),
            UNetBlock(512, 512, up=True),
            UNetBlock(512, 256, up=True),
            UNetBlock(256, 128, up=True),
            UNetBlock(128, 64, up=True),
            UNetBlock(64, 32, up=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
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
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)

        diffuse = self.diffuse_up_layers(x)
        light_normalized = self.light_normalized_up_layers(x)
        normal = self.normal_up_layers(x)

        specular = self.specular_up_layer(x)
        ao = self.ao_up_layer(x)
        translucency = self.translucency_up_layer(x)

        return diffuse, light_normalized, normal, specular, ao, translucency

    def decode(self, x):
        batch_size = x.shape[0]
        output_images = torch.zeros((batch_size, 4, 3, 512, 512), device=x.device)

        diffuse, light_normalized, normal, specular, ao, translucency = self.forward(x)

        output_images[:, 0, :, :, :] = diffuse
        output_images[:, 1, :, :, :] = light_normalized
        output_images[:, 2, :, :, :] = normal

        merged_gray_image = torch.cat((specular, ao, translucency), dim=1)
        output_images[:, 3, :, :, :] = merged_gray_image
        return output_images