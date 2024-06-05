import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class ResnetEncoder(nn.Module):
    def __init__(self, latent_dim=512, num_layers=2, in_channels=3):
        super(ResnetEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        resnet = models.resnet50(pretrained=True)

        # 첫 번째 합성곱 층의 in_channels 수정
        if in_channels != 64:  # ResNet50의 기본 in_channels는 64
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 마지막 FC 층을 제외한 모든 레이어를 사용
        modules = list(resnet.children())[:-1]
        self.features = nn.Sequential(*modules)  # 전체 레이어 사용
        self.fc = nn.Linear(resnet.fc.in_features, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x
    
    
if __name__=="__main__":
    
    # 사용 예시
    # 모델 초기화 및 입력 데이터 준비
    model = ResnetEncoder()
    input_tensor = torch.randn(1, 3, 512, 512)  # 가상의 이미지 데이터
    output = model(input_tensor)
    print("done")