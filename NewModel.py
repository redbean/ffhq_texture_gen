import torch.nn as nn
import torch

from Encoder import TransformerEncoder
from MyDecoder import MyDecoder


class NewModel(nn.Module):
    def __init__(self, latent_dim=256, num_layers=2):
        super(NewModel, self).__init__()
        self.encoder = TransformerEncoder(img_size=latent_dim, in_chans=3)
        self.decoder = MyDecoder()
        
    def forward(self, x, timesteps):
        latent = self.encoder(x)
        output_images = self.decoder.decode(latent, timesteps)
        return output_images
        
if __name__ == "__main__":
    model = NewModel()
    input_tensor = torch.randn(1, 3, 512, 512)  # 가상의 이미지 데이터
    timesteps = torch.linspace(1, 0, 4).to('cuda')  # 4개의 타임스텝 생성

    output = model(input_tensor, timesteps)
    print("done")