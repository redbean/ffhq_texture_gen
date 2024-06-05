import torch.nn as nn

from ResnetEncoder import ResnetEncoder
from SDDecoder import MyDecoder



class ResNetSD(nn.Module):
    def __init__(self, latent_dim=512, num_layers=2):
        super(ResNetSD, self).__init__()
        self.encoder = ResnetEncoder(latent_dim, num_layers, in_channels=3)
        self.decoder = MyDecoder()

    def forward(self, x, timesteps):
        latent = self.encoder(x)
        output_images = self.decoder.decode(latent, timesteps)
        return output_images
