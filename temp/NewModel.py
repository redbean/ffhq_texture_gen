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
        