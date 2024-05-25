import torch.nn as nn
import torch

from codec.Encoder import TransformerEncoder
from codec.MyDecoder import MyDecoder

class VAEStyleModel(nn.Module):
    def __init__(self, img_size=512,patch_size =16, in_chans=3, 
                 embed_dim=256, latent_dim=256, 
                 n_layers=4, n_heads=8, 
                 dim_feedforward=256, dropout=0.1):
        super(VAEStyleModel, self).__init__()
        self.encoder = TransformerEncoder(img_size=img_size, patch_size=patch_size,in_chans=in_chans,
                                          embed_dim=embed_dim, 
                                          n_layers=n_layers,
                                          n_heads=n_heads, 
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)
        self.decoder = MyDecoder(input_dim=latent_dim)
        
    def encode(self, x):
        h=self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder.decode(z), mu, logvar

    def generate(self, z):
        return self.decoder.decode(z)