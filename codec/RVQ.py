import torch
import torch.nn as nn

class RVQ(nn.Module):
    def __init__(self, num_quantizers, codebook_size, latent_dim):
        super(RVQ, self).__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim

        # 각 양자화 단계의 코드북
        self.codebooks = nn.Parameter(torch.randn(num_quantizers, codebook_size, latent_dim))

    def forward(self, z):
        residual = z
        quantized_vectors = []

        for i in range(self.num_quantizers):
            distances = torch.cdist(residual.unsqueeze(1), self.codebooks[i].unsqueeze(0), p=2)
            encoding_indices = torch.argmin(distances, dim=2)
            quantized_vector = torch.stack([self.codebooks[i][idx] for idx in encoding_indices])
            quantized_vector = quantized_vector.squeeze(1)
            quantized_vectors.append(quantized_vector)
            residual = residual - quantized_vector

        quantized_z = sum(quantized_vectors)
        return quantized_z, encoding_indices