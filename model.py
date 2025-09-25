import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
from VQ import VectorQuantizer

class VQVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_size, num_embeddings, out_channels):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, latent_size)
        self.vq = VectorQuantizer(num_embeddings, latent_size, 0.25)
        self.decoder = Decoder(latent_size, hidden_dim, out_channels)
        
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)

        x_recon = self.decoder(z_q)
        
        return x_recon, vq_loss, 0.0, indices 