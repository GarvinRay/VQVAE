import torch.nn as nn
from ResBlock import ResidualStack
import torch
class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_dim, out_channels):
        super().__init__()

        self.conv_in = nn.Conv2d(latent_size, hidden_dim, 3, stride=1, padding=1)

        self.resblock = ResidualStack(hidden_dim, 2, hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4, stride=2, padding=1)  # 7->14
        self.deconv2 = nn.ConvTranspose2d(hidden_dim//2, out_channels, 4, stride=2, padding=1)  # 14->28
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, latent_size, 7, 7]
        x = self.conv_in(x)             # [B, hidden_dim, 7, 7]
        x = self.resblock(x)            # [B, hidden_dim, 7, 7]
        x = self.relu(self.deconv1(x))  # [B, hidden_dim//2, 14, 14]
        x = torch.sigmoid(self.deconv2(x))  # [B, out_channels, 28, 28]
        return x