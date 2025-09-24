import torch.nn as nn
from ResBlock import ResidualStack

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim//2, 4, stride=2, padding=1)   
        self.conv2 = nn.Conv2d(hidden_dim//2, hidden_dim, 4, stride=2, padding=1)     
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1)        
        self.resblock = ResidualStack(hidden_dim, 2, hidden_dim)
        self.conv_out = nn.Conv2d(hidden_dim, latent_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, 1, 28, 28]
        x = self.relu(self.conv1(x))    # [B, hidden_dim//2, 14, 14]
        x = self.relu(self.conv2(x))    # [B, hidden_dim, 7, 7]
        x = self.conv3(x)               # [B, hidden_dim, 7, 7]
        x = self.resblock(x)            # [B, hidden_dim, 7, 7]
        x = self.conv_out(x)            # [B, latent_size, 7, 7]
        return x