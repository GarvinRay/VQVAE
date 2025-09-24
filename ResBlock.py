import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        # 如果输入输出通道不同，需要调整
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_layers, out_channels):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(ResBlock(in_channels, out_channels))
            else:
                layers.append(ResBlock(out_channels, out_channels))
        self.stack = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.stack(x)