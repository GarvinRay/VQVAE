import torch
import torch.nn as nn
import torch.nn.functional as F
from MaskedCNN import MaskedCNN
from ResBlock import ResidualStack

class PixelCNN(nn.Module):

    def __init__(self, in_channels=1, out_channels=256, num_residual_layer=6, hidden_channels=64, vocab_size=512):
        super().__init__()
        self.out_channels = out_channels

        self.embedding = nn.Embedding(vocab_size, hidden_channels)

        self.conv1 = MaskedCNN("A", hidden_channels, hidden_channels, 7, padding=3)
        self.res = ResidualStack(hidden_channels, num_residual_layer, hidden_channels)
        self.conv2 = MaskedCNN("B", hidden_channels, hidden_channels, 1)
        self.conv3 = MaskedCNN("B", hidden_channels, hidden_channels, 1)
        self.relu = nn.ReLU()
        self.out = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, x):
        out = self.embedding(x)  
        out = out.permute(0, 3, 1, 2) #  [B, hidden_channels, H, W]
        out = self.conv1(out)
        out = self.res(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.out(out)
        return out
    
    def sample(self, batch_size, height, width, device, temperature=1.0):
        self.eval()
        with torch.no_grad():
            samples = torch.zeros(batch_size, height, width, dtype=torch.long, device=device)

            for h in range(height):# 从左往右，从上到下
                for w in range(width):
                    # 归一化
                    x = samples.float().unsqueeze(1) / self.out_channels  # [B, 1, H, W]

                    logits = self.forward(x)  # [B, out_channels, H, W]
                    logits = logits[:, :, h, w] / temperature  # [B, out_channels]
                    probs = F.softmax(logits, dim=1)
                    next_pixel = torch.multinomial(probs, 1).squeeze(1)  # [B]
                    samples[:, h, w] = next_pixel
                if (h + 1) % 2 == 0:
                    print(f"Completed row {h+1}/{height}")
            
            return samples
        
    def get_loss(self, x, target):
        logits = self.forward(x)  # [B, out_channels, H, W]
        return F.cross_entropy(logits, target)