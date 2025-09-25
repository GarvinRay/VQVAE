import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from PixelCNN import PixelCNN

def train_pixelcnn():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get the CodeBook
    codes = np.load('extracted_data/mnist_discrete_codes.npy')
    codes_tensor = torch.from_numpy(codes).long() ### 必须是long在CrossEntropyLoss里面？？？
    dataset = TensorDataset(codes_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    vocab_size = int(codes.max() + 1)
    model = PixelCNN(
        in_channels=1,
        out_channels=vocab_size,
        num_residual_layer=6,
        hidden_channels=64,
        vocab_size=vocab_size
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for batch_idx, (batch_codes,) in enumerate(dataloader):
            batch_codes = batch_codes.to(device)  # [B, H, W]
            optimizer.zero_grad()
            logits = model(batch_codes)  # [B, vocab_size, H, W]
            loss = criterion(logits, batch_codes)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch:2d} [{batch_idx:4d}/{len(dataloader)}] Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch:2d} Average Loss: {avg_loss:.4f}')

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'pixelcnn_epoch_{epoch}.pth')

    torch.save(model.state_dict(), 'pixelcnn_final.pth')

if __name__ == '__main__':
    train_pixelcnn()