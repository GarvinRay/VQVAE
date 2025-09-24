import torch
import torch.optim as optim
from model import VQVAE
from Dataset import get_dataloader
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def train_vqvae_mnist():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = get_dataloader('mnist', batch_size=32)

    model = VQVAE(
        in_channels=1, 
        hidden_dim=64,        
        latent_size=64, 
        num_embeddings=512, 
        out_channels=1
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 11):  
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            x_recon, vq_loss, _, _ = model(data)

            recon_loss = torch.nn.functional.mse_loss(x_recon, data)
            loss = recon_loss + vq_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch} [{batch_idx:4d}/{len(train_loader)}] '
                     f'Loss: {loss.item():.4f} | '
                     f'Recon: {recon_loss.item():.4f} | '
                     f'VQ: {vq_loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon_loss / len(train_loader)
        avg_vq = total_vq_loss / len(train_loader)
        
        print(f'Epoch {epoch:2d} - Avg Loss: {avg_loss:.4f} | '
              f'Recon: {avg_recon:.4f} | VQ: {avg_vq:.4f}')
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'vqvae_mnist_epoch_{epoch}.pth')

            model.eval()
            with torch.no_grad():
                test_data, _ = next(iter(train_loader))
                test_data = test_data[:8].to(device)
                recon_data, _, _, _ = model(test_data)
                
                # 保存对比图
                comparison = torch.cat([test_data, recon_data], dim=0)
                vutils.save_image(comparison, f'reconstruction_epoch_{epoch}.png', 
                                nrow=8, normalize=True, padding=2)
                print(f"Reconstruction saved as reconstruction_epoch_{epoch}.png")

    torch.save(model.state_dict(), 'vqvae_mnist_final.pth')

if __name__ == '__main__':
    train_vqvae_mnist()


### TO DO !!! 目前的Encoder，Decoder，ResBlock相关的设计是理解了的。
### 但是对应的 VQ 部分 那个停止梯度的操作并不会，这个看一下 Pytorch 部分的实现。
### 加载数据集和绘图部分也不太会，需要学习一下。在 B站 找一下相关的视频。
# def train_vqvae(epoch, model, optimizer, train_loader, device='cuda'):
#     model.train()
#     total_loss = 0
#     for batch_idx, (data, _) in enumerate(train_loader): 
#         data = data.to(device)
#         optimizer.zero_grad()
#         x_recon, codebook_loss, commitment_loss, _ = model(data)
#         recon_loss = torch.nn.functional.mse_loss(x_recon, data)
#         beta = 0.25  
#         loss = recon_loss + codebook_loss + beta * commitment_loss
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         if batch_idx % 100 == 0:
#             print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | Codebook: {codebook_loss.item():.4f} | Commit: {commitment_loss.item():.4f}')
#     print(f'Epoch {epoch} Average Loss: {total_loss / len(train_loader):.4f}', flush=True)

# if __name__ == '__main__':
#     device = 'cuda'
#     train_loader = get_dataloader('cifar10', batch_size=32)
#     model = VQVAE(in_channels=3, hidden_dim=128, latent_size=64, num_embeddings=512, out_channels=3).to(device)
    
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)  
#     for epoch in range(1, 10): 
#         train_vqvae(epoch, model, optimizer, train_loader, device)
    
#     torch.save(model.state_dict(), 'vqvae_cifar10.pth')
#     print("VQ-VAE model saved!")
    
#     model.eval()
#     with torch.no_grad():
#         data, _ = next(iter(train_loader))
#         data = data.to(device)
#         x_recon, _, _, _ = model(data)
        
#         plt.figure(figsize=(8, 4))
#         plt.subplot(1,2,1)
#         plt.title('Original')
#         orig_img = data[0].cpu().permute(1, 2, 0).clamp(0, 1)
#         plt.imshow(orig_img)
#         plt.axis('off')
        
#         plt.subplot(1,2,2)
#         plt.title('Reconstructed')
#         recon_img = x_recon[0].cpu().permute(1, 2, 0).clamp(0, 1)
#         plt.imshow(recon_img)
#         plt.axis('off')
        
#         plt.tight_layout()
#         plt.show()
