import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from model import VQVAE
from PixelCNN import PixelCNN
import numpy as np
# PixelCNN 生成indices-> 查询 CodeBook 得到 z_q -> z_q 送入 Decoder -> 生成
def sample_from_pixelcnn(pixelcnn, batch_size, H, W, vocab_size, device, temperature=0.8):
    """使用 PixelCNN 自回归采样"""
    pixelcnn.eval()
    with torch.no_grad():
        samples = torch.zeros(batch_size, H, W, dtype=torch.long, device=device)
        
        print(f"Generating {H}x{W} samples...")
        for h in range(H):
            for w in range(W):
                logits = pixelcnn(samples)  # [B, vocab_size, H, W]
                
                # 在当前位置采样
                logits_hw = logits[:, :, h, w] / temperature  # [B, vocab_size]
                probs = F.softmax(logits_hw, dim=1)
                next_token = torch.multinomial(probs, 1).squeeze(1)  # [B]
                samples[:, h, w] = next_token
            
            if (h + 1) % 2 == 0:
                print(f"Generated row {h+1}/{H}")
        
        return samples

def generate_mnist_images(num_samples=8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载 VQ-VAE
    vqvae = VQVAE(
        in_channels=1, 
        hidden_dim=64, 
        latent_size=64, 
        num_embeddings=512, 
        out_channels=1,
    ).to(device)
    vqvae.load_state_dict(torch.load('vqvae_mnist_final.pth', weights_only=True))
    vqvae.eval()
    
    # 获取编码空间信息
    codes = np.load('extracted_data/mnist_discrete_codes.npy')
    H, W = codes.shape[1], codes.shape[2]
    vocab_size = int(codes.max() + 1)
    
    print(f"Code space: {H}x{W}, vocabulary size: {vocab_size}")
    
    # 加载 PixelCNN
    pixelcnn = PixelCNN(
        in_channels=1,
        out_channels=vocab_size,
        num_residual_layer=6,
        hidden_channels=64,
        vocab_size=vocab_size
    ).to(device)
    pixelcnn.load_state_dict(torch.load('pixelcnn_final.pth', weights_only=True))
    
    print(f"Generating {num_samples} MNIST samples...")
    
    # 用 PixelCNN 生成离散编码
    generated_codes = sample_from_pixelcnn(pixelcnn, num_samples, H, W, vocab_size, device)
    
    # 用 VQ-VAE 解码生成图片
    with torch.no_grad():
        quantized = vqvae.vq._embedding(generated_codes)  # [B, H, W, embed_dim]
        quantized = quantized.permute(0, 3, 1, 2)  # [B, embed_dim, H, W]
        
        generated_images = vqvae.decoder(quantized)
        generated_images = torch.clamp(generated_images, 0, 1)
    
    vutils.save_image(generated_images, 'generated_mnist.png', nrow=4, padding=2)
    print("Generated images saved")

    plt.figure(figsize=(12, 6))
    for i in range(min(num_samples, 8)):
        plt.subplot(2, 4, i+1)
        img = generated_images[i].cpu().squeeze()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f'Generated {i+1}')
    
    plt.tight_layout()
    plt.savefig('generated_mnist_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Generation completed!")

if __name__ == '__main__':
    generate_mnist_images(8)