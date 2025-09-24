import torch
import numpy as np
from model import VQVAE
from Dataset import get_dataloader
import os
def extract_mnist_codes():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = get_dataloader('mnist', batch_size=32)

    model = VQVAE(
        in_channels=1, 
        hidden_dim=64,
        latent_size=64, 
        num_embeddings=512, 
        out_channels=1
    ).to(device)
    model.load_state_dict(torch.load('vqvae_mnist_final.pth', weights_only=True))
    model.eval()
    all_codes = []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            _, _, _, indices = model(data)  
            all_codes.append(indices.cpu().numpy())
            
            if batch_idx % 100 == 0:
                print(f"Processed {batch_idx}/{len(train_loader)} batches")

    codes = np.concatenate(all_codes, axis=0)
    os.makedirs('extracted_data', exist_ok=True)
    np.save('extracted_data/mnist_discrete_codes.npy', codes)
    
    print(f"✅ Extracted codes shape: {codes.shape}")
    print(f"Code range: {codes.min()} - {codes.max()}")
    print(f"Vocabulary size: {codes.max() + 1}")
    print("Saved to extracted_data/mnist_discrete_codes.npy")

if __name__ == '__main__':
    extract_mnist_codes()