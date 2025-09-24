import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        # inputs: [B, embedding_dim, H, W]
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)  # [B*H*W, embedding_dim]
        ################################ 
        # (a - b)^2 = a^2 + b^2 - 2ab  #
        ################################
        distances = torch.sum(flat_input**2, dim=1, keepdim=True) + torch.sum(self._embedding.weight**2, dim=1) - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [B*H*W, 1]
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        ### Stop Gradient 
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        
        # [B, H, W]
        indices_shape = (input_shape[0], input_shape[2], input_shape[3])
        indices = encoding_indices.view(indices_shape)
        
        return quantized, loss, indices