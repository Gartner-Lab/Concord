

import torch
from torch import nn
from torch.nn import functional as F


# SimCLR contrastive loss function
class nt_xent_loss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat((z_i, z_j), dim=0)
        z = nn.functional.normalize(z, dim=1)

        similarity_matrix = torch.matmul(z, z.T) / self.temperature
        mask = torch.eye(batch_size * 2, dtype=torch.bool).to(z.device)
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0).to(z.device)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.masked_fill(mask, 0)

        loss = nn.CrossEntropyLoss()(similarity_matrix.masked_fill(mask, -float('inf')), labels.argmax(dim=1))
        return loss


def importance_penalty_loss(importance_weights, penalty_weight=0.1, norm_type='L1'):
    if penalty_weight == 0:
        return torch.tensor(0.0, device=importance_weights.device)

    input_dim = importance_weights.size(0)
    if norm_type == 'L1':
        base_penalty_weight = 1 / input_dim
        penalty_loss = base_penalty_weight * torch.norm(importance_weights, p=1)
    elif norm_type == 'L2':
        base_penalty_weight = 1 / torch.sqrt(torch.tensor(input_dim, dtype=torch.float32))
        penalty_loss = base_penalty_weight * torch.norm(importance_weights, p=2)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

    return penalty_weight * penalty_loss

