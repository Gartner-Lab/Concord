

import torch
from torch import nn
from torch.nn import functional as F
import math
    

class NTXent_general(nn.Module):
    def __init__(self, temperature=0.5, beta=0.0):
        super().__init__()
        self.T, self.beta = temperature, beta

    def forward(self, z_i, z_j):
        B = z_i.size(0)
        z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)     # 2B × d
        logits = z @ z.T / self.T                                # 2B × 2B
        pos_idx = (torch.arange(2*B, device=z.device) + B) % (2*B)

        # -------- hard-negative weighting --------
        if self.beta > 0:
            neg_mask = torch.ones_like(logits, dtype=torch.bool)
            neg_mask.fill_diagonal_(False)
            neg_mask[torch.arange(2*B, device=z.device), pos_idx] = False

            neg_logits = logits[neg_mask].view(2*B, -1)                # (2B,2B-2)
            m = neg_logits.size(1)                                     # 2B-2
            log_Z_beta = (torch.logsumexp(self.beta * neg_logits, 1, keepdim=True)
                          - math.log(m))                               # average
            neg_logits = (self.beta + 1) * neg_logits - log_Z_beta
            logits[neg_mask] = neg_logits.reshape(-1)

        # -------- cross-entropy NT-Xent --------
        logits.fill_diagonal_(float('-inf'))
        loss = F.cross_entropy(logits, pos_idx)
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

