
# augment.py
import torch
import torch.nn as nn

class Augmentation(nn.Module):
    """Base class – override __call__ if you need fancier logic."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# Legacy, to be removed in future versions
class DropoutAugment(Augmentation):
    """Exact behaviour of nn.Dropout (applied to *all* entries)."""
    def __init__(self, p: float):
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)

class MaskAugment(Augmentation):
    def __init__(self, p: float):
        super().__init__()
        assert 0.0 <= p <= 1.0
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # no masking during eval / inference
        if not self.training or self.p == 0.0:
            return x

        # keep-prob = 1 – p   →  True means *keep*
        keep_mask = torch.rand_like(x) >= self.p        # boolean
        return x * keep_mask                           # dropped entries = 0


class MaskNonZerosAugment(nn.Module):
    """
    Masks a fraction of non-zero entries (and, optionally, zero entries *if*
    they will change value).
    """
    def __init__(self, p: float,
                 mask_value: float = 0.0):
        super().__init__()
        assert 0.0 <= p <= 1.0
        self.p = p
        self.mask_value = mask_value

        # zero masking is pointless if it doesn't change the tensor
        self._do_zero_masking = (mask_value != 0.0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        out = x.clone()

        # ── 1) mask non-zeros  ───────────────────────────────────────────
        nz_idx = (out != 0).nonzero(as_tuple=False)
        if nz_idx.numel() > 0 and self.p > 0:
            keep_flags = torch.bernoulli(
                torch.full((nz_idx.size(0),), 1 - self.p,
                           device=out.device)
            ).bool()
            drop_idx = nz_idx[~keep_flags]
            out[drop_idx[:, 0], drop_idx[:, 1]] = self.mask_value

        return out


class FeatureDropAugment(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        G = x.size(1)
        keep = torch.rand(G, device=x.device) >= self.p   # 1=keep
        return x * keep.unsqueeze(0)