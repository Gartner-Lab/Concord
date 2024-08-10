# Unsupervised domain adaptation by backpropagation https://arxiv.org/abs/1409.7495

import torch
from torch import nn, Tensor
from torch.autograd import Function
from .build_layer import get_normalization_layer


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)



class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_domains: int,
        reverse_grad: bool = True,
        norm_type: str = 'layer_norm',
        lambd: float = 1.0
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()

        self.discriminator = nn.Sequential(
            nn.Linear(d_model, d_model),
            get_normalization_layer(norm_type, d_model),
            nn.LeakyReLU(0.1),
            nn.Linear(d_model, n_domains)
        )
        self.reverse_grad = reverse_grad
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=self.lambd)
        x = self.discriminator(x)
        return x
    