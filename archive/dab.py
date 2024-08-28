# Unsupervised domain adaptation by backpropagation https://arxiv.org/abs/1409.7495

import torch
from torch import nn, Tensor
from torch.autograd import Function
from ..Concord.model.build_layer import get_normalization_layer


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None



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

    def forward(self, x, alpha):
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = ReverseLayerF.apply(x, alpha)
        x = self.discriminator(x)
        return x
    