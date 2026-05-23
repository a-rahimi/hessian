"""Deep MNIST autoencoder from Hinton & Salakhutdinov 2006.

Architecture: 784 -> 1000 -> 500 -> 250 -> 30 -> 250 -> 500 -> 1000 -> 784,
sigmoid activations on hidden layers and sigmoid on the output. The training
objective is binary cross-entropy reconstruction loss on pixels normalized to
[0, 1]. This matches the canonical Martens & Grosse 2015 benchmark used by
K-FAC.
"""

from __future__ import annotations

import torch
from torch import nn


# Encoder widths and the symmetric decoder widths around the 30-d bottleneck.
ENCODER_DIMS = [784, 1000, 500, 250, 30]
DECODER_DIMS = [30, 250, 500, 1000, 784]


class DeepAutoencoder(nn.Module):
    """The deep autoencoder used as the K-FAC benchmark in Martens & Grosse 2015.

    All hidden activations are sigmoid (the historical choice that makes the
    benchmark hard to optimize with SGD and motivates second-order methods).
    The final layer outputs sigmoids so the output is a per-pixel Bernoulli
    mean compatible with binary cross-entropy.
    """

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        # Encoder.
        for in_dim, out_dim in zip(ENCODER_DIMS[:-1], ENCODER_DIMS[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Sigmoid())
        # Decoder.
        for in_dim, out_dim in zip(DECODER_DIMS[:-1], DECODER_DIMS[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        # The canonical autoencoder pretrains every layer as an RBM before
        # fine-tuning. We do not implement RBM pretraining; instead we use
        # Glorot-uniform init for sigmoid networks, which is the standard
        # replacement used by the K-FAC PyTorch reproductions.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected as (B, 784), already flattened and in [0, 1].
        return self.net(x)


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum-over-pixels binary cross-entropy averaged over the batch.

    The model's training objective is sigmoid cross-entropy (matching the
    official K-FAC autoencoder example, [autoencoder_mnist.py](
    https://github.com/tensorflow/kfac/blob/master/kfac/examples/autoencoder_mnist.py)).
    We use sum-over-pixels divided by batch size so the gradient scale matches
    standard implementations.
    """
    batch_size = recon.shape[0]
    return torch.nn.functional.binary_cross_entropy(recon, target, reduction="sum") / batch_size


def reconstruction_mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum-of-squared-errors per example, averaged over the batch.

    Martens & Grosse 2015 Figure 1 reports test reconstruction *error* as the
    pixel-wise MSE on the sigmoid output, summed across the 784 output units.
    The published numbers we want to reproduce (K-FAC ~0.96 vs SGD ~2.0-2.5)
    are in these units, so we evaluate with this metric in parallel with the
    BCE training loss.
    """
    batch_size = recon.shape[0]
    return ((recon - target) ** 2).sum() / batch_size
