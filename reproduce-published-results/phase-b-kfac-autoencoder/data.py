"""MNIST loaders for the deep autoencoder benchmark.

Pixels are normalized to [0, 1] and flattened to 784 so the same tensor is
both the network input and the reconstruction target. The cache lives under
``data/`` next to this file and is gitignored.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def _flatten_to_unit_interval(dataset: datasets.MNIST) -> torch.Tensor:
    """Stack all images into an (N, 784) float tensor in [0, 1]."""
    # ``dataset.data`` is uint8 with shape (N, 28, 28). We divide by 255 to
    # land in [0, 1], the range required by binary cross-entropy.
    return dataset.data.float().div_(255.0).view(-1, 784)


def get_mnist_tensors(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Download MNIST into ``data_dir`` and return (train_x, test_x) tensors."""
    data_dir.mkdir(parents=True, exist_ok=True)
    train = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transforms.ToTensor())
    return _flatten_to_unit_interval(train), _flatten_to_unit_interval(test)


def make_loaders(
    data_dir: Path, batch_size: int, device: torch.device
) -> tuple[DataLoader, torch.Tensor]:
    """Return a shuffled training loader and the held-out test tensor on device.

    The training set is materialized as a single tensor on ``device`` (MNIST is
    47 MB at float32, so this fits everywhere) so each batch is a plain
    ``TensorDataset`` slice rather than incurring a fresh copy from disk per
    step.
    """
    train_x, test_x = get_mnist_tensors(data_dir)
    train_x = train_x.to(device)
    test_x = test_x.to(device)
    # The autoencoder uses the input as its own target, so we package only the
    # inputs in the dataset and let the training loop reuse the batch as the
    # target.
    train_ds = TensorDataset(train_x)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, test_x
