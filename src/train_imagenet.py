"""
Training script for ImageNet with Hessian-preconditioned gradients.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import argparse
from pathlib import Path

from hessian import DeepMLPWithTracking, hessian_inverse_vector_product


def get_imagenet_loaders(data_dir: str, batch_size: int = 256, num_workers: int = 4):
    """
    Create ImageNet data loaders with standard preprocessing.

    Args:
        data_dir: path to ImageNet dataset
        batch_size: batch size for training
        num_workers: number of data loading workers

    Returns:
        train_loader, val_loader
    """
    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Training transforms
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Validation transforms
    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Load datasets
    train_dataset = torchvision.datasets.ImageFolder(
        root=f"{data_dir}/train", transform=train_transforms
    )

    val_dataset = torchvision.datasets.ImageFolder(
        root=f"{data_dir}/val", transform=val_transforms
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def flatten_gradients(model: nn.Module) -> torch.Tensor:
    """Flatten all parameter gradients into a single vector."""
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.flatten())
    return torch.cat(grads)


def unflatten_and_apply_gradients(model: nn.Module, flat_grad: torch.Tensor, lr: float):
    """
    Unflatten a gradient vector and apply it to model parameters.

    Args:
        model: the neural network
        flat_grad: flattened gradient vector
        lr: learning rate
    """
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param_grad = flat_grad[offset : offset + numel].view(param.shape)
        param.data.add_(param_grad, alpha=-lr)
        offset += numel


def train_epoch_standard(model, train_loader, optimizer, device, epoch):
    """Standard gradient descent training for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Flatten inputs for MLP
        inputs_flat = inputs.view(inputs.size(0), -1)

        # Forward pass
        optimizer.zero_grad()
        loss = model(inputs_flat, targets, track_activations=False)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        with torch.no_grad():
            outputs = model(inputs_flat, track_activations=False)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Acc: {100.*correct/total:.2f}%"
            )

    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, epoch_time


def train_epoch_hessian(model, train_loader, lr, device, epoch, use_hessian_every_n=1):
    """
    Training with Hessian-preconditioned gradients.

    Args:
        model: the neural network
        train_loader: training data loader
        lr: learning rate
        device: device to use
        epoch: current epoch number
        use_hessian_every_n: apply Hessian preconditioning every N batches
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()
    hessian_time = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Flatten inputs for MLP
        inputs_flat = inputs.view(inputs.size(0), -1)

        # Zero gradients
        for param in model.parameters():
            param.grad = None

        # Forward pass with activation tracking
        loss = model(inputs_flat, targets, track_activations=True)

        # Backward pass to compute gradients
        loss.backward(create_graph=True)  # Need create_graph for second derivatives

        # Get gradient vector
        gradient = flatten_gradients(model)

        # Apply Hessian-inverse to gradient
        if batch_idx % use_hessian_every_n == 0:
            hess_start = time.time()
            try:
                preconditioned_grad = hessian_inverse_vector_product(
                    model, loss, gradient
                )
                hessian_time += time.time() - hess_start

                # Apply preconditioned gradient
                unflatten_and_apply_gradients(model, preconditioned_grad, lr)
            except Exception as e:
                print(f"Warning: Hessian computation failed: {e}")
                print("Falling back to standard gradient descent")
                # Fall back to standard gradient if Hessian fails
                unflatten_and_apply_gradients(model, gradient, lr)
        else:
            # Standard gradient update
            unflatten_and_apply_gradients(model, gradient, lr)

        # Statistics
        total_loss += loss.item()
        with torch.no_grad():
            outputs = model(inputs_flat, track_activations=False)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Acc: {100.*correct/total:.2f}% "
                f"Hessian time: {hessian_time:.2f}s"
            )

    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, epoch_time, hessian_time


def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_flat = inputs.view(inputs.size(0), -1)

            loss = model(inputs_flat, targets, track_activations=False)

            total_loss += loss.item()
            outputs = model(inputs_flat, track_activations=False)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP on ImageNet with Hessian preconditioning"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True, help="path to ImageNet dataset"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=90, help="number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--hidden-dim", type=int, default=512, help="hidden dimension for MLP layers"
    )
    parser.add_argument(
        "--num-layers", type=int, default=20, help="number of hidden layers"
    )
    parser.add_argument(
        "--use-hessian",
        action="store_true",
        help="use Hessian-preconditioned gradients",
    )
    parser.add_argument(
        "--hessian-freq", type=int, default=1, help="apply Hessian every N batches"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="number of data loading workers"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="directory to save checkpoints",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading ImageNet data...")
    train_loader, val_loader = get_imagenet_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ImageNet: 224x224x3 = 150528 input features, 1000 classes
    input_dim = 224 * 224 * 3
    num_classes = 1000

    # Create model
    print(f"Creating {args.num_layers}-layer MLP with hidden dim {args.hidden_dim}...")
    model = DeepMLPWithTracking(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_hidden_layers=args.num_layers - 1,  # -1 because of output layer
        activation=torch.tanh,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create optimizer (only for standard training)
    if not args.use_hessian:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training loop
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        if args.use_hessian:
            train_loss, train_acc, epoch_time, hess_time = train_epoch_hessian(
                model,
                train_loader,
                args.lr,
                device,
                epoch,
                use_hessian_every_n=args.hessian_freq,
            )
            print(
                f"\nTraining - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
                f"Time: {epoch_time:.2f}s, Hessian time: {hess_time:.2f}s"
            )
        else:
            train_loss, train_acc, epoch_time = train_epoch_standard(
                model, train_loader, optimizer, device, epoch
            )
            print(
                f"\nTraining - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
                f"Time: {epoch_time:.2f}s"
            )
            scheduler.step()

        # Validation
        val_loss, val_acc = validate(model, val_loader, device)
        print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_acc": best_acc,
            "args": args,
        }

        torch.save(checkpoint, checkpoint_dir / "checkpoint_latest.pth")
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "checkpoint_best.pth")
            print(f"New best validation accuracy: {best_acc:.2f}%")

    print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
