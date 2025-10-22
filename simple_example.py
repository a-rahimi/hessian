"""
Simple example demonstrating the Hessian-inverse-vector product algorithm.

This script trains a small MLP on CIFAR-10 comparing:
1. Standard gradient descent
2. Hessian-preconditioned gradient descent
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

from hessian_inverse import DeepMLPWithTracking, hessian_inverse_vector_product


def flatten_gradients(model: nn.Module) -> torch.Tensor:
    """Flatten all parameter gradients into a single vector."""
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.flatten())
    return torch.cat(grads)


def unflatten_and_apply_gradients(model: nn.Module, flat_grad: torch.Tensor, lr: float):
    """Unflatten and apply gradients to model parameters."""
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param_grad = flat_grad[offset:offset + numel].view(param.shape)
        param.data.add_(param_grad, alpha=-lr)
        offset += numel


def train_with_standard_gd(model, train_loader, lr, num_epochs, device):
    """Train with standard gradient descent."""
    print("\n" + "="*60)
    print("Training with STANDARD GRADIENT DESCENT")
    print("="*60)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_flat = inputs.view(inputs.size(0), -1)
            
            # Zero gradients
            for param in model.parameters():
                param.grad = None
            
            # Forward pass
            loss = model(inputs_flat, targets, track_activations=False)
            
            # Backward pass
            loss.backward()
            
            # Get gradient
            gradient = flatten_gradients(model)
            
            # Apply gradient
            unflatten_and_apply_gradients(model, gradient, lr)
            
            # Statistics
            total_loss += loss.item()
            with torch.no_grad():
                outputs = model(inputs_flat, track_activations=False)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, Time={epoch_time:.2f}s")


def train_with_hessian_gd(model, train_loader, lr, num_epochs, device):
    """Train with Hessian-preconditioned gradient descent."""
    print("\n" + "="*60)
    print("Training with HESSIAN-PRECONDITIONED GRADIENT DESCENT")
    print("="*60)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        hessian_time = 0.0
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_flat = inputs.view(inputs.size(0), -1)
            
            # Zero gradients
            for param in model.parameters():
                param.grad = None
            
            # Forward pass with tracking
            loss = model(inputs_flat, targets, track_activations=True)
            
            # Backward pass (need create_graph for second derivatives)
            loss.backward(create_graph=True)
            
            # Get gradient
            gradient = flatten_gradients(model)
            
            # Apply Hessian-inverse
            hess_start = time.time()
            try:
                preconditioned_grad = hessian_inverse_vector_product(model, loss, gradient)
                hessian_time += time.time() - hess_start
                
                # Apply preconditioned gradient
                unflatten_and_apply_gradients(model, preconditioned_grad, lr)
            except Exception as e:
                print(f"Warning: Hessian computation failed: {e}")
                # Fall back to standard gradient
                unflatten_and_apply_gradients(model, gradient, lr)
            
            # Statistics
            total_loss += loss.item()
            with torch.no_grad():
                outputs = model(inputs_flat, track_activations=False)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, "
              f"Time={epoch_time:.2f}s (Hessian: {hessian_time:.2f}s)")


def main():
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # CIFAR-10 dataset
    print("\nLoading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Use a subset for quick demonstration
    subset_size = 1000
    trainset = torch.utils.data.Subset(trainset, range(subset_size))
    
    train_loader = DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=2
    )
    
    # Model parameters
    input_dim = 32 * 32 * 3  # CIFAR-10 images
    hidden_dim = 64
    num_classes = 10
    num_layers = 6
    lr = 0.01
    num_epochs = 5
    
    print(f"\nModel: {num_layers} layers, {hidden_dim} hidden units")
    
    # Test 1: Standard GD
    print("\n" + "="*60)
    print("EXPERIMENT 1: Standard Gradient Descent")
    print("="*60)
    
    model1 = DeepMLPWithTracking(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_hidden_layers=num_layers - 1,
        activation='tanh'
    ).to(device)
    
    train_with_standard_gd(model1, train_loader, lr, num_epochs, device)
    
    # Test 2: Hessian-preconditioned GD
    print("\n" + "="*60)
    print("EXPERIMENT 2: Hessian-Preconditioned Gradient Descent")
    print("="*60)
    
    model2 = DeepMLPWithTracking(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_hidden_layers=num_layers - 1,
        activation='tanh'
    ).to(device)
    
    # Copy weights from model1 for fair comparison
    model2.load_state_dict(model1.state_dict())
    
    train_with_hessian_gd(model2, train_loader, lr, num_epochs, device)
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print("\nNote: This is a simple demonstration on a small dataset.")
    print("The Hessian preconditioning should show better convergence properties,")
    print("especially for deeper networks, though the overhead may be significant.")


if __name__ == '__main__':
    main()

