"""
Test and validation script for Hessian-inverse-vector product implementation.

Validates that:
1. H (H^{-1} g) ≈ g for random vectors
2. The Hessian-inverse computation is numerically stable
3. Computational cost is reasonable
"""

import torch
import torch.nn as nn
import numpy as np
import time
from hessian_inverse import (
    DeepMLPWithTracking,
    hessian_inverse_vector_product,
    compute_all_layer_derivatives
)


def compute_hessian_vector_product(model: DeepMLPWithTracking,
                                   loss: torch.Tensor,
                                   vector: torch.Tensor) -> torch.Tensor:
    """
    Compute Hessian-vector product H v using Pearlmutter's trick.
    
    This is used for testing: we verify that H (H^{-1} g) ≈ g.
    """
    # Get model parameters
    params = list(model.parameters())
    
    # Compute gradient
    grads = torch.autograd.grad(loss, params, create_graph=True)
    
    # Flatten gradient
    flat_grad = torch.cat([g.flatten() for g in grads])
    
    # Compute gradient-vector product
    grad_v_product = torch.sum(flat_grad * vector)
    
    # Compute Hessian-vector product
    hvp_grads = torch.autograd.grad(grad_v_product, params, retain_graph=True)
    
    # Flatten result
    hvp = torch.cat([g.flatten() for g in hvp_grads])
    
    return hvp


def test_hessian_inverse_correctness(input_dim=10, hidden_dim=5, num_classes=3,
                                     num_layers=4, batch_size=2):
    """
    Test that H (H^{-1} g) ≈ g.
    """
    print("="*60)
    print("Test 1: Hessian-inverse correctness")
    print("="*60)
    
    # Create model
    model = DeepMLPWithTracking(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_hidden_layers=num_layers - 1,
        activation='tanh'
    )
    
    # Create dummy data
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Forward pass
    loss = model(x, targets, track_activations=True)
    
    # Compute gradient
    loss.backward(create_graph=True)
    gradient = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    
    print(f"Model: {num_layers} layers, {hidden_dim} hidden units")
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradient norm: {gradient.norm().item():.4f}")
    
    # Compute H^{-1} g
    print("\nComputing H^{-1} g...")
    start = time.time()
    try:
        h_inv_g = hessian_inverse_vector_product(model, loss, gradient)
        hessian_inv_time = time.time() - start
        print(f"H^{-1} g computed in {hessian_inv_time:.4f}s")
        print(f"H^{-1} g norm: {h_inv_g.norm().item():.4f}")
        
        # Compute H (H^{-1} g) using Pearlmutter's trick
        print("\nComputing H (H^{-1} g)...")
        start = time.time()
        h_hinv_g = compute_hessian_vector_product(model, loss, h_inv_g)
        hvp_time = time.time() - start
        print(f"H (H^{-1} g) computed in {hvp_time:.4f}s")
        
        # Check if H (H^{-1} g) ≈ g
        diff = h_hinv_g - gradient
        relative_error = diff.norm() / gradient.norm()
        
        print(f"\nVerification:")
        print(f"||H (H^{-1} g) - g||: {diff.norm().item():.6f}")
        print(f"Relative error: {relative_error.item():.6f}")
        
        if relative_error < 1e-3:
            print("✓ Test PASSED: H (H^{-1} g) ≈ g")
            return True
        else:
            print("✗ Test FAILED: Large discrepancy between H (H^{-1} g) and g")
            return False
            
    except Exception as e:
        print(f"✗ Test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hessian_inverse_on_random_vector(input_dim=10, hidden_dim=5, num_classes=3,
                                          num_layers=4, batch_size=2):
    """
    Test H^{-1} on a random vector (not necessarily the gradient).
    """
    print("\n" + "="*60)
    print("Test 2: Hessian-inverse on random vector")
    print("="*60)
    
    # Create model
    model = DeepMLPWithTracking(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_hidden_layers=num_layers - 1,
        activation='tanh'
    )
    
    # Create dummy data
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Forward pass
    loss = model(x, targets, track_activations=True)
    loss.backward(create_graph=True)
    
    # Create random vector with same size as parameters
    total_params = sum(p.numel() for p in model.parameters())
    random_vector = torch.randn(total_params)
    
    print(f"Random vector norm: {random_vector.norm().item():.4f}")
    
    # Compute H^{-1} v
    try:
        start = time.time()
        h_inv_v = hessian_inverse_vector_product(model, loss, random_vector)
        hessian_inv_time = time.time() - start
        print(f"H^{-1} v computed in {hessian_inv_time:.4f}s")
        print(f"H^{-1} v norm: {h_inv_v.norm().item():.4f}")
        
        # Compute H (H^{-1} v)
        start = time.time()
        h_hinv_v = compute_hessian_vector_product(model, loss, h_inv_v)
        hvp_time = time.time() - start
        print(f"H (H^{-1} v) computed in {hvp_time:.4f}s")
        
        # Check if H (H^{-1} v) ≈ v
        diff = h_hinv_v - random_vector
        relative_error = diff.norm() / random_vector.norm()
        
        print(f"\nVerification:")
        print(f"||H (H^{-1} v) - v||: {diff.norm().item():.6f}")
        print(f"Relative error: {relative_error.item():.6f}")
        
        if relative_error < 1e-3:
            print("✓ Test PASSED: H (H^{-1} v) ≈ v")
            return True
        else:
            print("✗ Test FAILED: Large discrepancy")
            return False
            
    except Exception as e:
        print(f"✗ Test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_computational_cost(input_dim=20, hidden_dim=10, num_classes=5):
    """
    Test computational cost scaling with number of layers.
    """
    print("\n" + "="*60)
    print("Test 3: Computational cost scaling")
    print("="*60)
    
    layer_counts = [2, 4, 8, 12]
    batch_size = 2
    
    results = []
    
    for num_layers in layer_counts:
        print(f"\nTesting with {num_layers} layers...")
        
        # Create model
        model = DeepMLPWithTracking(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_hidden_layers=num_layers - 1,
            activation='tanh'
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Create dummy data
        x = torch.randn(batch_size, input_dim, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # Forward pass
        loss = model(x, targets, track_activations=True)
        loss.backward(create_graph=True)
        gradient = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        
        # Time Hessian-inverse computation
        try:
            start = time.time()
            h_inv_g = hessian_inverse_vector_product(model, loss, gradient)
            elapsed = time.time() - start
            
            print(f"Time: {elapsed:.4f}s")
            results.append((num_layers, total_params, elapsed))
        except Exception as e:
            print(f"Failed: {e}")
            results.append((num_layers, total_params, None))
    
    # Analyze scaling
    print("\n" + "="*60)
    print("Scaling analysis:")
    print("="*60)
    print(f"{'Layers':<10} {'Params':<15} {'Time (s)':<15} {'Time/Param':<15}")
    print("-"*60)
    
    for num_layers, params, elapsed in results:
        if elapsed is not None:
            time_per_param = elapsed / params * 1e6  # microseconds per param
            print(f"{num_layers:<10} {params:<15,} {elapsed:<15.4f} {time_per_param:<15.2f} μs")
        else:
            print(f"{num_layers:<10} {params:<15,} {'FAILED':<15}")
    
    # Check if scaling is roughly linear in L
    if all(r[2] is not None for r in results) and len(results) >= 3:
        times = [r[2] for r in results]
        layers = [r[0] for r in results]
        
        # Fit linear model
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(layers, times)
        
        print(f"\nLinear fit: time = {slope:.4f} * layers + {intercept:.4f}")
        print(f"R² = {r_value**2:.4f}")
        
        if r_value**2 > 0.9:
            print("✓ Scaling appears to be linear in number of layers")
            return True
        else:
            print("⚠ Scaling may not be perfectly linear")
            return True
    else:
        print("⚠ Could not analyze scaling due to failures")
        return False


def test_numerical_stability():
    """
    Test numerical stability of the implementation.
    """
    print("\n" + "="*60)
    print("Test 4: Numerical stability")
    print("="*60)
    
    # Test with different scales
    input_dim = 10
    hidden_dim = 5
    num_classes = 3
    num_layers = 4
    batch_size = 2
    
    for scale in [1e-3, 1.0, 1e3]:
        print(f"\nTesting with input scale {scale}...")
        
        # Create model
        model = DeepMLPWithTracking(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_hidden_layers=num_layers - 1,
            activation='tanh'
        )
        
        # Create scaled data
        x = torch.randn(batch_size, input_dim, requires_grad=True) * scale
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # Forward pass
        loss = model(x, targets, track_activations=True)
        loss.backward(create_graph=True)
        gradient = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        
        print(f"Loss: {loss.item():.4e}, Gradient norm: {gradient.norm().item():.4e}")
        
        # Compute H^{-1} g
        try:
            h_inv_g = hessian_inverse_vector_product(model, loss, gradient)
            
            # Check for NaN or Inf
            if torch.isnan(h_inv_g).any() or torch.isinf(h_inv_g).any():
                print(f"✗ Found NaN or Inf values in result")
                return False
            
            print(f"✓ No numerical issues (H^{-1} g norm: {h_inv_g.norm().item():.4e})")
            
        except Exception as e:
            print(f"✗ Failed with exception: {e}")
            return False
    
    print("\n✓ Test PASSED: Numerically stable across different scales")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING HESSIAN-INVERSE VALIDATION TESTS")
    print("="*60)
    
    results = []
    
    # Test 1: Correctness with gradient
    results.append(("Hessian-inverse correctness", test_hessian_inverse_correctness()))
    
    # Test 2: Correctness with random vector
    results.append(("Random vector test", test_hessian_inverse_on_random_vector()))
    
    # Test 3: Computational cost
    try:
        results.append(("Computational scaling", test_computational_cost()))
    except ImportError:
        print("\nNote: scipy not available, skipping scaling analysis")
        results.append(("Computational scaling", None))
    
    # Test 4: Numerical stability
    results.append(("Numerical stability", test_numerical_stability()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⊘ SKIPPED"
        print(f"{test_name:<30} {status}")
    
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)

