"""
Test suite for Hessian-inverse-vector product implementation using pytest.

Run with:
    pytest test_hessian.py              # All tests
    pytest test_hessian.py -v           # Verbose
    pytest test_hessian.py -k scaling   # Only scaling tests
    pytest test_hessian.py --tb=short   # Shorter tracebacks

Validates that:
1. H (H^{-1} g) ≈ g for random vectors
2. The Hessian-inverse computation is numerically stable
3. Computational cost scales linearly with depth
"""

import pytest
import torch
import time
from scipy import stats
from hessian import (
    DeepMLPWithTracking,
    hessian_inverse_vector_product,
)


def compute_hessian_vector_product(
    model: DeepMLPWithTracking, loss: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    """
    Compute Hessian-vector product H v.

    First computes the gradient-vector product g v, then computs the gradient of
    that with respect to the parameters.
    """
    # Get model parameters
    params = list(model.parameters())

    # Compute gradient
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # Flatten gradient
    flat_grad = torch.cat([g.flatten() for g in grads])

    # Compute gradient-vector product
    # TODO: Use @ instead sum(x * y)
    grad_v_product = torch.sum(flat_grad * vector)

    # Compute Hessian-vector product
    hvp_grads = torch.autograd.grad(grad_v_product, params, retain_graph=True)

    # Flatten result
    hvp = torch.cat([g.flatten() for g in hvp_grads])

    return hvp


def test_hessian_inverse_correctness_on_gradient():
    """
    Test that H(H^{-1}g) ≈ g on the gradient vector.

    This is the fundamental property: applying the Hessian-inverse
    and then the Hessian should recover the original vector.
    """
    # Model parameters
    input_dim = 10
    hidden_dim = 5
    num_classes = 3
    num_layers = 4
    batch_size = 2

    # Create model
    model = DeepMLPWithTracking(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_hidden_layers=num_layers - 1,
        activation=torch.tanh,
    )

    # Create dummy data
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Forward pass
    loss = model(x, targets, track_activations=True)

    # Compute gradient
    loss.backward(create_graph=True)
    gradient = torch.cat(
        [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    )

    # Compute H^{-1} g
    h_inv_g = hessian_inverse_vector_product(model, loss, gradient)

    # Compute H (H^{-1} g) using Pearlmutter's trick
    h_hinv_g = compute_hessian_vector_product(model, loss, h_inv_g)

    # Check if H (H^{-1} g) ≈ g
    diff = h_hinv_g - gradient
    relative_error = (diff.norm() / gradient.norm()).item()

    # Assert with informative message
    assert relative_error < 1e-3, (
        f"H(H^{{-1}}g) does not match g: "
        f"relative error = {relative_error:.6f} (threshold: 1e-3)"
    )


def test_hessian_inverse_on_random_vector():
    """
    Test that H(H^{-1}v) ≈ v for a random vector v.

    This verifies that the Hessian-inverse works on arbitrary vectors,
    not just gradients. The Hessian-inverse is a linear operator, so
    it should work correctly on any vector.
    """
    # Model parameters
    input_dim = 10
    hidden_dim = 5
    num_classes = 3
    num_layers = 4
    batch_size = 2

    # Create model
    model = DeepMLPWithTracking(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_hidden_layers=num_layers - 1,
        activation=torch.tanh,
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

    # Compute H^{-1} v
    h_inv_v = hessian_inverse_vector_product(model, loss, random_vector)

    # Compute H (H^{-1} v)
    h_hinv_v = compute_hessian_vector_product(model, loss, h_inv_v)

    # Check if H (H^{-1} v) ≈ v
    diff = h_hinv_v - random_vector
    relative_error = (diff.norm() / random_vector.norm()).item()

    # Assert with informative message
    assert relative_error < 1e-3, (
        f"H(H^{{-1}}v) does not match v for random vector: "
        f"relative error = {relative_error:.6f} (threshold: 1e-3)"
    )


@pytest.mark.parametrize("num_layers", [2, 4, 8, 12])
def test_computational_cost_per_depth(num_layers):
    """
    Test that computational cost scales reasonably with number of layers.

    This test runs for different network depths to measure the actual cost.
    A separate test analyzes the overall scaling trend.
    """
    input_dim = 20
    hidden_dim = 10
    num_classes = 5
    batch_size = 2

    # Create model
    model = DeepMLPWithTracking(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_hidden_layers=num_layers - 1,
        activation=torch.tanh,
    )

    total_params = sum(p.numel() for p in model.parameters())

    # Create dummy data
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Forward pass
    loss = model(x, targets, track_activations=True)
    loss.backward(create_graph=True)
    gradient = torch.cat(
        [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    )

    # Time Hessian-inverse computation
    start = time.time()
    h_inv_g = hessian_inverse_vector_product(model, loss, gradient)
    elapsed = time.time() - start

    # Just verify it completes in reasonable time (< 10s for small models)
    assert elapsed < 10.0, (
        f"Computation took too long: {elapsed:.2f}s for {num_layers} layers "
        f"({total_params:,} params)"
    )


@pytest.mark.slow
def test_scaling_is_linear():
    """
    Test that computational cost scales linearly with number of layers.

    This test runs multiple models and fits a linear regression to verify
    O(L) scaling as claimed in the paper.
    """
    input_dim = 20
    hidden_dim = 10
    num_classes = 5
    batch_size = 2
    layer_counts = [2, 4, 8, 12]

    results = []

    for num_layers in layer_counts:
        # Create model
        model = DeepMLPWithTracking(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_hidden_layers=num_layers - 1,
            activation=torch.tanh,
        )

        # Create dummy data
        x = torch.randn(batch_size, input_dim, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size,))

        # Forward pass
        loss = model(x, targets, track_activations=True)
        loss.backward(create_graph=True)
        gradient = torch.cat(
            [p.grad.flatten() for p in model.parameters() if p.grad is not None]
        )

        # Time Hessian-inverse computation
        start = time.time()
        h_inv_g = hessian_inverse_vector_product(model, loss, gradient)
        elapsed = time.time() - start

        results.append((num_layers, elapsed))

    # Fit linear model
    layers = [r[0] for r in results]
    times = [r[1] for r in results]

    slope, intercept, r_value, p_value, std_err = stats.linregress(layers, times)
    r_squared = r_value**2

    # Assert that scaling is approximately linear (R² > 0.8)
    # Not requiring perfect linearity due to small sample and overhead
    assert r_squared > 0.8, (
        f"Scaling does not appear linear: R² = {r_squared:.4f} "
        f"(fit: time = {slope:.4f} * layers + {intercept:.4f})"
    )


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_numerical_stability_at_scale(scale):
    """
    Test numerical stability across different input scales.

    The implementation should handle inputs at different scales without
    producing NaN or Inf values.
    """
    # Test with different scales
    input_dim = 10
    hidden_dim = 5
    num_classes = 3
    num_layers = 4
    batch_size = 2

    # Create model
    model = DeepMLPWithTracking(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_hidden_layers=num_layers - 1,
        activation=torch.tanh,
    )

    # Create scaled data
    x = torch.randn(batch_size, input_dim, requires_grad=True) * scale
    targets = torch.randint(0, num_classes, (batch_size,))

    # Forward pass
    loss = model(x, targets, track_activations=True)
    loss.backward(create_graph=True)
    gradient = torch.cat(
        [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    )

    # Compute H^{-1} g
    h_inv_g = hessian_inverse_vector_product(model, loss, gradient)

    # Check for NaN or Inf
    assert not torch.isnan(
        h_inv_g
    ).any(), f"Found NaN values in H^{{-1}}g at scale {scale}"
    assert not torch.isinf(
        h_inv_g
    ).any(), f"Found Inf values in H^{{-1}}g at scale {scale}"

    # Also verify the result is finite and reasonable
    assert h_inv_g.norm().isfinite(), f"H^{{-1}}g norm is not finite at scale {scale}"
