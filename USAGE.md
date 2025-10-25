# Quick Start Guide

## File Structure

This project has a clean separation between library code and executable scripts:

- **Library** (import, don't run): `hessian.py`
- **Test suite** (run to validate): `python3 test_hessian.py`
- **Training script** (run to train): `python3 train_imagenet.py`
- **Demo** (run to see example): `python3 simple_example.py`

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch>=2.0.0 torchvision>=0.15.0 numpy>=1.24.0 scipy>=1.10.0
```

## Running Tests

### Basic Validation
```bash
pytest test_hessian.py -v
```

Or the old way (still works):
```bash
python3 test_hessian.py
```

This runs 10 test cases:
1. **Correctness test**: Verifies H(H^{-1}g) ≈ g on the gradient
2. **Random vector test**: Tests on arbitrary vectors
3. **Scaling tests**: 4 parametrized tests for different depths (2, 4, 8, 12 layers)
4. **Linear scaling test**: Verifies O(L) scaling (marked as "slow")
5. **Stability tests**: 3 parametrized tests for different scales (1e-3, 1.0, 1e3)

Expected output:
```
test_hessian.py::test_hessian_inverse_correctness_on_gradient PASSED
test_hessian.py::test_hessian_inverse_on_random_vector PASSED
test_hessian.py::test_computational_cost_per_depth[2] PASSED
test_hessian.py::test_computational_cost_per_depth[4] PASSED
test_hessian.py::test_computational_cost_per_depth[8] PASSED
test_hessian.py::test_computational_cost_per_depth[12] PASSED
test_hessian.py::test_scaling_is_linear PASSED
test_hessian.py::test_numerical_stability_at_scale[0.001] PASSED
test_hessian.py::test_numerical_stability_at_scale[1.0] PASSED
test_hessian.py::test_numerical_stability_at_scale[1000.0] PASSED
========================= 10 passed in 15.23s =========================
```

### Pytest Advanced Usage

```bash
# Run only fast tests (skip the slow linear scaling test)
pytest test_hessian.py -m "not slow"

# Run only a specific test
pytest test_hessian.py::test_hessian_inverse_correctness_on_gradient

# Run tests matching a pattern
pytest test_hessian.py -k "stability"

# Stop at first failure
pytest test_hessian.py -x

# Show local variables on failures
pytest test_hessian.py -l --tb=long
```

## Simple CIFAR-10 Example

```bash
python3 simple_example.py
```

This trains a 6-layer MLP on a subset of CIFAR-10:
- First with standard gradient descent
- Then with Hessian-preconditioned gradient descent

Expected runtime: ~5-10 minutes on CPU

Output shows:
- Training loss and accuracy for each method
- Time comparison
- Hessian computation overhead

## Training on ImageNet

### Prerequisites
- Download ImageNet dataset (ILSVRC2012)
- Organize as:
  ```
  /path/to/imagenet/
    train/
      n01440764/
      n01443537/
      ...
    val/
      n01440764/
      n01443537/
      ...
  ```

### Standard Training (baseline)
```bash
python3 train_imagenet.py \
    --data-dir /path/to/imagenet \
    --batch-size 256 \
    --epochs 90 \
    --lr 0.1 \
    --hidden-dim 512 \
    --num-layers 20
```

### Hessian-Preconditioned Training
```bash
python3 train_imagenet.py \
    --data-dir /path/to/imagenet \
    --batch-size 64 \
    --epochs 90 \
    --lr 0.01 \
    --hidden-dim 256 \
    --num-layers 20 \
    --use-hessian \
    --hessian-freq 1 \
    --device cuda
```

**Important notes:**
- Use smaller batch size with Hessian (more memory needed for second derivatives)
- Start with narrower networks (smaller `--hidden-dim`)
- The cost scales as O(max(a,p)³) per layer, so wide layers are expensive
- Use `--hessian-freq N` to apply Hessian every N batches if too slow

### Recommended Settings for Different Scenarios

#### Tall-Skinny Network (Best for Hessian method)
```bash
--num-layers 40 --hidden-dim 128 --batch-size 32 --use-hessian
```
Reasoning: Many layers, narrow width → linear scaling helps

#### Short-Fat Network (Not ideal for Hessian)
```bash
--num-layers 5 --hidden-dim 1024 --batch-size 256
```
Reasoning: Few layers, wide width → cubic cost dominates

#### Balanced Network
```bash
--num-layers 20 --hidden-dim 256 --batch-size 64 --use-hessian --hessian-freq 5
```
Reasoning: Apply Hessian periodically to balance cost and benefit

## Interpreting Results

### Test Output
- **Relative error < 1e-3**: Good numerical accuracy
- **Relative error < 1e-1**: Acceptable, may indicate numerical issues
- **Relative error > 1**: Implementation error or numerical instability

### Training Output
Look for:
1. **Faster convergence**: Fewer epochs to reach same accuracy
2. **Better final accuracy**: Especially in later epochs
3. **Overhead acceptable**: Hessian time should be < 10x forward+backward

Example good output:
```
Epoch 1: Loss=2.3000, Acc=10.00%, Time=120.00s (Hessian: 80.00s)
Epoch 2: Loss=1.8000, Acc=25.00%, Time=120.00s (Hessian: 80.00s)
Epoch 3: Loss=1.2000, Acc=45.00%, Time=120.00s (Hessian: 80.00s)
...
```

Compare with standard GD at same epoch - Hessian should show:
- Lower loss
- Higher accuracy
- Smoother convergence (less oscillation)

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` (try 32, 16, or even 8)
- Reduce `--hidden-dim` (network width)
- Use `--hessian-freq N` with N>1
- Use mixed precision training (add to code)

### Numerical Instability (NaN or Inf)
- Check activation functions are smooth (tanh, sigmoid)
- Reduce learning rate
- Add gradient clipping
- Check for degenerate weights (rerun with different initialization)

### Too Slow
- Increase `--hessian-freq` (apply Hessian less often)
- Reduce network width (`--hidden-dim`)
- Use GPU (`--device cuda`)
- Consider approximations (not implemented yet)

### Poor Convergence
- Hessian method may need different learning rate than SGD
- Try lr=0.001 to lr=0.1 range
- Ensure smooth activation functions
- Check that test passes first

## Advanced Usage

### Using in Your Own Code

```python
from hessian import DeepMLPWithTracking, hessian_inverse_vector_product

# 1. Create model
model = DeepMLPWithTracking(
    input_dim=784,
    hidden_dim=128,
    num_classes=10,
    num_hidden_layers=19,
    activation='tanh'  # Must be smooth!
)

# 2. Training loop
for x, y in dataloader:
    # Forward with tracking
    loss = model(x, y, track_activations=True)
    
    # Backward with create_graph=True
    loss.backward(create_graph=True)
    
    # Get gradient
    grad = torch.cat([p.grad.flatten() for p in model.parameters()])
    
    # Compute preconditioned gradient
    try:
        precon_grad = hessian_inverse_vector_product(model, loss, grad)
        
        # Apply update manually
        # (unflatten and apply to parameters)
        
    except Exception as e:
        print(f"Hessian failed: {e}, using standard gradient")
        # Fall back to standard gradient
```

### Extending to Other Architectures

Current implementation supports:
- ✅ Fully-connected layers
- ✅ Smooth activations (tanh, sigmoid, softplus)
- ✅ Sequential architectures

Not yet supported:
- ❌ Convolutional layers
- ❌ Residual connections
- ❌ Batch normalization
- ❌ ReLU (needs smoothing)

To add support, you would need to:
1. Modify `compute_layer_derivatives()` for new layer types
2. Handle non-sequential architectures in matrix construction
3. Implement specialized derivative computations

## Performance Expectations

### Computational Cost
For L layers, hidden dim a, params p per layer:

| Network Type | Layers | Width | Cost per Iteration | Best Method |
|-------------|--------|-------|-------------------|-------------|
| Tall-skinny | 40 | 64 | O(40 × 64³) ≈ 10M | Hessian |
| Balanced | 20 | 256 | O(20 × 256³) ≈ 335M | Either |
| Short-fat | 5 | 1024 | O(5 × 1024³) ≈ 5B | Standard |

### Memory Requirements
- Standard GD: O(Lp) for gradients
- Hessian: O(Lp²) for second derivatives (dominant)
- Activations: O(La) for tracked activations

### When to Use Hessian Method
✅ Use when:
- Deep networks (L ≥ 20)
- Narrow layers (a,p ≤ 512)
- Second-order convergence matters
- Have sufficient GPU memory

❌ Don't use when:
- Shallow networks (L < 10)
- Very wide layers (a,p > 1024)
- Memory constrained
- Need maximum throughput

## Citation

If you use this implementation, please cite:

```bibtex
@article{rahimi2025hessian,
  title={The Hessian of tall-skinny networks is easy to invert},
  author={Rahimi, Ali},
  year={2025}
}
```

## Support

For issues or questions:
1. Check test_hessian.py passes
2. Review this usage guide
3. Check README.md for algorithm details
4. Verify PyTorch >= 2.0.0 installed

