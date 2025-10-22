# Hessian-Inverse-Vector Product for Deep Networks

Implementation of the efficient Hessian-inverse-vector product algorithm described in "The Hessian of tall-skinny networks is easy to invert" by Ali Rahimi.

## Overview

This implementation provides a way to compute Hessian-inverse-vector products `H^{-1}g` for deep neural networks in time and space that scales **linearly** with the number of layers `L`, rather than the cubic scaling of naive approaches.

For an L-layer network with `p` parameters per layer and `a` activations per layer:
- **Naive approach**: `O(L³p³)` operations and `O(L²p²)` memory
- **This algorithm**: `O(L max(a,p)³)` operations and `O(L max(a,p)²)` memory

This makes the algorithm particularly efficient for "tall-skinny" networks (many layers, moderate width).

## Key Features

- ✅ Exact Hessian-inverse-vector products (not approximations)
- ✅ Linear scaling in network depth
- ✅ Second-order optimization without storing full Hessian
- ✅ Complete implementation with PyTorch autograd
- ✅ Training loops for ImageNet and CIFAR-10
- ✅ Comprehensive test suite

## Files

- `hessian_inverse.py`: Core implementation of the algorithm
  - `DeepMLPWithTracking`: MLP model with activation tracking
  - `compute_layer_derivatives()`: Computes mixed partial derivatives
  - `hessian_inverse_vector_product()`: Main algorithm (Section 5.1 of paper)

- `train_imagenet.py`: Training script for ImageNet with Hessian preconditioning
- `test_hessian.py`: Validation tests to verify correctness
- `simple_example.py`: Quick demonstration on CIFAR-10
- `requirements.txt`: Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- scipy (for tests)

## Usage

### Quick Example (CIFAR-10)

```bash
python simple_example.py
```

This trains a small MLP on CIFAR-10 comparing standard gradient descent vs. Hessian-preconditioned gradient descent.

### Running Tests

```bash
python test_hessian.py
```

Tests verify:
1. **Correctness**: `H(H^{-1}g) ≈ g`
2. **Random vectors**: Works on arbitrary vectors, not just gradients
3. **Computational cost**: Confirms linear scaling with depth
4. **Numerical stability**: Handles different input scales

### Training on ImageNet

```bash
python train_imagenet.py \
    --data-dir /path/to/imagenet \
    --batch-size 256 \
    --epochs 90 \
    --lr 0.1 \
    --hidden-dim 512 \
    --num-layers 20 \
    --use-hessian \
    --device cuda
```

Options:
- `--use-hessian`: Enable Hessian preconditioning (default: standard SGD)
- `--hessian-freq N`: Apply Hessian every N batches (default: 1)
- `--hidden-dim`: Width of hidden layers
- `--num-layers`: Total number of layers (including output)

### Using in Your Own Code

```python
from hessian_inverse import DeepMLPWithTracking, hessian_inverse_vector_product

# Create model
model = DeepMLPWithTracking(
    input_dim=784,
    hidden_dim=256,
    num_classes=10,
    num_hidden_layers=19,
    activation='tanh'
)

# Forward pass (must track activations)
loss = model(x, targets, track_activations=True)

# Compute gradient
loss.backward(create_graph=True)  # Need create_graph for second derivatives
gradient = torch.cat([p.grad.flatten() for p in model.parameters()])

# Compute H^{-1} g (preconditioned gradient)
preconditioned_gradient = hessian_inverse_vector_product(model, loss, gradient)

# Apply update
# ... (unflatten and apply to parameters)
```

## Algorithm Details

The implementation follows Section 5.1 of the paper (lines 670-741):

### Step 1: Compute auxiliary vector
```
g' = [D_x; I] g
```
Cost: `O(Lap)` operations

### Step 2: Form matrix A
```
A = M̂ Q^{-1} M̂^T + [[D_x D_x^T, D_x], [D_x^T, I]]
```

Where `Q^{-1}` is computed using the partitioned inverse formula:
- `Q = [[P^T D_M D_{zz} P, P^T D_M D_{xz}], [D_D D_{zx} P, D_D D_{xx} - I]]`
- Compute blocks using Schur complement

Cost: `O(12L max(a,p)³)` operations

### Step 3: Apply A^{-1} using LDL^T decomposition
```
A = [[I, A_12 A_22^{-1}], [0, I]] [[schur, 0], [0, A_22]] [[I, A_12 A_22^{-1}], [0, I]]^T
```

Cost: `O(15L max(a,p)³)` operations

### Step 4: Compute final result
```
y = g - [D_x; I]^T g''
```

### Total Cost
- **Operations**: `O(16L max(a,p)³)`
- **Memory**: `O(L max(a,p)²)`

Linear scaling in `L` makes this efficient for deep networks!

## Implementation Notes

### Mixed Partial Derivatives

The algorithm requires computing six types of derivatives for each layer `f_ℓ(z_{ℓ-1}; x_ℓ)`:

- `∇_x f_ℓ`: gradient w.r.t. parameters (a × p)
- `∇_z f_ℓ`: gradient w.r.t. inputs (a × a)
- `∇_{xx} f_ℓ`: Hessian w.r.t. parameters (ap × p)
- `∇_{zx} f_ℓ`: mixed partial (ap × a)
- `∇_{xz} f_ℓ`: mixed partial (a² × p)
- `∇_{zz} f_ℓ`: Hessian w.r.t. inputs (a² × a)

These are computed using PyTorch's autograd with `create_graph=True`.

### Block-Structured Matrices

Key matrices are block-structured and stored efficiently:

- **Block-diagonal** (D_x, D_{xx}, etc.): Stored as lists of blocks
- **Block bi-diagonal** (M): Two lists (diagonal and lower diagonal)
- Operations use block structure to avoid dense matrix formation

### Activation Functions

The paper assumes differentiable activations. We use:
- `tanh`: Smooth and twice-differentiable
- `sigmoid`: Alternative smooth activation
- `softplus`: Smooth approximation to ReLU

ReLU would require special handling of non-differentiable points.

## Comparison: Standard vs. Hessian-Preconditioned GD

| Property | Standard GD | Hessian-Preconditioned GD |
|----------|-------------|---------------------------|
| Convergence rate | Linear | Quadratic (near optimum) |
| Per-iteration cost | `O(Lp)` | `O(L max(a,p)³)` |
| Memory | `O(Lp)` | `O(L max(a,p)²)` |
| Hyperparameters | Learning rate | Learning rate |
| Best for | Shallow/wide networks | Deep/narrow networks |

## Limitations

1. **Computational overhead**: `O(max(a,p)³)` per layer can be expensive for wide layers
2. **Memory**: Requires storing activations and intermediate derivatives
3. **Smooth activations**: ReLU requires approximation or subgradients
4. **Batch size**: Second derivatives can be memory-intensive; use small batches
5. **Implementation**: Currently uses some dense matrices (can be optimized)

## Future Improvements

- [ ] More efficient block-structured operations (avoid dense conversion)
- [ ] GPU optimizations for block operations
- [ ] Support for convolutional layers
- [ ] Adaptive Hessian frequency based on convergence
- [ ] Mixed precision support
- [ ] Distributed training support

## References

Ali Rahimi. "The Hessian of tall-skinny networks is easy to invert." 2025.

Related work:
- Pearlmutter (1994): Fast exact multiplication by the Hessian
- Martens & Grosse (2015): Optimizing neural networks with Kronecker-factored approximate curvature
- Botev et al. (2017): Practical Gauss-Newton optimisation for deep learning

## Citation

If you use this code, please cite:

```bibtex
@article{rahimi2025hessian,
  title={The Hessian of tall-skinny networks is easy to invert},
  author={Rahimi, Ali},
  year={2025}
}
```

## License

[Add appropriate license]

## Contact

[Add contact information]

