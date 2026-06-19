# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project implementing efficient Hessian-inverse-vector product computation for deep neural networks. The algorithm achieves linear time and space complexity (similar to Pearlmutter's algorithm) rather than the cubic scaling of naive approaches. See `hessian.pdf` for full mathematical documentation.

## Development Commands

### Python Environment
Always use the virtual environment at `/Users/ali/hessian/.venv`:
```bash
source /Users/ali/hessian/.venv/bin/activate
```

Or use the interpreter directly:
```bash
/Users/ali/hessian/.venv/bin/python <script>
```

### Running Tests
```bash
# Run all tests
pytest src/

# Run specific test file
pytest src/test_hessian.py

# Run specific test
pytest src/test_hessian.py::test_function_name
```

### Running Examples
```bash
# Simple CIFAR-10 demonstration
python src/simple_example.py

# ImageNet training (requires ImageNet dataset)
python src/train_imagenet.py --data-dir /path/to/imagenet
```

## Code Architecture

### Core Implementation (`src/hessian.py`)

**BlockWithMixedDerivatives**: Abstract base class for network layers that can compute mixed partial derivatives. Key methods:
- `forward()`: Caches inputs/outputs for derivative computation
- `derivatives()`: Computes first and second-order derivatives using PyTorch functorch

**LayerDerivatives**: Named tuple storing:
- `Dx`, `Dz`: First-order derivatives w.r.t. parameters and inputs
- `DD_Dxx`, `DD_Dzx`, `DM_Dzz`: Second-order derivatives pre-multiplied by diagonal matrices (avoids storing full Hessian blocks)

**SequenceOfBlocks**: Neural network as a sequence of blocks. Key methods:
- `derivatives()`: Computes block-diagonal derivative matrices for all layers
- `hessian_vector_product()`: Implements Eq. from paper using block matrix operations
- `hessian_inverse_product()`: Core algorithm for computing H^(-1) @ g efficiently

The Hessian inversion leverages the network's sequential structure to avoid explicit Hessian materialization. The algorithm exploits block structure where each layer contributes diagonal blocks.

### Block Partitioned Matrices (`src/block_partitioned_matrices.py`)

Hierarchical matrix library supporting:
- **Vertical**: Vertically stacked blocks
- **Diagonal**: Block diagonal matrices
- **UpperBiDiagonal/LowerBiDiagonal**: Bi-diagonal block matrices
- **Symmetric2x2**: Symmetric 2x2 block matrices
- **Identity/Zero**: Placeholder matrices (no storage/compute)

Blocks can be nested (e.g., a Symmetric2x2 where block11 is itself a Diagonal). All types support:
- Matrix multiplication (`@` operator)
- Linear system solving (`.solve()`)
- Inversion (`.invert()`)
- Conversion to dense tensors (`.to_tensor()`)

This abstraction allows expressing complex linear algebra operations on structured matrices without materializing full dense representations.

### Example Scripts

**`src/simple_example.py`**: Demonstrates standard vs. Hessian-preconditioned gradient descent on CIFAR-10 subset. Compares convergence properties.

**`src/train_imagenet.py`**: Production training script with ImageNet data loading, preprocessing, and Hessian-preconditioned optimization.

## Coding Conventions

### Comments
Avoid redundant comments that simply restate code. Comments should add context or explain non-obvious logic:
```python
# BAD: Assign b to a
a = b

# GOOD: Save b before modification so we can restore it later
a = b
```

### Testing
- Tests run via pytest (no `if __name__ == "__main__"` blocks in test files)
- Use pytest fixtures for shared test setup/data
- Test files: `test_*.py` with test functions named `test_*`

## Dependencies

See `src/requirements.txt`:
- PyTorch >= 2.0.0 (uses `torch.func` for automatic differentiation)
- torchvision >= 0.15.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- pytest >= 7.0.0
