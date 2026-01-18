"""
Implementation of the Hessian-inverse-vector product algorithm
as described in "The Hessian of tall-skinny networks is easy to invert"
"""

from typing import Callable, Iterator, Iterable, NamedTuple, Sequence
import numpy as np
import torch
import torch.func as TF
import torch.nn as nn
import torch.nn.functional as F
import contextlib

import block_partitioned_matrices as bpm


def reshape_starting(v: torch.Tensor, starting_shape: tuple[int, ...]):
    if v.shape[: len(starting_shape)] != starting_shape:
        raise ValueError(
            f"Shape {v.shape} does not match starting shape {starting_shape}"
        )
    return v.reshape((*starting_shape, -1))


def reshape_pytree(pytree, starting_shape: tuple[int, ...]):
    return torch.cat(
        [reshape_starting(v, starting_shape) for v in pytree.values()],
        dim=-1,
    )


def flatten_2d_pytree(pytree):
    full_hessian = []
    param_names = list(pytree.keys())
    for param_name in param_names:
        ndims = pytree[param_name][param_name].ndim // 2
        row = [
            block.reshape(np.prod(block.shape[:ndims]), np.prod(block.shape[ndims:]))
            for block in pytree[param_name].values()
        ]
        full_hessian.append(torch.cat(row, dim=1))
    return torch.cat(full_hessian, dim=0)


class LayerDerivatives(NamedTuple):
    """Stores first and second-order derivatives for a single layer."""

    # First-order derivatives
    Dx: torch.Tensor  # ∇_x f_ℓ: gradient w.r.t. parameters (a × p)
    Dz: torch.Tensor  # ∇_z f_ℓ: gradient w.r.t. inputs (a × a)

    # Second-order derivatives premultiplied by D_D or D_M
    # These avoid storing the full Hessian blocks since they're always
    # used after multiplication by D_D = diag(I ⊗ b_ℓ) or D_M = diag(I ⊗ b_ℓ)
    DD_Dxx: torch.Tensor  # D_D ∇_{xx} f_ℓ: (ap × p), scaled by b_ℓ
    DD_Dzx: torch.Tensor  # D_D ∇_{zx} f_ℓ: (ap × a), scaled by b_ℓ
    DM_Dzz: torch.Tensor  # D_M ∇_{zz} f_ℓ: (a² × a), scaled by b_ℓ


class BlockWithMixedDerivatives(nn.Module):
    "An abstract layer for which various partial derivatives can be computed."

    def __init__(self):
        super().__init__()
        # Caches for the input and output of the layer
        self.input = None
        self.output = None

    def naked_forward(self, *args) -> torch.Tensor:
        """Forward without caching the input and output.

        Must be implemented by subclasses."""
        raise NotImplementedError

    def forward(self, z_in: torch.Tensor, *args) -> torch.Tensor:
        # Cache the input and output for use in derivatives().
        self.input = z_in
        self.output = self.naked_forward(z_in, *args)
        return self.output

    def derivatives(self, dloss_dz: torch.Tensor, *forward_args) -> LayerDerivatives:
        dloss_dz = dloss_dz.flatten()
        if dloss_dz.shape != (self.output.numel(),):
            raise ValueError(
                f"dloss_dz must have {self.output.numel()} elements after flattening, got {dloss_dz.shape[0]}"
            )

        # Flatten activations over their batch dimension to represent
        # derivatives.  But we need to remember their shapes to call the layer,
        # since layers have a notion of a batch.
        input_shape = self.input.shape
        z_in = self.input.flatten()
        params = dict(self.named_parameters())

        def f(x, z):
            return TF.functional_call(self, x, (z.reshape(input_shape), *forward_args))

        def dloss_dz_f(x, z):
            return dloss_dz.flatten() @ f(x, z).flatten()

        return LayerDerivatives(
            Dx=reshape_pytree(
                TF.jacrev(lambda x: f(x, z_in))(params),
                starting_shape=self.output.shape,
            ).reshape(self.output.numel(), -1),
            Dz=torch.func.jacrev(lambda z: f(params, z))(z_in).reshape(
                self.output.numel(), -1
            ),
            DD_Dxx=flatten_2d_pytree(TF.hessian(lambda x: dloss_dz_f(x, z_in))(params)),
            DD_Dzx=torch.func.jacrev(
                lambda z_in: reshape_pytree(
                    TF.jacrev(lambda x: dloss_dz_f(x, z_in))(params),
                    starting_shape=(),
                ),
            )(z_in).reshape(-1, self.input.numel()),
            DM_Dzz=TF.hessian(lambda z: dloss_dz_f(params, z))(z_in),
        )


class DenseBlock(BlockWithMixedDerivatives):
    "A linear layer."

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation

    def naked_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


class LossLayer(DenseBlock):
    """Final layer that fuses the last linear layer with the loss computation."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__(input_dim, num_classes, nn.Identity())

    def naked_forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(super().naked_forward(x), targets)

    def derivatives(
        self, dloss_dz: torch.Tensor, targets: torch.Tensor
    ) -> LayerDerivatives:
        if targets.dtype != torch.int64:
            raise ValueError("For the loss layer, targets must be an integer tensor")

        # Ignore dloss_dz. For the loss layer, it's always ∂z_L/∂z_L = 1.
        derivs = super().derivatives(torch.tensor(1.0), targets)

        # Sanity check the shapes of the derivatives.
        dim_out = self.output.numel()
        dim_in = self.input.numel()
        dim_params = sum(p.numel() for p in self.parameters())
        assert dim_out == 1
        assert derivs.Dx.shape == (1, dim_params)
        assert derivs.Dz.shape == (1, dim_in)
        assert derivs.DD_Dxx.shape == (dim_params, dim_params)
        assert derivs.DD_Dzx.shape == (dim_params, dim_in)
        assert derivs.DM_Dzz.shape == (dim_in, dim_in)

        # Fixup the shape of the derivatives. Some of these are 1D tensors
        # because the loss is scalar. But we need them all to be 2D tensors.
        return LayerDerivatives(
            Dx=derivs.Dx,
            Dz=derivs.Dz,
            DD_Dxx=derivs.DD_Dxx,
            DD_Dzx=derivs.DD_Dzx,
            DM_Dzz=derivs.DM_Dzz,
        )


def _validate_vector_is_Hessian_shaped(b: bpm.Vertical, Dx: bpm.Diagonal):
    if b.num_blocks() != Dx.num_blocks():
        raise ValueError(
            "b must have as many Vertical blocks as there are layers. "
            f"It has {b.num_blocks()}"
        )
    for b_layer, Dx_layer in zip(b.flat, Dx.flatten()):
        if b_layer.height != Dx_layer.width:
            raise ValueError(
                f"b_layer has height {b_layer.height} "
                f"but the layer has width {Dx_layer.width} parameters"
            )


class SequenceOfBlocks(nn.Module):
    "A sequence of blocks for which mixed derivatives can be computed."

    def __init__(
        self, layers: Sequence[BlockWithMixedDerivatives], loss_layer: LossLayer
    ):
        super().__init__()
        self.layers = nn.Sequential(*layers)
        self.loss_layer = loss_layer

    def __iter__(self) -> Iterator[nn.Module]:
        yield from self.layers
        yield self.loss_layer

    @contextlib.contextmanager
    def save_dloss_douts(self):
        """Record the derivative of the loss wrt the output of each layer.

        For each layer, the derivative of the pipeline's loss wrt to the layer's output is recorded
        in the layer's dloss_dout field.
        """

        def hook(
            module: nn.Module,
            grad_input: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            (module.dloss_dout,) = grad_output  # Ensure layer has exactly one output.

        callbacks = [layer.register_full_backward_hook(hook) for layer in self.layers]
        callbacks.append(self.loss_layer.register_full_backward_hook(hook))

        yield

        for callback in callbacks:
            callback.remove()

    def derivatives(
        self, z_in: torch.Tensor, target: torch.Tensor
    ) -> Iterable[bpm.Diagonal]:
        """
        Compute the derivatives of the loss with respect to the inputs and parameters of the layers.

        Since the model is a chain, all these derivatives has a block-diagonal structure.

        Args:
            z_in: The input to the model.
            target: The target output of the model. Used to compute the loss.

        Returns a Dx, Dz, DD_Dxx, DD_Dzx, DM_Dzz, all block-diagonal matrices,
        one blockper layer in the network.
        """
        with self.save_dloss_douts():
            # Populate the input and output caches of the layers and ∂z_L/∂z_ℓ for each layer ℓ.
            self(z_in, target).backward()

        return map(
            bpm.Diagonal,
            zip(
                *(
                    [layer.derivatives(layer.dloss_dout) for layer in self.layers]
                    + [self.loss_layer.derivatives(None, target)]
                )
            ),
        )

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_layer(self.layers(x), targets)

    def hessian_vector_product(
        self, z_in: torch.Tensor, target: torch.Tensor, v: bpm.Vertical
    ) -> bpm.Vertical:
        """Multiply the pipeline's Hessian by a given vector.

        Implements equation \ref{eq:hessian} from hessian.tex.  Turns out to be
        equivalent to writing Plearlmutter's algortihm using explicit matrix
        operations instead of backprop operations.
        """
        Dx, Dz, DD_Dxx, DD_Dzx, DM_Dzz = self.derivatives(z_in, target)
        M = bpm.IdentityWithLowerDiagonal((-Dz).flat[1:])
        P = bpm.downshifting_matrix(z_in.numel(), [b.shape[0] for b in Dx.flatten()])

        _validate_vector_is_Hessian_shaped(v, Dx)

        # Compute equation \ref{eq:hessian} from hessian.tex:
        # H v = D_D D_xx v + D_D D_zx P M⁻¹ Dₓ v
        #         + Dₓᵀ M⁻ᵀ Pᵀ D_M D_xz v
        #         + Dₓᵀ M⁻ᵀ Pᵀ D_M D_zz P M⁻¹ Dₓ v
        t1 = P @ M.solve(Dx @ v)
        return (
            DD_Dxx @ v
            + DD_Dzx @ t1
            + Dx.T @ M.T.solve((P.T @ (DD_Dzx.T @ v)))
            + Dx.T @ M.T.solve((P.T @ (DM_Dzz @ t1)))
        )

    def hessian_inverse_product(
        self, z_in: torch.Tensor, target: torch.Tensor, b: bpm.Vertical, epsilon: float
    ) -> bpm.Vertical:
        "Solve (H + epsilon I) x = b using the algorithm in hessian.tex."

        # Compute the terms we used to compute the hessian-vector product H x.
        Dx, Dz, DD_Dxx, DD_Dzx, DM_Dzz = self.derivatives(z_in, target)
        M = bpm.IdentityWithLowerDiagonal((-Dz).flat[1:])
        P = bpm.downshifting_matrix(z_in.numel(), [b.shape[0] for b in Dx.flatten()])

        _validate_vector_is_Hessian_shaped(b, Dx)

        zero_block = bpm.Diagonal(
            [
                bpm.Zero((d.height, mt.width))
                for d, mt in zip(Dx.diagonal_blocks, M.T.diagonal_blocks)
            ]
        )

        # Write (H + epsilon I) x = b as an augmented system K [x;y;z] = [b;0;0].  K is a 3x3
        # block matrix. These  blocks are either diagonal, or bi-diagonal.
        K = bpm.Generic(
            [
                [DD_Dxx + epsilon * bpm.Identity(DD_Dxx.height), DD_Dzx @ P, Dx.T],
                [-Dx, M, zero_block],
                [-P.T @ DD_Dzx.T, -P.T @ DM_Dzz @ P, M.T],
            ]
        )

        zeros = bpm.Vertical([bpm.Zero((b.height, 1)) for b in M.diagonal_blocks])
        b00 = bpm.Vertical([b, zeros, zeros])

        # To solve K xyz = [b;0;0] for xyz efficiently, transform the equation
        # by pivoting the rows and columns of K with a permutation π so that K' = π K π is a
        # block-tridiagonal matrix. Such a permutation exists because the blocks of K
        # have bandwidth no greater than 2.  The pivoted system is
        #      π K π π⁻¹ xyz = π [b;0;0].
        # We can solve K' xyz' = π [b;0;0] for xyz' by factorizing K' and
        # applying the inverse of these factors, then report xyz = π⁻¹ xyz'. The paper shows
        # that  π⁻¹ = π, so we can just report π xyz'.
        K_pivoted = bpm.Tridiagonal.blockwise_transpose(K)

        # Confirm that all the blocks of the resulting tridiagonal matrix are 3x3 block matrices.
        # Then cast these explicit to Generic3x3 blocks so we can use a fast solver for them.
        assert all(b.shape == (3, 3) for b in K_pivoted.flatten())
        K_pivoted = bpm.Tridiagonal(
            [bpm.Generic3x3(b.blocks) for b in K_pivoted.diagonal_blocks],
            lower_blocks=[bpm.Generic3x3(b.blocks) for b in K_pivoted.lower_blocks],
            upper_blocks=[bpm.Generic3x3(b.blocks) for b in K_pivoted.upper_blocks],
        )

        b00_pivoted = b00.blockwise_transpose()

        xyz_pivoted = K_pivoted.solve(b00_pivoted)

        # Pivot back to the original order of x, y, z.
        xyz = xyz_pivoted.blockwise_transpose()

        # Just need the first block of xyz, which is x.
        return xyz.blocks[0][0]


class SequenceOfDenseBlocks(SequenceOfBlocks):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 19,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
    ):
        super().__init__(
            [DenseBlock(input_dim, hidden_dim, activation)]
            + [
                DenseBlock(hidden_dim, hidden_dim, activation)
                for _ in range(num_layers - 2)
            ],
            LossLayer(hidden_dim, num_classes),
        )
