"""
Implementation of the Hessian-inverse-vector product algorithm
as described in "The Hessian of tall-skinny networks is easy to invert"
"""

from typing import Callable, NamedTuple
import numpy as np
import torch
import torch.func as TF
import torch.nn as nn
import torch.nn.functional as F
import contextlib

import partitioned


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
    def __init__(self):
        super().__init__()
        # Caches for the input and output of the layer
        self.input = None
        self.output = None

    def backward_hook(
        self,
        _: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.dloss_dout = grad_output[0]

    def naked_forward(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, z_in: torch.Tensor, *args) -> torch.Tensor:
        self.input = z_in
        self.output = self.naked_forward(z_in, *args)
        return self.output

    def derivatives(self, dloss_dz: torch.Tensor, *forward_args) -> LayerDerivatives:
        if not (self.input.ndim == 2 and self.input.shape[0] == 1):
            raise ValueError("Input must be a 2D tensor with batch dimension=1")

        dloss_dz = dloss_dz.flatten()
        if dloss_dz.shape != (self.output.numel(),):
            raise ValueError(
                f"dloss_dz must have {self.output.numel()} elements after flattening, got {dloss_dz.shape[0]}"
            )

        z_in = self.input.flatten()
        params = dict(self.named_parameters())

        def f(x, z):
            return TF.functional_call(self, x, (z, *forward_args))

        def dloss_dz_f(x, z):
            return dloss_dz.flatten() @ f(x, z).flatten()

        return LayerDerivatives(
            Dx=reshape_pytree(
                TF.jacrev(lambda x: f(x, z_in))(params),
                starting_shape=self.output.shape,
            ),
            Dz=torch.func.jacrev(lambda z: f(params, z))(z_in),
            DD_Dxx=flatten_2d_pytree(TF.hessian(lambda x: dloss_dz_f(x, z_in))(params)),
            DD_Dzx=torch.func.jacrev(
                lambda z_in: reshape_pytree(
                    TF.jacrev(lambda x: dloss_dz_f(x, z_in))(params),
                    starting_shape=(),
                ),
            )(z_in),
            DM_Dzz=TF.hessian(lambda z: dloss_dz_f(params, z))(z_in),
        )


class DenseBlock(BlockWithMixedDerivatives):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.act_fn = activation

    def naked_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_fn(self.linear(x))


class LossLayer(DenseBlock):
    """Final layer that fuses the last linear layer with the loss computation."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__(input_dim, num_classes, nn.Identity())

    def naked_forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(super().naked_forward(x), targets)

    def derivatives(
        self, dloss_dz: torch.Tensor, targets: torch.Tensor
    ) -> LayerDerivatives:
        if targets.numel() != 1 and targets.dtype != torch.int64:
            raise ValueError("For the loss layer, targets must be an integer tensor")
        # Ignore dloss_dz. For the loss layer, it's always ∂z_L/∂z_L = 1.
        return super().derivatives(torch.tensor(1.0), targets.squeeze())


def save_dloss_dout(
    module: nn.Module,
    grad_input: tuple[torch.Tensor, ...],
    grad_output: tuple[torch.Tensor, ...],
) -> None:
    # Layer must have exactly one output.
    (module.dloss_dout,) = grad_output


class SequenceOfBlocks(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 19,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            *(
                [DenseBlock(input_dim, hidden_dim, activation)]
                + [
                    DenseBlock(hidden_dim, hidden_dim, activation)
                    for _ in range(num_layers - 2)
                ]
            )
        )
        self.loss_layer = LossLayer(hidden_dim, num_classes)

    @contextlib.contextmanager
    def save_dloss_douts(self):
        callbacks = [
            layer.register_full_backward_hook(save_dloss_dout) for layer in self.layers
        ]
        callbacks.append(self.loss_layer.register_full_backward_hook(save_dloss_dout))

        yield

        for callback in callbacks:
            callback.remove()

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_layer(self.layers(x), targets)

    def hessian_vector_product(
        self, x: torch.Tensor, target: torch.Tensor, v: partitioned.Vector
    ) -> None:
        with self.save_dloss_douts():
            # Populate the input and output caches of the layers and ∂z_L/∂z_ℓ for each layer ℓ.
            self(x, target).backward()

        Dx, Dz, DD_Dxx, DD_Dzx, DM_Dzz = map(
            partitioned.BlockDiagonalMatrix,
            zip(
                *(
                    [layer.derivatives(layer.dloss_dout) for layer in self.layers]
                    + [self.loss_layer.derivatives(1.0, target)]
                )
            ),
        )
        M = partitioned.IdentityWithLowerBlockDiagonalMatrix(-Dz)

        # Compute equation \ref{eq:hessian} from hessian.tex
        return (
            DD_Dxx @ v
            + DD_Dzx @ partitioned.downshift(M.solve(Dx @ v))
            + Dx.T @ M.T.solve(partitioned.upshift(DD_Dzx.T @ v))
            + Dx.T
            @ M.T.solve(
                partitioned.upshift(DM_Dzz @ partitioned.downshift(M.solve(Dx @ v)))
            )
        )
