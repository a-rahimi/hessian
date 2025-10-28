"""
Implementation of the Hessian-inverse-vector product algorithm
as described in "The Hessian of tall-skinny networks is easy to invert"
"""

from typing import Callable, NamedTuple
import abc
import numpy as np
import torch
import torch.func as TF
import torch.nn as nn
import torch.nn.functional as F

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
        self.input = None
        self.output = None

    def naked_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input = x
        self.output = self.naked_forward(x)
        return self.output

    def derivatives(self, dloss_dz: torch.Tensor) -> LayerDerivatives:
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
            return TF.functional_call(self, x, (z,))

        return LayerDerivatives(
            Dx=reshape_pytree(
                TF.jacrev(lambda x: f(x, z_in))(params),
                starting_shape=self.output.shape,
            ),
            Dz=torch.func.jacrev(lambda z: f(params, z))(z_in),
            DD_Dxx=flatten_2d_pytree(
                TF.hessian(lambda x: dloss_dz @ f(x, z_in))(params)
            ),
            DD_Dzx=torch.func.jacrev(
                lambda z_in: reshape_pytree(
                    TF.jacrev(lambda x: dloss_dz @ f(x, z_in))(params),
                    starting_shape=(),
                ),
            )(z_in),
            DM_Dzz=TF.hessian(lambda z: dloss_dz @ f(params, z))(z_in),
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

    def naked_forward(self, x):
        return self.act_fn(self.linear(x))


class LossLayer(BlockWithMixedDerivatives):
    """Final layer that fuses the last linear layer with the loss computation."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.layer = DenseBlock(input_dim, num_classes, nn.Identity())

    def forward(self, x, targets):
        return F.cross_entropy(self.layer(x), targets)

    def derivatives(self, dloss_dz: torch.Tensor) -> LayerDerivatives:
        raise NotImplementedError


class SequenceOfBlocks(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 19,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
    ):
        super().__init__(
            *(
                [DenseBlock(input_dim, hidden_dim, activation)]
                + [
                    DenseBlock(hidden_dim, hidden_dim, activation)
                    for _ in range(num_layers - 2)
                ]
                + [LossLayer(hidden_dim, num_classes)]
            )
        )
        assert len(self) == num_layers

    def hessian_vector_product(
        self, x: torch.Tensor, targets: torch.Tensor, v: partitioned.Vector
    ) -> None:
        self.zero_grad()
        loss = self(x, targets)
        loss.backward()

        # ∂z_L/∂x_ℓ
        dloss_dx = torch.autograd.grad(
            loss, self.parameters(), create_graph=True, retain_graph=True
        )

        # b_ℓ = ∂z_L/∂z_ℓ
        dloss_dz = [
            torch.autograd.grad(loss, layer.input, create_graph=True, retain_graph=True)
            for layer in self
        ]
        Dx, Dz, DD_Dxx, DD_Dzx, DM_Dxz, DM_Dzz = map(
            partitioned.Matrix,
            zip(*[layer.derivatives(dloss_dz) for layer in self]),
        )
        M = partitioned.IdentityWithLowerBlockDiagonalMatrix(-Dz)

        # Compute equation \ref{eq:hessian} from hessian.tex
        return (
            DD_Dxx @ v
            + DD_Dzx @ partitioned.downshift(M.solve(Dx @ v))
            + Dx.T @ M.T.solve(partitioned.upshift(DM_Dxz @ v))
            + Dx.T
            @ M.T.solve(
                partitioned.upshift(DM_Dzz @ partitioned.downshift(M.solve(Dx @ v)))
            )
        )
