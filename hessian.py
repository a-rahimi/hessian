"""
Implementation of the Hessian-inverse-vector product algorithm
as described in "The Hessian of tall-skinny networks is easy to invert"
"""

from typing import Callable, NamedTuple
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F

import partitioned


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
    DM_Dxz: torch.Tensor  # D_M ∇_{xz} f_ℓ: (a² × p), scaled by b_ℓ
    DM_Dzz: torch.Tensor  # D_M ∇_{zz} f_ℓ: (a² × a), scaled by b_ℓ


class BlockWithMixedDerivatives(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def derivatives(self, x: torch.Tensor) -> LayerDerivatives:
        pass


class BasicBlock(BlockWithMixedDerivatives):
    """A single layer f_ℓ(z_{ℓ-1}; x_ℓ) = activation(W @ z_{ℓ-1} + bias)"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.act_fn = activation
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        self.output = self.act_fn(self.linear(x))
        return self.output

    def derivatives(self, x: torch.Tensor) -> LayerDerivatives:
        raise NotImplementedError


class LossLayer(BlockWithMixedDerivatives):
    """Final layer that fuses the last linear layer with the loss computation."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.layer = BasicBlock(input_dim, num_classes, nn.Identity())

    def forward(self, x, targets):
        """
        Args:
            x: input activations
            targets: target labels
        Returns:
            scalar loss value
        """
        return F.cross_entropy(self.layer(x), targets)

    def derivatives(self, x: torch.Tensor) -> LayerDerivatives:
        raise NotImplementedError


class SequenceOfBlocks(nn.Sequential):
    """
    Deep MLP that tracks intermediate activations for Hessian computation.
    """

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
                [BasicBlock(input_dim, hidden_dim, activation)]
                + [
                    BasicBlock(hidden_dim, hidden_dim, activation)
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
            zip(*[layer.derivatives(dloss_dx, dloss_dz) for layer in self]),
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
