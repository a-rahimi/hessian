"""
Implementation of the Hessian-inverse-vector product algorithm
as described in "The Hessian of tall-skinny networks is easy to invert"
"""

from typing import Callable, Iterator, Iterable, NamedTuple, Sequence
import numpy as np
import scipy.sparse.linalg
import torch
import torch.func as TF
import torch.nn as nn
import torch.nn.functional as F
import contextlib

import block_partitioned_matrices as bpm
import timing


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


class HessianInverseSetup(NamedTuple):
    """Epsilon-independent terms shared across (H + epsilon I) solves.

    Produced by `SequenceOfBlocks.hessian_inverse_setup` and consumed by
    `hessian_inverse_solve`. Holds the network derivatives plus the structural
    matrices (M, P, zero_block) so a sweep over the damping epsilon recomputes
    only the cheap per-epsilon assembly and factorization.
    """

    Dx: bpm.Diagonal
    DD_Dxx: bpm.Diagonal
    DD_Dzx: bpm.Diagonal
    DM_Dzz: bpm.Diagonal
    M: "bpm.IdentityWithLowerDiagonal"
    P: torch.Tensor
    zero_block: bpm.Diagonal


class BlockWithMixedDerivatives(nn.Module):
    "An abstract layer for which various partial derivatives can be computed."

    # Whether Dz and DM_Dzz are batch-block-diagonal for this layer, i.e.
    # whether sample i's output depends only on sample i's input. This holds
    # for ordinary per-sample layers (DenseBlock etc.) but not for LossLayer,
    # whose scalar mean-loss output couples every sample together.
    batch_structured_dz = True

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

        if self.batch_structured_dz:
            Dz, DM_Dzz = self._per_sample_Dz_DM_Dzz(dloss_dz, params, input_shape)
        else:
            Dz = torch.func.jacrev(lambda z: f(params, z))(z_in).reshape(
                self.output.numel(), -1
            )
            DM_Dzz = TF.hessian(lambda z: dloss_dz_f(params, z))(z_in)

        return LayerDerivatives(
            Dx=reshape_pytree(
                TF.jacrev(lambda x: f(x, z_in))(params),
                starting_shape=self.output.shape,
            ).reshape(self.output.numel(), -1),
            Dz=Dz,
            DD_Dxx=flatten_2d_pytree(TF.hessian(lambda x: dloss_dz_f(x, z_in))(params)),
            DD_Dzx=torch.func.jacrev(
                lambda z_in: reshape_pytree(
                    TF.jacrev(lambda x: dloss_dz_f(x, z_in))(params),
                    starting_shape=(),
                ),
            )(z_in).reshape(-1, self.input.numel()),
            DM_Dzz=DM_Dzz,
        )

    def _per_sample_Dz_DM_Dzz(
        self, dloss_dz: torch.Tensor, params: dict, input_shape: torch.Size
    ) -> tuple["bpm.Diagonal", "bpm.Diagonal"]:
        """Compute Dz and DM_Dzz per-sample and return them as nested `bpm.Diagonal`s.

        Sample i's output depends only on sample i's input (verified in
        docs/batch-structure-plan.md), so the (batch*w) x (batch*w) Dz and
        DM_Dzz blocks are exactly block-diagonal in the batch dimension. Rather
        than materializing the dense (batch*w)^2 tensor and discarding the
        zeros, vmap a per-sample jacobian/hessian over the batch to get the
        `batch` non-zero w x w sub-blocks directly.
        """
        batch = input_shape[0]
        sample_shape = input_shape[1:]
        z_batch = self.input.reshape(batch, -1)
        dloss_batch = dloss_dz.reshape(batch, -1)

        def naked_single(z_s: torch.Tensor) -> torch.Tensor:
            out = TF.functional_call(
                self, params, (z_s.reshape(sample_shape).unsqueeze(0),)
            )
            return out.reshape(-1)

        def scalar_single(z_s: torch.Tensor, dloss_s: torch.Tensor) -> torch.Tensor:
            return dloss_s @ naked_single(z_s)

        Dz_batched = torch.func.vmap(torch.func.jacrev(naked_single))(z_batch)
        DM_Dzz_batched = torch.func.vmap(torch.func.hessian(scalar_single))(
            z_batch, dloss_batch
        )

        Dz = bpm.Diagonal([bpm.Tensor(block) for block in Dz_batched])
        DM_Dzz = bpm.Diagonal([bpm.Tensor(block) for block in DM_Dzz_batched])
        return Dz, DM_Dzz


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

    # The loss layer's output is the scalar mean loss over the whole batch, not
    # a per-sample vector, so Dz (a single row) cannot be split into batch x w
    # x w sub-blocks the way a normal layer's can -- there is no batch axis on
    # the output to block against. (DM_Dzz's cross-sample terms happen to be
    # numerically zero for mean cross-entropy too, but reproducing that
    # per-sample -- correctly accounting for the 1/batch mean scaling -- adds
    # risk without the memory payoff that matters here, so both stay dense.)
    batch_structured_dz = False

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


def _splu_solve(K: bpm.Generic, b: bpm.Vertical, Dx: bpm.Diagonal) -> bpm.Vertical:
    """Solve K [x;y;z] = [b;0;0] with scipy's sparse LU and return x.

    The factorization runs in float64 with partial pivoting, so it is both
    more accurate and faster than the unpivoted block LDU path. SuperLU's
    fill-reducing column ordering makes the blockwise-transpose pivoting of
    the block path unnecessary.
    """
    with torch.no_grad():
        with timing.record("solve/splu/to-csc"):
            K_csc = K.to_scipy_csc()

        with timing.record("solve/splu/factorize"):
            lu = scipy.sparse.linalg.splu(K_csc)

        with timing.record("solve/splu/pack"):
            rhs = np.zeros(K_csc.shape[0], dtype=np.float64)
            offset = 0
            for block in b.flat:
                block_np = block.detach().to(torch.float64).numpy().ravel()
                rhs[offset : offset + block_np.size] = block_np
                offset += block_np.size

        with timing.record("solve/splu/substitute"):
            xyz = lu.solve(rhs)

        with timing.record("solve/splu/pack"):
            x_blocks = []
            offset = 0
            for b_block, Dx_block in zip(b.flat, Dx.diagonal_blocks):
                width = Dx_block.width
                x_np = np.ascontiguousarray(xyz[offset : offset + width])
                x_blocks.append(
                    bpm.Tensor(
                        torch.from_numpy(x_np).to(b_block.dtype).reshape(width, 1)
                    )
                )
                offset += width
            return bpm.Vertical(x_blocks)


def _densify_per_sample_blocks(diagonal: bpm.Diagonal) -> bpm.Diagonal:
    """Collapse each layer's per-sample nested `Diagonal` block into a dense `Tensor`.

    `Dz` and `DM_Dzz` carry a per-sample block-diagonal structure (a nested
    `Diagonal` of `batch` sub-blocks). The reference matrix-vector and block
    paths eliminate the activation space in a way that couples the batch samples
    through the shared parameters, so that structure cannot survive the
    computation. Materializing each layer's block densely lets those paths do
    only dense activation-space linear algebra, with no structured matrix ever
    multiplying an unstructured tensor. The default splu path keeps the full
    per-sample structure (via `to_scipy_csc`), so the setup's memory win is
    unaffected.
    """
    return bpm.Diagonal(
        [bpm.Tensor(block.to_tensor()) for block in diagonal.diagonal_blocks]
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
        # The activation-space elimination below couples the batch samples, so
        # collapse Dz's and DM_Dzz's per-sample structure into dense blocks.
        Dz = _densify_per_sample_blocks(Dz)
        DM_Dzz = _densify_per_sample_blocks(DM_Dzz)
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

    def hessian_inverse_setup(
        self, z_in: torch.Tensor, target: torch.Tensor
    ) -> "HessianInverseSetup":
        """Precompute the epsilon-independent terms of the (H + epsilon I) solve.

        The network derivatives (and the structural matrices M, P, zero_block
        derived from them) do not depend on the damping epsilon. Computing them
        is the expensive part of `hessian_inverse_product` -- it runs functorch
        over the whole network. Splitting it out lets a caller that needs to
        solve `(H + epsilon I) x = b` for many epsilon values (e.g. the
        trust-region subproblem's search over the damping lambda) pay this cost
        once and reuse it across all the solves.
        """
        with timing.record("setup/derivatives"):
            Dx, Dz, DD_Dxx, DD_Dzx, DM_Dzz = self.derivatives(z_in, target)
        M = bpm.IdentityWithLowerDiagonal((-Dz).flat[1:])
        P = bpm.downshifting_matrix(z_in.numel(), [b.shape[0] for b in Dx.flatten()])

        zero_block = bpm.Diagonal(
            [
                bpm.Zero((d.height, mt.width))
                for d, mt in zip(Dx.diagonal_blocks, M.T.diagonal_blocks)
            ]
        )

        return HessianInverseSetup(
            Dx=Dx,
            DD_Dxx=DD_Dxx,
            DD_Dzx=DD_Dzx,
            DM_Dzz=DM_Dzz,
            M=M,
            P=P,
            zero_block=zero_block,
        )

    def hessian_inverse_solve(
        self,
        setup: "HessianInverseSetup",
        b: bpm.Vertical,
        epsilon: float,
        solver: str = "splu",
    ) -> bpm.Vertical:
        """Solve (H + epsilon I) x = b reusing a precomputed `setup`.

        Only the cheap, epsilon- and b-dependent assembly and factorization are
        redone here. `solver` selects the factorization: "splu" (default) uses
        scipy's sparse LU with partial pivoting in float64, which is faster and
        far more accurate; "block" uses the paper's unpivoted block-tridiagonal
        LDU factorization.
        """
        if solver not in ("splu", "block"):
            raise ValueError(f"solver must be 'splu' or 'block', got {solver!r}")

        Dx, DD_Dxx, DD_Dzx, DM_Dzz, M, P, zero_block = setup

        _validate_vector_is_Hessian_shaped(b, Dx)

        # Write (H + epsilon I) x = b as an augmented system K [x;y;z] = [b;0;0].  K is a 3x3
        # block matrix. These  blocks are either diagonal, or bi-diagonal.
        with timing.record("solve/assemble-K"):
            K = bpm.Generic(
                [
                    [DD_Dxx + epsilon * bpm.Identity(DD_Dxx.height), DD_Dzx @ P, Dx.T],
                    [-Dx, M, zero_block],
                    [-P.T @ DD_Dzx.T, -P.T @ DM_Dzz @ P, M.T],
                ]
            )

        if solver == "splu":
            return _splu_solve(K, b, Dx)

        # The block factorization couples the batch samples through the shared
        # parameters (the Schur complements S12, S22 are dense across samples),
        # so the per-sample structure of `M` and `DM_Dzz` cannot survive the
        # solve. Densify those nested per-sample blocks so the block path is
        # entirely dense activation-space linear algebra.
        M = bpm.IdentityWithLowerDiagonal(
            [bpm.Tensor(block.to_tensor()) for block in M.lower_blocks]
        )
        DM_Dzz = _densify_per_sample_blocks(DM_Dzz)
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
        with timing.record("solve/block/pivot"):
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

        # Spell out Tridiagonal.solve here so the factorization and the
        # triangular substitutions are timed as separate stages.
        with timing.record("solve/block/factorize"):
            L, D, U = K_pivoted.LDU_decomposition()

        with timing.record("solve/block/substitute"):
            xyz_pivoted = U.solve(D.solve(L.solve(b00_pivoted)))

        # Pivot back to the original order of x, y, z.
        xyz = xyz_pivoted.blockwise_transpose()

        # Just need the first block of xyz, which is x.
        return xyz.blocks[0][0]

    def hessian_inverse_product(
        self,
        z_in: torch.Tensor,
        target: torch.Tensor,
        b: bpm.Vertical,
        epsilon: float,
        solver: str = "splu",
    ) -> bpm.Vertical:
        "Solve (H + epsilon I) x = b using the algorithm in hessian.tex."
        setup = self.hessian_inverse_setup(z_in, target)
        return self.hessian_inverse_solve(setup, b, epsilon, solver=solver)


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
