"""A small library of operations on partitioned matrices.

The blocks of these matrices can in turn be block matrices themselves, or they
canbe torch tensors. This way, it becomes easy to express linear algebra
operations like matrix multiplication, inversion, and solving linear systems
with matrices that have hierarchical structure.

For example, if a matrix A has a 2x2 block structre and each of these blocks are in turn block diagonal matrices,
one can define

A = SymmetricBlock2x2Matrix(
    # A11 is a block diagona matrix with two blocks.
    block11=BlockDiagonalMatrix([torch.randn(3, 3), torch.randn(2, 2)]),
    # A12 is a blockdiagonal matrix with two blocks.
    block12=BlockDiagonalMatrix([torch.randn(3, 2), torch.randn(2, 2)]),
    A A22 is just a dense matrix
    block22=torch.randn(2, 2)
)

One can then multiply A by a vector v with A @ v, or solve the linear system A @
x = b with x = A.solve(b), or recover the inverse of A with A.invert().
"""

import dataclasses as dc
from functools import singledispatchmethod
from math import sin
from typing import Sequence
import torch


class Matrix:
    "Base class for partitioned matrices."

    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    def invert(self) -> "Matrix":
        raise NotImplementedError

    def solve(self, rhs: "Matrix") -> "Matrix":
        raise NotImplementedError

    @property
    def width(self) -> int:
        raise NotImplementedError

    @property
    def height(self) -> int:
        raise NotImplementedError


class TorchMatrix(torch.Tensor, Matrix):
    "Endows torch.Tensor with invert() and solve() methods."

    def invert(self) -> "TorchMatrix":
        return TorchMatrix(torch.linalg.inv(self))

    def solve(self, rhs: "TorchMatrix") -> "TorchMatrix":
        return torch.linalg.solve(self, rhs)

    def to_tensor(self) -> torch.Tensor:
        return self

    @singledispatchmethod
    def __matmul__(self, other: Matrix) -> Matrix:
        return torch.Tensor.__matmul__(self, other)

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[0]

    @staticmethod
    def wrap(tensor: torch.Tensor | Matrix) -> "TorchMatrix":
        if isinstance(tensor, Matrix):
            return tensor
        elif isinstance(tensor, torch.Tensor):
            if tensor.ndim != 2:
                raise ValueError("Tensor must be a 2D tensor")
            return TorchMatrix(tensor)
        raise ValueError("Tensor must be a torch.Tensor or Matrix")


@dc.dataclass(frozen=True)
class Identity(Matrix):
    dimension: int = 0

    def __matmul__(self, other: Matrix) -> Matrix:
        return other

    def solve(self, rhs: Matrix) -> Matrix:
        return rhs

    def invert(self) -> "Identity":
        return self

    def to_tensor(self) -> torch.Tensor:
        return torch.eye(self.dimension)

    @property
    def T(self) -> "Identity":
        return self

    @property
    def width(self) -> int:
        return self.dimension

    @property
    def height(self) -> int:
        return self.dimension


@TorchMatrix.__matmul__.register
def _(self, _: Identity) -> TorchMatrix:
    return self


@dc.dataclass(frozen=True)
class Zero(Matrix):
    shape: tuple[int, int] = ()

    def __matmul__(self, other: Matrix) -> Matrix:
        if self.shape:
            if other.height != self.shape[1]:
                raise ValueError(
                    f"Shape mismatch {self} vs {other.height} x {other.width}"
                )
            return torch.zeros(self.shape[0], other.width)
        return TorchMatrix(torch.tensor([[0.0]]))

    def to_tensor(self) -> torch.Tensor:
        return torch.zeros(*self.shape)

    def invert(self) -> "Zero":
        raise ValueError("Zero matrix is not invertible")

    def solve(self, rhs: "Matrix") -> "Matrix":
        raise ValueError("Zero matrix is not invertible")

    @property
    def T(self) -> "Zero":
        if not self.shape:
            return Zero()
        return Zero(shape=(self.shape[1], self.shape[0]))


class Stacked(Matrix):
    "Abstract base class of blocks stacked either veritcally, horizontally, or diagonally."

    def __init__(self, blocks: Sequence[Matrix]):
        self.blocks = list(map(TorchMatrix.wrap, blocks))

    def __neg__(self) -> "Stacked":
        """Return negation of the matrix."""
        return self.__class__([-block for block in self.blocks])

    @singledispatchmethod
    def __matmul__(self, other: Matrix) -> Matrix:
        raise NotImplementedError

    @singledispatchmethod
    def __add__(self, other: Matrix) -> Matrix:
        raise NotImplementedError

    @singledispatchmethod
    def __sub__(self, other: Matrix) -> Matrix:
        raise NotImplementedError


@Stacked.__add__.register
def _(self, other: Stacked) -> Stacked:
    if len(self.blocks) != len(other.blocks):
        raise ValueError("Number of blocks must match")
    return self.__class__(
        [b + other_b for b, other_b in zip(self.blocks, other.blocks)]
    )


@Stacked.__sub__.register
def _(self, other: Stacked) -> Stacked:
    if len(self.blocks) != len(other.blocks):
        raise ValueError("Number of blocks in matrices must match")
    return self.__class__(
        [block - other_block for block, other_block in zip(self.blocks, other.blocks)]
    )


class Vertical(Stacked):
    """Blocked stacked vertically."""

    def to_tensor(self) -> torch.Tensor:
        return torch.cat([b.to_tensor() for b in self.blocks], dim=0)

    @property
    def width(self) -> int:
        width = self.blocks[0].width
        assert all(
            b.width == width for b in self.blocks
        ), "All blocks must have the same width"
        return width

    @property
    def height(self) -> int:
        return sum(b.height for b in self.blocks)


class Diagonal(Stacked):
    """Represents a block-diagonal matrix as a list of blocks."""

    def invert(self) -> "Diagonal":
        return Diagonal([b.invert() for b in self.blocks])

    def solve(self, rhs: Stacked) -> Stacked:
        """Solve M x = rhs for block-diagonal M."""
        if len(self.blocks) != len(rhs.blocks):
            raise ValueError("Number of blocks in matrix and vector must match")

        return self.__class__(
            [
                block.solve(rhs_block)
                for block, rhs_block in zip(self.blocks, rhs.blocks)
            ]
        )

    def to_tensor(self) -> torch.Tensor:
        return torch.block_diag(*[b.to_tensor() for b in self.blocks])

    @singledispatchmethod
    def __add__(self, other: Matrix) -> Matrix:
        raise NotImplementedError

    @singledispatchmethod
    def __sub__(self, other: Matrix) -> Matrix:
        raise NotImplementedError

    @property
    def T(self) -> "Diagonal":
        """Return transpose of this matrix."""
        return Diagonal([block.T for block in self.blocks])

    @property
    def width(self) -> int:
        return sum(b.width for b in self.blocks)

    @property
    def height(self) -> int:
        return sum(b.height for b in self.blocks)


@Diagonal.__matmul__.register
def _(self, other: Stacked) -> Stacked:
    if len(self.blocks) != len(other.blocks):
        raise ValueError("Number of blocks must match")

    return other.__class__(
        [block @ oblock for block, oblock in zip(self.blocks, other.blocks)]
    )


@Diagonal.__add__.register
def _(self, other: Diagonal) -> Diagonal:
    "Diagonal + Diagonal is the same code as Vertical + Vertical."
    return Stacked.__add__(other, self)


@Diagonal.__add__.register
def _(self, other: Identity) -> Diagonal:
    "Diagonal + I"
    return Diagonal([b + torch.eye(b.shape[0]) for b in self.blocks])


@Diagonal.__sub__.register
def _(self, other: Diagonal) -> Diagonal:
    "Diagonal - Diagonal is the same code as Vertical - Vertical."
    return Stacked.__sub__(self, other)


@Diagonal.__sub__.register
def _(self, other: Identity) -> Diagonal:
    "Diagonal - I"
    return Diagonal([b - torch.eye(b.shape[0]) for b in self.blocks])


class LowerBlockBiDiagonal(Matrix):
    """Represents a block bi-diagonal matrix.

    [D_0    0
     L_1  D_2    0
          L_1  D_3
              ...
               L_N  D_N]
    """

    def __init__(
        self, lower_blocks: Sequence[Matrix], diagonal_blocks: Sequence[Matrix]
    ):
        if len(lower_blocks) != len(diagonal_blocks) - 1:
            raise ValueError(
                "Number of lower blocks must be one less than the number of diagonal blocks"
            )
        self.lower_blocks = list(map(TorchMatrix.wrap, lower_blocks))
        self.diagonal_blocks = list(map(TorchMatrix.wrap, diagonal_blocks))

    @singledispatchmethod
    def __matmul__(self, v: Matrix) -> Matrix:
        raise NotImplementedError

    @__matmul__.register
    def _(self, v: Vertical) -> Vertical:
        if len(v.blocks) != len(self.diagonal_blocks):
            raise ValueError("Number of blocks in vector and matrix must match")

        return Vertical(
            [self.diagonal_blocks[0] @ v.blocks[0]]
            + [
                self.lower_blocks[i] @ v.blocks[i]
                + self.diagonal_blocks[i + 1] @ v.blocks[i + 1]
                for i in range(len(self.lower_blocks))
            ]
        )

    def solve(self, rhs: Vertical) -> Vertical:
        """Solve M x = rhs using back-substitution."""
        # We have that
        #   diagonal_blocks[0] @ x[0] = rhs[0]
        # so
        #   x[0] = diagonal_block[0] \ rhs[0]
        # For i>0, we have
        #   lower_blocks[i] @ x[i] + diagonal_blocks[i+1] @ x[i+1] = rhs[i+1]
        # so
        #   x[i+1] = diagonal_blocks[i+1] \ (rhs[i+1] - lower_blocks[i] @ x[i])

        if len(rhs.blocks) != len(self.lower_blocks) + 1:
            raise ValueError("Number of blocks in vector and matrix must match")

        result_blocks = [self.diagonal_blocks[0].solve(rhs.blocks[0])]
        for i in range(len(self.lower_blocks)):
            result_blocks.append(
                self.diagonal_blocks[i + 1].solve(
                    rhs.blocks[i + 1] - self.lower_blocks[i] @ result_blocks[i]
                )
            )
        return Vertical(result_blocks)

    @property
    def T(self) -> "UpperBlockBiDiagonal":
        return UpperBlockBiDiagonal(
            upper_blocks=[b.T for b in self.lower_blocks],
            diagonal_blocks=[b.T for b in self.diagonal_blocks],
        )


def IdentityWithLowerBlockDiagonalMatrix(lower_blocks: Sequence[Matrix]):
    return LowerBlockBiDiagonal(
        lower_blocks=list(map(TorchMatrix.wrap, lower_blocks)),
        diagonal_blocks=[Identity()] * (len(lower_blocks) + 1),
    )


class UpperBlockBiDiagonal(Matrix):
    """Represents a block bi-diagonal matrix

    M = [D_0         U_1           0        ...              0
           0         D_1           U_2      ...              0
           ...       ...           ...      ...             ...
           0                       ...          D_{N-1}     U_N
           0                       ...          0           D_N
        ]
    """

    def __init__(
        self,
        upper_blocks: Sequence[Matrix],
        diagonal_blocks: Sequence[Matrix],
    ):
        if len(upper_blocks) != len(diagonal_blocks) - 1:
            raise ValueError(
                "Number of upper blocks must be one less than the number of diagonal blocks"
            )
        self.upper_blocks = list(map(TorchMatrix.wrap, upper_blocks))
        self.diagonal_blocks = list(map(TorchMatrix.wrap, diagonal_blocks))

    @singledispatchmethod
    def __matmul__(self, v: Matrix) -> Matrix:
        raise NotImplementedError

    @__matmul__.register
    def __matmul__(self, v: Vertical) -> Vertical:
        if len(v.blocks) != len(self.diagonal_blocks):
            raise ValueError("Number of blocks in vector and matrix must match")

        return Vertical(
            [
                self.diagonal_blocks[i] @ v.blocks[i]
                + self.upper_blocks[i] @ v.blocks[i + 1]
                for i in range(len(self.upper_blocks))
            ]
            + [self.diagonal_blocks[-1] @ v.blocks[-1]]
        )

    def solve(self, rhs: Vertical) -> Vertical:
        """Solve M x = rhs using forward-substitution."""
        # We have for i<N,
        #   diagonal_blocks[i] @ x[i] + upper_blocks[i] @ x[i+1] = rhs[i]
        # so
        #   x[i] = diagonal_blocks[i] \ (rhs[i] - upper_blocks[i] @ x[i+1])
        # For i=N, we have
        #   diagonal_blocks[N] @ x[N] = rhs[N]
        # so
        #   x[N] = diagonal_blocks[N] \ rhs[N]

        if len(rhs.blocks) != len(self.upper_blocks) + 1:
            raise ValueError("Number of blocks in vector and matrix must match")

        result_blocks = [self.diagonal_blocks[-1].solve(rhs.blocks[-1])]
        for i in range(len(self.upper_blocks) - 1, -1, -1):
            result_blocks.insert(
                0,
                self.diagonal_blocks[i].solve(
                    rhs.blocks[i] - self.upper_blocks[i] @ result_blocks[0]
                ),
            )
        return Vertical(result_blocks)

    @property
    def T(self) -> LowerBlockBiDiagonal:
        return LowerBlockBiDiagonal(
            lower_blocks=[b.T for b in self.upper_blocks],
            diagonal_blocks=[b.T for b in self.diagonal_blocks],
        )


def IdentityWithUpperBlockDiagonalMatrix(upper_blocks: Sequence[Matrix]):
    return UpperBlockBiDiagonal(
        upper_blocks=list(map(TorchMatrix.wrap, upper_blocks)),
        diagonal_blocks=[Identity()] * (len(upper_blocks) + 1),
    )


class SymmetricBlock2x2Matrix(Matrix):
    """Represents a symmetric 2x2block matrix whose blocks are in turn Matrices."""

    def __init__(self, block11: Matrix, block12: Matrix, block22: Matrix):
        self.block11 = TorchMatrix.wrap(block11)
        self.block12 = TorchMatrix.wrap(block12)
        self.block22 = TorchMatrix.wrap(block22)

    @singledispatchmethod
    def __matmul__(self, v: Matrix) -> Matrix:
        raise NotImplementedError

    @__matmul__.register
    def __matmul__(self, v: Vertical) -> Vertical:
        if len(v.blocks) != 2:
            raise ValueError("Number of blocks in vector and matrix must match")
        return Vertical(
            [
                self.block11 @ v.blocks[0] + self.block12 @ v.blocks[1],
                self.block22 @ v.blocks[1] + self.block12.T @ v.blocks[0],
            ]
        )

    def invert(self) -> "SymmetricBlock2x2Matrix":
        # The Schur complement S = Q11 - Q12 @ Q22^{-1} @ Q21, and its inverse.
        block22_inv = self.block22.invert()
        S = self.block11 - self.block12 @ block22_inv @ self.block12.T
        S_inv = S.invert()

        return SymmetricBlock2x2Matrix(
            # Q_inv_11 = S^{-1}
            block11=S_inv,
            # Q_inv_12 = -S^{-1} @ Q12 @ Q22^{-1}
            block12=-S_inv @ self.block12 @ block22_inv,
            # Q_inv_22 = Q22^{-1} + Q22^{-1} @ Q21 @ S^{-1} @ Q12 @ Q22^{-1}
            block22=block22_inv
            + block22_inv @ self.block12.T @ S_inv @ self.block12 @ block22_inv,
        )

    def to_tensor(self) -> torch.Tensor:
        return torch.vstack(
            [
                torch.hstack([self.block11.to_tensor(), self.block12.to_tensor()]),
                torch.hstack([self.block12.T.to_tensor(), self.block22.to_tensor()]),
            ]
        )

    def UDU_decomposition(
        self,
    ) -> tuple[IdentityWithUpperBlockDiagonalMatrix, Diagonal]:
        b22_inv = self.block22.invert()
        U = IdentityWithUpperBlockDiagonalMatrix([self.block12 @ b22_inv])
        D = Diagonal(
            [self.block11 - self.block12 @ b22_inv @ self.block12.T, self.block22]
        )
        return U, D


class LowerBlockDiagonal(Stacked):
    def __init__(
        self,
        height_leading_zeros: int,
        lower_blocks: Sequence[Matrix],
        width_trailing_zeros: int,
    ):
        super().__init__(blocks=lower_blocks)
        self.height_leading_zeros = height_leading_zeros
        self.width_trailing_zeros = width_trailing_zeros

    @singledispatchmethod
    def __matmul__(self, other: Matrix) -> Matrix:
        raise NotImplementedError

    @__matmul__.register
    def _(self, other: Diagonal) -> "LowerBlockDiagonal":
        if len(other.blocks) != len(self.blocks) + 1:
            raise ValueError("Number of blocks in the operands must match")

        return LowerBlockDiagonal(
            height_leading_zeros=self.height_leading_zeros,
            lower_blocks=[
                b @ other_block
                for b, other_block in zip(self.blocks, other.blocks[:-1])
            ],
            width_trailing_zeros=other.blocks[-1].shape[1],
        )

    @__matmul__.register
    def _(self, other: Vertical) -> Vertical:
        if len(other.blocks) != len(self.blocks) + 1:
            raise ValueError("Number of blocks in the operands must match")

        return Vertical(
            [torch.zeros(self.height_leading_zeros, other.width)]
            + [
                b @ other_block
                for b, other_block in zip(self.blocks, other.blocks[:-1])
            ],
        )

    def to_tensor(self) -> torch.Tensor:
        t = torch.zeros(self.height, self.width)

        row, col = self.height_leading_zeros, 0
        for block in self.blocks:
            t[row : row + block.height, col : col + block.width] = block.to_tensor()
            row += block.height
            col += block.width

        return t

    @property
    def height(self) -> int:
        return self.height_leading_zeros + sum(b.height for b in self.blocks)

    @property
    def width(self) -> int:
        return sum(b.width for b in self.blocks) + self.width_trailing_zeros

    @property
    def T(self) -> "UpperBlockDiagonal":
        return UpperBlockDiagonal(
            width_leading_zeros=self.height_leading_zeros,
            upper_blocks=[b.T for b in self.blocks],
            height_trailing_zeros=self.width_trailing_zeros,
        )


@Diagonal.__matmul__.register
def _(self, other: LowerBlockDiagonal) -> Diagonal:
    if len(self.blocks) != len(other.blocks) + 1:
        raise ValueError("Number of blocks in the operands must match")
    return LowerBlockDiagonal(
        height_leading_zeros=self.blocks[0].height,
        lower_blocks=[
            b @ other_block for b, other_block in zip(self.blocks[1:], other.blocks)
        ],
        width_trailing_zeros=other.width_trailing_zeros,
    )


def downshifting_matrix(
    height_leading_zeros, v_heights: Sequence[int]
) -> LowerBlockDiagonal:
    return LowerBlockDiagonal(
        height_leading_zeros=height_leading_zeros,
        lower_blocks=list(map(Identity, v_heights[:-1])),
        width_trailing_zeros=v_heights[-1],
    )


class UpperBlockDiagonal(Stacked):
    def __init__(
        self,
        width_leading_zeros: int,
        upper_blocks: Sequence[Matrix],
        height_trailing_zeros: int,
    ):
        super().__init__(blocks=upper_blocks)
        self.height_trailing_zeros = height_trailing_zeros
        self.width_leading_zeros = width_leading_zeros

    @singledispatchmethod
    def __matmul__(self, other: Matrix) -> Matrix:
        raise NotImplementedError

    @__matmul__.register
    def _(self, other: Diagonal) -> Matrix:
        if len(other.blocks) != len(self.blocks) + 1:
            raise ValueError("Number of blocks in the operands must match")
        return UpperBlockDiagonal(
            width_leading_zeros=other.blocks[0].shape[1],
            upper_blocks=[
                b @ other_block for b, other_block in zip(self.blocks, other.blocks[1:])
            ],
            height_trailing_zeros=self.height_trailing_zeros,
        )

    @__matmul__.register
    def _(self, other: Vertical) -> Vertical:
        if len(other.blocks) != len(self.blocks) + 1:
            raise ValueError("Number of blocks in the operands must match")
        return Vertical(
            [b @ other_block for b, other_block in zip(self.blocks, other.blocks[1:])]
            + [torch.zeros(self.height_trailing_zeros, other.width)]
        )

    @__matmul__.register
    def _(self, other: LowerBlockDiagonal) -> Diagonal:
        if len(other.blocks) != len(self.blocks):
            raise ValueError("Number of blocks in the operands must match")
        return Diagonal(
            [b @ other_b for b, other_b in zip(self.blocks, other.blocks)]
            + [torch.zeros(self.height_trailing_zeros, other.width_trailing_zeros)]
        )

    def to_tensor(self) -> torch.Tensor:
        t = torch.zeros(self.height, self.width)
        row, col = 0, self.width_leading_zeros
        for block in self.blocks:
            t[row : row + block.height, col : col + block.width] = block.to_tensor()
            row += block.height
            col += block.width
        return t

    @property
    def height(self) -> int:
        return sum(b.height for b in self.blocks) + self.height_trailing_zeros

    @property
    def width(self) -> int:
        return self.width_leading_zeros + sum(b.width for b in self.blocks)

    @property
    def T(self) -> LowerBlockDiagonal:
        return LowerBlockDiagonal(
            height_leading_zeros=self.width_leading_zeros,
            lower_blocks=[b.T for b in self.blocks],
            width_trailing_zeros=self.height_trailing_zeros,
        )


@Diagonal.__matmul__.register
def _(self, other: UpperBlockDiagonal) -> Diagonal:
    if len(self.blocks) != len(other.blocks) + 1:
        raise ValueError("Number of blocks in the operands must match")
    return UpperBlockDiagonal(
        width_leading_zeros=other.width_leading_zeros,
        upper_blocks=[
            b @ other_b for b, other_b in zip(self.blocks[:-1], other.blocks)
        ],
        height_trailing_zeros=self.blocks[-1].height,
    )


@LowerBlockDiagonal.__matmul__.register
def _(self, other: UpperBlockDiagonal) -> Diagonal:
    if len(other.blocks) != len(self.blocks):
        raise ValueError("Number of blocks in the operands must match")
    return Diagonal(
        [torch.zeros(self.height_leading_zeros, other.width_leading_zeros)]
        + [b @ other_block for b, other_block in zip(self.blocks, other.blocks)]
    )
