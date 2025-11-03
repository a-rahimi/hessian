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

from typing import Sequence
import torch


class Matrix:
    """Base class for matrix operations."""

    pass


class TorchMatrix(torch.Tensor, Matrix):
    "Overloads torch.Tensor with invert() and solve() methods."

    def invert(self) -> "TorchMatrix":
        return TorchMatrix(torch.linalg.inv(self))

    def solve(self, rhs: "TorchMatrix") -> "TorchMatrix":
        return torch.linalg.solve(self, rhs)

    @staticmethod
    def wrap(tensor: torch.Tensor | Matrix, required_2d: bool = True) -> "TorchMatrix":
        if isinstance(tensor, Matrix):
            return tensor
        elif isinstance(tensor, torch.Tensor):
            if required_2d and tensor.ndim != 2:
                raise ValueError("Tensor must be a 2D tensor")
            return TorchMatrix(tensor)
        raise ValueError("Tensor must be a torch.Tensor or Matrix")


class Vertical(Matrix):
    """Represents a block vector as a list of blocks."""

    def __init__(self, blocks: Sequence[Matrix]):
        self.blocks = [TorchMatrix.wrap(block, required_2d=False) for block in blocks]

    def __neg__(self) -> "Vertical":
        """Return negation of the matrix."""
        return self.__class__([-block for block in self.blocks])

    def __add__(self, other: "Vertical") -> "Vertical":
        if not isinstance(other, Vertical):
            raise ValueError("Other must be a BlockDiagonalMatrix")
        if len(self.blocks) != len(other.blocks):
            raise ValueError("Number of blocks must match")
        return self.__class__(
            [
                block + other_block
                for block, other_block in zip(self.blocks, other.blocks)
            ]
        )

    def __sub__(self, other: "Diagonal") -> "Diagonal":
        if not isinstance(other, Diagonal):
            raise ValueError("Other must be a BlockDiagonalMatrix")
        if len(self.blocks) != len(other.blocks):
            raise ValueError("Number of blocks in matrices must match")
        return Diagonal(
            [
                block - other_block
                for block, other_block in zip(self.blocks, other.blocks)
            ]
        )

    def to_tensor(self) -> torch.Tensor:
        return torch.cat(self.blocks)


class Diagonal(Vertical):
    """Represents a block-diagonal matrix as a list of blocks."""

    def __init__(self, blocks: Sequence[Matrix]):
        self.blocks = list(map(TorchMatrix.wrap, blocks))

    def __matmul__(self, other: Matrix) -> Matrix:
        if len(self.blocks) != len(other.blocks):
            raise ValueError("Number of blocks must match")

        return other.__class__(
            [block @ oblock for block, oblock in zip(self.blocks, other.blocks)]
        )

    def invert(self) -> "Diagonal":
        return Diagonal([b.invert() for b in self.blocks])

    def solve(self, rhs: Vertical) -> Vertical:
        """Solve M x = rhs for block-diagonal M."""
        if len(self.blocks) != len(rhs.blocks):
            raise ValueError("Number of blocks in matrix and vector must match")

        return self.__class__(
            [
                block.solve(rhs_block)
                for block, rhs_block in zip(self.blocks, rhs.blocks)
            ]
        )

    @property
    def T(self) -> "Diagonal":
        """Return transpose of this matrix."""
        return Diagonal([block.T for block in self.blocks])


class IdentityWithLowerBlockDiagonalMatrix(Matrix):
    """Represents a block bi-diagonal matrix with identity on the diagonal and a
    lower-diagonal blocks.

    This represents the M matrix from the paper:
    M = [I
         -∇_z f_2  I
                   -∇_z f_3  I
                             ...
                                  -∇_z f_L  1]
    """

    def __init__(self, lower_blocks: Sequence[Matrix]):
        """
        Args:
            lower_blocks: lower diagonal blocks (L blocks, where first is typically zero/None)
                         These are the -∇_z f_ℓ terms. Can be a list of tensors or a BlockDiagonalMatrix.
        """
        self.lower_blocks = list(map(TorchMatrix.wrap, lower_blocks))

    def __matmul__(self, v: Vertical) -> Vertical:
        if len(v.blocks) != len(self.lower_blocks) + 1:
            raise ValueError("Number of blocks in vector and matrix must match")

        return Vertical(
            [v.blocks[0]]
            + [
                self.lower_blocks[i] @ v.blocks[i] + v.blocks[i + 1]
                for i in range(len(self.lower_blocks))
            ]
        )

    def solve(self, rhs: Vertical) -> Vertical:
        """Solve M x = rhs using back-substitution."""
        # We have that
        #   x[0] = rhs[0]
        # and for i>0,
        #   lower_blocks[i] @ x[i] + x[i+1] = rhs[i+1]
        # so
        #   x[i+1] = rhs[i+1] - lower_blocks[i] @ x[i]

        if len(rhs.blocks) != len(self.lower_blocks) + 1:
            raise ValueError("Number of blocks in vector and matrix must match")

        result_blocks = [rhs.blocks[0]]
        for i in range(len(self.lower_blocks)):
            result_blocks.append(
                rhs.blocks[i + 1] - self.lower_blocks[i] @ result_blocks[-1]
            )
        return Vertical(result_blocks)

    @property
    def T(self) -> "IdentityWithUpperBlockDiagonalMatrix":
        """Return transpose of this matrix."""
        return IdentityWithUpperBlockDiagonalMatrix(
            [block.T for block in self.lower_blocks]
        )


class IdentityWithUpperBlockDiagonalMatrix(Matrix):
    """Represents a block bi-diagonal matrix with identity on the diagonal and an upper-diagonal blocks.
    This represents the M^T matrix from the paper:
    M^T = [I         -∇_z f_2^T           0             ...          0
             0           I         -∇_z f_3^T        ...          0
             ...        ...           ...           ...          ...
             0           0           ...          I       -∇_z f_L^T
             0           0           ...          0            1]
    """

    def __init__(self, upper_blocks: Sequence[torch.Tensor]):
        self.upper_blocks = list(map(TorchMatrix.wrap, upper_blocks))

    def __matmul__(self, v: Vertical) -> Vertical:
        if len(v.blocks) != len(self.upper_blocks) + 1:
            raise ValueError("Number of blocks in vector and matrix must match")

        return Vertical(
            [
                v.blocks[i] + self.upper_blocks[i] @ v.blocks[i + 1]
                for i in range(len(self.upper_blocks))
            ]
            + [v.blocks[-1]]
        )

    def solve(self, rhs: Vertical) -> Vertical:
        """Solve M x = rhs using forward-substitution."""
        # We have for i<L,
        #   x[i] + upper_blocks[i] @ x[i+1] = rhs[i]
        # and
        #   x[L] = rhs[L]
        # So
        #   x[i] = rhs[i] - upper_blocks[i] @ x[i+1]

        if len(rhs.blocks) != len(self.upper_blocks) + 1:
            raise ValueError("Number of blocks in vector and matrix must match")

        result_blocks = [rhs.blocks[-1]]
        for i in range(len(self.upper_blocks) - 1, -1, -1):
            result_blocks.insert(
                0, rhs.blocks[i] - self.upper_blocks[i] @ result_blocks[0]
            )
        return Vertical(result_blocks)

    @property
    def T(self) -> IdentityWithLowerBlockDiagonalMatrix:
        """Return transpose of this matrix."""
        return IdentityWithLowerBlockDiagonalMatrix(
            [block.T for block in self.upper_blocks]
        )


class SymmetricBlock2x2Matrix(Matrix):
    """Represents a symmetric 2x2block matrix whose blocks are in turn Matrices."""

    def __init__(self, block11: Matrix, block12: Matrix, block22: Matrix):
        self.block11 = TorchMatrix.wrap(block11)
        self.block12 = TorchMatrix.wrap(block12)
        self.block22 = TorchMatrix.wrap(block22)

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

    def UDU_decomposition(
        self,
    ) -> tuple[IdentityWithUpperBlockDiagonalMatrix, Diagonal]:
        b22_inv = self.block22.invert()
        U = IdentityWithUpperBlockDiagonalMatrix([self.block12 @ b22_inv])
        D = Diagonal(
            [self.block11 - self.block12 @ b22_inv @ self.block12.T, self.block22]
        )
        return U, D


def downshift(v: Vertical, empty_block_size: torch.Size) -> Vertical:
    """
    Apply the downshifting matrix P to a vector.

    P shifts blocks down by one position, inserting zeros at the top:
    P @ [v_0, v_1, v_2, ..., v_{L-1}] = [0, v_0, v_1, ..., v_{L-2}]
    """
    return Vertical([torch.zeros(empty_block_size)] + v.blocks[:-1])


def upshift(v: Vertical, empty_block_size: torch.Size) -> Vertical:
    """
    Apply the transpose of the downshifting matrix P to a vector.
    """
    return Vertical(v.blocks[1:] + [torch.zeros(empty_block_size)])
