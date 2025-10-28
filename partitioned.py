"""Implement operations on partitioned tensors."""

from typing import Any, Sequence
import torch


class Matrix:
    """Base class for matrix operations."""

    def __matmul__(self, other: Any) -> Any:
        """Matrix-vector or matrix-matrix multiplication."""
        return self.apply(other)


class BlockVector(Matrix):
    """Represents a block vector as a list of blocks."""

    def __init__(self, blocks: Sequence[torch.Tensor]):
        self.blocks = blocks

    def to_tensor(self) -> torch.Tensor:
        """Convert block vector to a single concatenated tensor."""
        return torch.cat(self.blocks)


class BlockDiagonalMatrix(Matrix):
    """Represents a block-diagonal matrix as a list of blocks."""

    def __init__(self, blocks: Sequence[torch.Tensor]):
        self.blocks = blocks

    def __neg__(self) -> "BlockDiagonalMatrix":
        """Return negation of the matrix."""
        return BlockDiagonalMatrix([-block for block in self.blocks])

    def apply(self, v: BlockVector) -> BlockVector:
        if len(self.blocks) != len(v.blocks):
            raise ValueError("Number of blocks in matrix and vector must match")

        return BlockVector(
            [block @ v_block for block, v_block in zip(self.blocks, v.blocks)]
        )

    def invert(self) -> "BlockDiagonalMatrix":
        """Return inverse of block-diagonal matrix."""
        return BlockDiagonalMatrix(list(map(torch.linalg.inv, self.blocks)))

    def solve(self, rhs: BlockVector) -> BlockVector:
        """Solve M x = rhs for block-diagonal M."""
        if len(self.blocks) != len(rhs.blocks):
            raise ValueError("Number of blocks in matrix and vector must match")

        return BlockVector(
            [
                torch.linalg.solve(block, rhs_block)
                for block, rhs_block in zip(self.blocks, rhs.blocks)
            ]
        )

    @property
    def T(self) -> "BlockDiagonalMatrix":
        """Return transpose of this matrix."""
        return BlockDiagonalMatrix([block.T for block in self.blocks])


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

    def __init__(self, lower_blocks: Sequence[torch.Tensor]):
        """
        Args:
            lower_blocks: lower diagonal blocks (L blocks, where first is typically zero/None)
                         These are the -∇_z f_ℓ terms. Can be a list of tensors or a BlockDiagonalMatrix.
        """
        self.lower_blocks = lower_blocks

    def apply(self, v: BlockVector) -> BlockVector:
        if len(v.blocks) != len(self.lower_blocks) + 1:
            raise ValueError("Number of blocks in vector and matrix must match")

        return BlockVector(
            [v.blocks[0]]
            + [
                self.lower_blocks[i] @ v.blocks[i] + v.blocks[i + 1]
                for i in range(len(self.lower_blocks))
            ]
        )

    def solve(self, rhs: BlockVector) -> BlockVector:
        """Solve M x = rhs using back-substitution."""
        # We have that
        #   x[0] = rhs[0]
        # and for i>0,
        #   lower_blocks[i] @ x[i] + x[i+1] = rhs[i+1]
        # so
        #   x[i+1] = rhs[i+1] - lower_blocks[i] @ x[i]

        result_blocks = [rhs.blocks[0]]
        for i in range(len(self.lower_blocks)):
            result_blocks.append(
                rhs.blocks[i + 1] - self.lower_blocks[i] @ result_blocks[-1]
            )
        return BlockVector(result_blocks)

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
        self.upper_blocks = upper_blocks

    def apply(self, v: BlockVector) -> BlockVector:
        if len(v.blocks) != len(self.upper_blocks) + 1:
            raise ValueError("Number of blocks in vector and matrix must match")

        return BlockVector(
            [
                v.blocks[i] + self.upper_blocks[i] @ v.blocks[i + 1]
                for i in range(len(self.upper_blocks))
            ]
            + [v.blocks[-1]]
        )

    def solve(self, rhs: BlockVector) -> BlockVector:
        """Solve M x = rhs using forward-substitution."""
        # We have for i<L,
        #   x[i] + upper_blocks[i] @ x[i+1] = rhs[i]
        # and
        #   x[L] = rhs[L]
        # So
        #   x[i] = rhs[i] - upper_blocks[i] @ x[i+1]

        result_blocks = [rhs.blocks[-1]]
        for i in range(len(self.upper_blocks) - 1, -1, -1):
            result_blocks.insert(
                0, rhs.blocks[i] - self.upper_blocks[i] @ result_blocks[0]
            )
        return BlockVector(result_blocks)

    @property
    def T(self) -> IdentityWithLowerBlockDiagonalMatrix:
        """Return transpose of this matrix."""
        return IdentityWithLowerBlockDiagonalMatrix(
            [block.T for block in self.upper_blocks]
        )


def downshift(v: BlockVector) -> torch.Tensor:
    """
    Apply the downshifting matrix P to a vector.

    P shifts blocks down by one position, inserting zeros at the top:
    P @ [v_0, v_1, v_2, ..., v_{L-1}] = [0, v_0, v_1, ..., v_{L-2}]

    Args:
        v: input vector (concatenated blocks, BlockVector, or PartitionedVector)
        block_sizes: sizes of each block (list), or single size for uniform blocks
        num_blocks: number of blocks (required if block_sizes is a single int)

    Returns:
        P @ v (downshifted vector)
    """
    return BlockVector([torch.zeros_like(v.blocks[0])] + v.blocks[:-1])


def upshift(v: BlockVector) -> BlockVector:
    """
    Apply the upshifting matrix Q to a vector.

    Q shifts blocks up by one position, removing the top block:
    Q @ [v_0, v_1, v_2, ..., v_{L-1}] = [v_1, v_2, ..., v_{L-1}, 0]
    """
    return BlockVector(v.blocks[1:] + [torch.zeros_like(v.blocks[-1])])
