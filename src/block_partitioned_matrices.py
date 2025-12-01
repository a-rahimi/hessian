"""A library of operations on partitioned matrices.

The blocks of these matrices can in turn be block matrices themselves, or they
canbe torch tensors. This way, it becomes easy to express linear algebra
operations like matrix multiplication, inversion, and solving linear systems
with matrices that have hierarchical structure.  For example, if a matrix A has
a 2x2 block structre and each of these blocks are in turn block diagonal
matrices,
one can define

A = Symmetri2x2(
    # A11 is a block diagona matrix with two blocks.
    block11=Diagonal([torch.randn(3, 3), torch.randn(2, 2)]),
    # A12 is a blockdiagonal matrix with two blocks.
    block12=Diagonal([torch.randn(3, 2), torch.randn(2, 2)]),
    A A22 is just a dense matrix
    block22=torch.randn(2, 2)
)

One can then multiply A by a vector v with A @ v, or solve the linear system A @
x = b with x = A.solve(b), or recover the inverse of A with A.invert().

The package provides placeholder matrices for Zero and Identity that don't take up
any space in memory or compute other than for their metadata.

Here is the class hierarchy:

Matrix
├── Tensor (also torch.Tensor)
├── Identity
├── Zero
└── Ragged
    ├── Generic
    │   ├── Vertical
    │   └── Horizontal
    ├── Symmetric2x2
    └── Tridiagonal
        ├── SymmetricTriDiagonal
        ├── LowerBiDiagonal
        │   ├── IdentityWithLowerDiagonal
        │   └── LowerDiagonal
        ├── UpperBiDiagonal
        │   ├── IdentityWithUpperDiagonal
        │   └── UpperDiagonal
        └── Diagonal
"""

from typing import Any, Callable, Iterator, Sequence
from functools import singledispatchmethod, cached_property

import numpy as np
import torch


class Matrix:
    "Base class for partitioned matrices."

    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    def invert(self) -> "Matrix":
        raise NotImplementedError

    def solve(self, rhs: "Matrix") -> "Matrix":
        raise NotImplementedError

    def __matmul__(self, other: "Matrix") -> "Matrix":
        raise NotImplementedError

    def __add__(self, other: "Matrix") -> "Matrix":
        raise NotImplementedError

    def __sub__(self, other: "Matrix") -> "Matrix":
        raise NotImplementedError

    @property
    def width(self) -> int:
        raise NotImplementedError

    @property
    def height(self) -> int:
        raise NotImplementedError

    @property
    def T(self) -> "Matrix":
        raise NotImplementedError


class Tensor(torch.Tensor, Matrix):
    "Endows torch.Tensor with extra matrix operations."

    def __init__(self, *args):
        torch.Tensor.__init__(*args)
        if self.ndim != 2:
            raise ValueError("Tensor must be a 2D tensor")

    def invert(self) -> "Tensor":
        return Tensor(torch.linalg.inv(self))

    def solve(self, rhs: "Tensor") -> "Tensor":
        return torch.linalg.solve(self, rhs)

    def to_tensor(self) -> torch.Tensor:
        return self

    @singledispatchmethod
    def __matmul__(self, other: Matrix) -> Matrix:
        return Tensor.wrap(torch.Tensor.__matmul__(self, other))

    @singledispatchmethod
    def __add__(self, other: Matrix) -> Matrix:
        return Tensor.wrap(torch.Tensor.__add__(self, other))

    @singledispatchmethod
    def __sub__(self, other: Matrix) -> Matrix:
        return Tensor.wrap(torch.Tensor.__sub__(self, other))

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[0]

    @staticmethod
    def wrap(tensor: torch.Tensor | Matrix) -> "Tensor":
        if isinstance(tensor, Matrix):
            return tensor
        elif isinstance(tensor, torch.Tensor):
            if tensor.ndim != 2:
                raise ValueError("Tensor must be a 2D tensor")
            return Tensor(tensor)
        raise ValueError("Tensor must be a torch.Tensor or Matrix")


class Identity(Matrix):
    def __init__(self, dimension: int = 0):
        self.dimension = dimension

    def __matmul__(self, other: Matrix) -> Matrix:
        return other

    def solve(self, rhs: Matrix) -> Matrix:
        return rhs

    def invert(self) -> "Identity":
        return self

    def to_tensor(self) -> torch.Tensor:
        return torch.eye(self.dimension)

    def __eq__(self, other: Matrix) -> bool:
        return isinstance(other, Identity) and self.dimension == other.dimension

    def __add__(self, other: Matrix) -> Matrix:
        return other + self

    @property
    def T(self) -> "Identity":
        return self

    @property
    def width(self) -> int:
        return self.dimension

    @property
    def height(self) -> int:
        return self.dimension


@Tensor.__matmul__.register
def _(self, _: Identity) -> Tensor:
    return self


class Zero(Matrix):
    def __init__(self, shape: tuple[int, int] = ()):
        self.shape = shape

    def __matmul__(self, other: Matrix) -> Matrix:
        if not self.shape:
            return Tensor(torch.tensor([[0.0]]))
        if other.height != self.shape[1]:
            raise ValueError(f"Shape mismatch {self} vs {other.height} x {other.width}")
        return torch.zeros(self.shape[0], other.width)

    def to_tensor(self) -> torch.Tensor:
        return torch.zeros(*self.shape)

    def invert(self) -> "Zero":
        raise ValueError("Zero matrix is not invertible")

    def solve(self, rhs: "Matrix") -> "Matrix":
        raise ValueError("Zero matrix is not invertible")

    def __eq__(self, other: Matrix) -> bool:
        return isinstance(other, Zero) and self.shape == other.shape

    def __add__(self, other: Matrix) -> Matrix:
        if other.width != self.width or other.height != self.height:
            raise ValueError(f"Shape mismatch {self} vs {other.height} x {other.width}")
        return other

    def __sub__(self, other: Matrix) -> Matrix:
        return self.__add__(other)

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def T(self) -> "Zero":
        if not self.shape:
            return Zero()
        return Zero((self.shape[1], self.shape[0]))


@Tensor.__matmul__.register
def _(self, other: Zero) -> Tensor:
    if self.width != other.height:
        raise ValueError(f"Shape mismatch {self} vs {other.height} x {other.width}")
    return Zero((self.shape[0], other.width))


@Tensor.__add__.register
def _(self, other: Zero) -> Tensor:
    if self.width != other.width or self.height != other.height:
        raise ValueError(f"Shape mismatch {self} vs {other.height} x {other.width}")
    return self


@Tensor.__sub__.register
def _(self, other: Zero) -> Tensor:
    return self.__add__(other)


def reshape_to_2d_list(lst: Sequence[Any], shape: tuple[int, int]) -> list[list[Any]]:
    if len(lst) != shape[0] * shape[1]:
        raise ValueError(f"Length of list {len(lst)} does not match shape {shape}")
    return [lst[row * shape[1] : (row + 1) * shape[1]] for row in range(shape[0])]


class Ragged(Matrix):
    """An unstructured block-partitioned matrix.

    Each block in this matrix can in turn be a matrix. You can index into the
    blocks, traverse them, and reshape the block structure.
    """

    def __init__(self, blocks: Sequence[Sequence[Matrix]]):
        self.blocks = [list(map(Tensor.wrap, row)) for row in blocks]

    def __neg__(self) -> "Ragged":
        return self.apply_unary_operation(lambda m: -m)

    @singledispatchmethod
    def __add__(self, other: Matrix) -> Matrix:
        raise NotImplementedError  # Special cases implemented below.

    @singledispatchmethod
    def __sub__(self, other: Matrix) -> Matrix:
        raise NotImplementedError  # Special cases implemented below.

    def flatten(self) -> Iterator[Matrix]:
        "Iterate over all the blocks in the matrix in row-major order."
        for row in self.blocks:
            yield from row

    @cached_property
    def flat(self) -> list[Matrix]:
        return list(self.flatten())

    def num_blocks(self) -> int:
        return len(self.flat)

    def apply_unary_operation(self, op: Callable[[Matrix], Matrix]) -> list[Matrix]:
        return self.__class__(Ragged([[op(b) for b in row] for row in self.blocks]))

    def apply_binary_operation(
        self, other: "Ragged", op: Callable[[Matrix, Matrix], Matrix]
    ) -> "Ragged":
        if len(self.blocks) != len(other.blocks):
            raise ValueError("Number of rows in the operands must match")
        result_blocks = []
        for row, other_row in zip(self.blocks, other.blocks):
            if len(row) != len(other_row):
                raise ValueError("Number of columns in each row must match")
            result_blocks.append([op(b, b_other) for b, b_other in zip(row, other_row)])

        return self.__class__(Ragged(result_blocks))


@Ragged.__add__.register
def _(self, other: Ragged) -> Ragged:
    return self.apply_binary_operation(other, lambda m1, m2: m1 + m2)


@Ragged.__sub__.register
def _(self, other: Ragged) -> Ragged:
    return self.apply_binary_operation(other, lambda m1, m2: m1 - m2)


class Generic(Ragged):
    """A ragged array whose rows have the same number of columns."""

    @singledispatchmethod
    def __init__(self, blocks: Sequence[Sequence[Matrix]], validate=True):
        super().__init__(blocks)
        self.shape = (len(blocks), len(blocks[0]))
        if validate:
            self.validate()

    @__init__.register
    def _(self, blocks: Ragged, validate=True):
        self.__init__(blocks.flat, validate)

    def validate(self):
        # Ensure the matrix isn't ragged.
        for row in self.blocks:
            if len(row) != len(self.blocks[0]):
                raise ValueError("All rows must have the same length")

        # Ensure the blocks have compatible shapes. The blocks in a row must
        # have  the same height, and the blocks in a column must have the same width.
        for r in range(self.shape[0]):
            heights = np.array([b.height for b in self[r, :].flatten()])
            if not np.all(heights == heights[0]):
                raise ValueError("All blocks in row must have the same height")

        for c in range(self.shape[1]):
            widths = np.array([b.width for b in self[:, c].flatten()])
            if not np.all(widths == widths[0]):
                raise ValueError("All blocks in column must have the same width")

    def __getitem__(self, index: Any) -> Matrix | "Horizontal" | "Vertical" | "Generic":
        if not isinstance(index, tuple):
            raise ValueError("Index must be a tuple of two elements (row, column)")
        row, col = index

        if isinstance(row, int) and isinstance(col, int):
            # Return an element
            return self.blocks[row][col]

        if isinstance(row, int):
            # col is a slice. Return a horizontally stacked matrix.
            return Horizontal(self.blocks[row][col], validate=False)

        if isinstance(col, int):
            # row is a slice. Return a vertically stacked matrix.
            return Vertical([row[col] for row in self.blocks[row]], validate=False)

        # Both row and col are slices. Return a block matrix.
        return Generic([row[col] for row in self.blocks[row]], validate=False)

    def reshape(self, shape: tuple[int, int]) -> "Generic":
        return Generic(reshape_to_2d_list(list(self.flatten()), shape))

    def to_tensor(self) -> torch.Tensor:
        return torch.vstack(
            [torch.hstack([b.to_tensor() for b in row]) for row in self.blocks]
        )

    @property
    def width(self) -> int:
        # Sum up the width of the blocks of the first row.
        return sum(b.width for b in self.blocks[0])

    @property
    def height(self) -> int:
        # Sum up the height of the blocks of the first column.
        return sum(row[0].height for row in self.blocks)


class Symmetric2x2(Ragged):
    """Represents a symmetric 2x2 block matrix whose blocks are in turn Matrices."""

    def __init__(self, block11: Matrix, block12: Matrix, block22: Matrix):
        super().__init__([[block11, block12], [block22]])

    @property
    def block11(self) -> Matrix:
        return self.blocks[0][0]

    @property
    def block12(self) -> Matrix:
        return self.blocks[0][1]

    @property
    def block21(self) -> Matrix:
        return self.block12.T

    @property
    def block22(self) -> Matrix:
        return self.blocks[1][0]

    @singledispatchmethod
    def __matmul__(self, v: Matrix) -> Matrix:
        raise NotImplementedError  # Special cases implemented below.

    def invert(self) -> "Symmetric2x2":
        # S = UDU^T
        U, D = self.UDU_decomposition()

        # S^{-1} = U^-T D^-1 U^-1
        #   =   [I      0]  [D0^{-1}      0  ]   [I  -U0]  = [D0^{-1}        -D0^{-1} U0               ]
        #       [-U0^T  I]  [   0     D1^{-1}]   [0    I]    [-U0^T D0^{-1}   U0^T D0^{-1} U0 + D1^{-1}]
        Dinv = D.invert()
        block12 = -Dinv.flat[0] @ U.upper_blocks[0]
        return Symmetric2x2(
            block11=Dinv.flat[0],
            block12=block12,
            block22=Dinv.flat[1] - U.upper_blocks[0].T @ block12,
        )

    def invert_via_LDL(self) -> "Symmetric2x2":
        # S = LDL^T
        L, D = self.LDL_decomposition()

        # S^{-1} = L^-T D^-1 L^-1
        #   =   [I      -L0']  [D0^{-1}      0  ]   [I     0]  = [D0^{-1} + L0' D1^{-1} L0          -L0' D1^{-1} ]
        #       [0        I ]  [   0     D1^{-1}]   [-L0   I]    [-D1^{-1} L0                        D1^{-1}     ]
        Dinv = D.invert()
        block12 = -L.lower_blocks[0].T @ Dinv.flat[1]
        return Symmetric2x2(
            block11=Dinv.flat[0]
            + L.lower_blocks[0].T @ Dinv.flat[1] @ L.lower_blocks[0],
            block12=block12,
            block22=Dinv.flat[1],
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
    ) -> tuple[IdentityWithUpperDiagonal, Diagonal]:
        b22_inv = self.block22.invert()
        U = IdentityWithUpperDiagonal([self.block12 @ b22_inv])
        D = Diagonal(
            [self.block11 - self.block12 @ b22_inv @ self.block12.T, self.block22]
        )
        return U, D

    def LDL_decomposition(self) -> tuple[IdentityWithLowerDiagonal, Diagonal]:
        a11_inv = self.block11.invert()
        L = IdentityWithLowerDiagonal([self.block12.T @ a11_inv])
        D = Diagonal(
            [self.block11, self.block22 - self.block12.T @ a11_inv @ self.block12]
        )
        return L, D


class Vertical(Generic):
    """Blocks stacked vertically."""

    @singledispatchmethod
    def __init__(self, blocks: Sequence[Matrix], validate=True):
        super().__init__([[b] for b in blocks], validate)
        if self.shape[1] != 1:
            raise ValueError("Vertical matrix must have exactly one column")

    @__init__.register
    def _(self, blocks: Ragged, validate=True):
        self.__init__(blocks.flat, validate=validate)

    def blockwise_transpose(self) -> "Vertical":
        """Assuming this object has blocks v_ij, return a new Vertical object with blocks v_ji."""

        # For the operation to make sense, every block v_i must have the
        # the same number of sub-blocks. Ensure this is the case.
        num_subblocks = self.flat[0].num_blocks()
        for b in self.flat:
            if b.num_blocks() != num_subblocks:
                raise ValueError("All blocks must have the same number of sub-blocks")

        return Vertical(
            [
                Vertical([self.flat[i].flat[j] for i in range(self.num_blocks())])
                for j in range(num_subblocks)
            ]
        )


@Symmetric2x2.__matmul__.register
def __matmul__(self, v: Vertical) -> Vertical:
    if v.num_blocks() != 2:
        raise ValueError("Number of blocks in vector and matrix must match")
    return Vertical(
        [
            self.block11 @ v.flat[0] + self.block12 @ v.flat[1],
            self.block21 @ v.flat[0] + self.block22 @ v.flat[1],
        ]
    )


class Horizontal(Generic):
    """Blocks stacked horizontally."""

    @singledispatchmethod
    def __init__(self, blocks: Sequence[Matrix], validate=True):
        super().__init__([blocks], validate)
        if self.shape[0] != 1:
            raise ValueError("Horizontal matrix must have exactly one row")

    @__init__.register
    def _(self, blocks: Ragged, validate=True):
        self.__init__(blocks.flat, validate=validate)


class Tridiagonal(Ragged):
    @singledispatchmethod
    def __init__(
        self,
        diagonal_blocks: Sequence[Matrix],
        lower_blocks: Sequence[Matrix],
        upper_blocks: Sequence[Matrix],
    ):
        if lower_blocks != [] and (len(diagonal_blocks) != len(lower_blocks) + 1):
            raise ValueError(
                "Number of diagonal blocks must be one more than the number of lower blocks"
            )
        if upper_blocks != [] and (len(diagonal_blocks) != len(upper_blocks) + 1):
            raise ValueError(
                "Number of diagonal blocks must be one more than the number of upper blocks"
            )
        super().__init__([diagonal_blocks, lower_blocks, upper_blocks])

    @__init__.register
    def _(self, diagonal_blocks: Ragged):
        if len(diagonal_blocks.blocks) != 3:
            raise ValueError("Ragged input must have exactly three rows")
        self.__init__(*diagonal_blocks.blocks)

    @property
    def diagonal_blocks(self) -> Sequence[Matrix]:
        return self.blocks[0]

    @property
    def lower_blocks(self) -> Sequence[Matrix]:
        return self.blocks[1]

    @property
    def upper_blocks(self) -> Sequence[Matrix]:
        return self.blocks[2]

    @singledispatchmethod
    def __matmul__(self, other: Matrix):
        raise NotImplementedError

    @__matmul__.register
    def _(self, v: Vertical) -> Vertical:
        L = LowerDiagonal(
            self.diagonal_blocks[0].height,
            self.lower_blocks,
            self.diagonal_blocks[-1].width,
        )
        D = Diagonal(self.diagonal_blocks)
        U = UpperDiagonal(
            self.diagonal_blocks[0].width,
            self.upper_blocks,
            self.diagonal_blocks[-1].height,
        )
        return L @ v + D @ v + U @ v

    @staticmethod
    def from_matrix(matrix: Matrix) -> "Tridiagonal":
        if isinstance(matrix, Tridiagonal):
            return matrix
        if isinstance(matrix, Diagonal):
            return Tridiagonal(
                diagonal_blocks=matrix.blocks,
                lower_blocks=[Zero()] * (len(matrix.blocks) - 1),
                upper_blocks=[Zero()] * (len(matrix.blocks) - 1),
            )
        if isinstance(matrix, LowerBiDiagonal):
            return Tridiagonal(
                diagonal_blocks=matrix.diagonal_blocks,
                lower_blocks=matrix.lower_blocks,
                upper_blocks=[Zero()] * (len(matrix.diagonal_blocks) - 1),
            )
        if isinstance(matrix, UpperBiDiagonal):
            return Tridiagonal(
                diagonal_blocks=matrix.diagonal_blocks,
                lower_blocks=[Zero()] * (len(matrix.diagonal_blocks) - 1),
                upper_blocks=matrix.upper_blocks,
            )
        raise ValueError(
            f"Input matrix must be subclass of a diagonal class. Got {type(matrix)}"
        )

    @property
    def height(self) -> int:
        return sum(b.height for b in self.diagonal_blocks)

    @property
    def width(self) -> int:
        return sum(b.width for b in self.diagonal_blocks)

    @staticmethod
    def blockwise_transpose(M: Generic) -> "Tridiagonal":
        """Compute the blockwise transpose of matrix whose blocks are tridiagonal.

        M be a block matrix. Denote the uv'th element of its ij'th block by M_{ij, uv}.
        The blockwise transpsoe of M is a block matrix whose uv'th element of its ij'th
        block is M_{vu, ji}.

        The blocks of M are presumbed to be tridiagonal. That means M_{ij, uv} = 0
        whenever |u-v| > 1. That means the blockwise tranpose of M satisfies M_{ij} = 0
        whenever |i-j| > 1. In other words, M is block-tridiagonal.
        """
        # Ensure all blocks are tridiagonal.
        generic_tridiagonal_blocks = []
        num_diags = None
        for block in M.flatten():
            if num_diags is None:
                num_diags = len(block.diagonal_blocks)
            if len(block.diagonal_blocks) != num_diags:
                raise ValueError(
                    "All diagonal blocks must have the same number of blocks"
                )
            generic_tridiagonal_blocks.append(Tridiagonal.from_matrix(block))
        M = Generic([generic_tridiagonal_blocks]).reshape(M.shape)

        return Tridiagonal(
            [
                Generic([[b.diagonal_blocks[i] for b in M.flatten()]]).reshape(M.shape)
                for i in range(num_diags)
            ],
            lower_blocks=[
                Generic([[b.lower_blocks[i] for b in M.flatten()]]).reshape(M.shape)
                for i in range(num_diags - 1)
            ],
            upper_blocks=[
                Generic([[b.upper_blocks[i] for b in M.flatten()]]).reshape(M.shape)
                for i in range(num_diags - 1)
            ],
        )

    def solve(self, v: Vertical) -> Vertical:
        L, D, U = self.LDU_decomposition()
        return U.solve(D.solve(L.solve(v)))

    def LDU_decomposition(
        self,
    ) -> tuple["IdentityWithLowerDiagonal", "Diagonal", "IdentityWithUpperDiagonal"]:
        "Factorize into L D U, where L is lower block-diagonal, U is upper block-diagonal, and D is diagonal."

        # An illustration of block LDU decomposition:
        #
        #   T = L D U
        #
        #    [ D₀    U₀     0     ...               ]   =   [ I   0   0   ...           ] [ B₀  0   0  ...          ] [  I   V₀   0   ...          ]
        #    [ L₀    D₁     U₁    ...               ]       [ K₀  I   0   ...           ] [  0  B₁  0  ...          ] [  0   I   V₁   ...          ]
        #    [ 0     L₁     D₂      ...             ]       [ 0   K₁  I   ...           ] [  0   0  B₂ ...          ] [  0   0    I   ...          ]
        #                 ...                                                           ...
        #    [ 0     0    0       ... D_{n-2} U_{n-2}  ]    [ 0   0    0 ... I   0      ] [ 0 0 0 ... B_{n-2} 0     ] [ 0 ... 0        I   V_{n-2} ]
        #    [ 0     0    0       ... L_{n-2} D_{n-1}  ]    [ 0   0    0 ... K_{n-2} I  ] [ 0 0 0 ...    0  B_{n-1} ] [ 0 ...          0   I      ]
        #
        #
        # We have the base case
        #   B[0] = D[0]
        #
        # For i < n-1, we also have
        #   U[i] = B[i] @ V[i]
        #   L[i] = K[i] @ B[i]
        #   D[i+1] = B[i+1] + K[i] @ B[i] @ V[i]
        #
        # These imply, respectively,
        #
        #   V[i] = B[i] \ U[i]
        #   K[i] = (B[i].T \ L[i].T).T
        #   B[i+1] = D[i+1] - K[i] @ B[i] @ V[i]

        Bs = [self.diagonal_blocks[0]]
        Vs = []
        Ks = []
        for U, L, Dnext in zip(
            self.upper_blocks, self.lower_blocks, self.diagonal_blocks[1:]
        ):
            Vs.append(Bs[-1].solve(U))
            Ks.append(Bs[-1].T.solve(L.T).T)
            Bs.append(Dnext - Ks[-1] @ Bs[-1] @ Vs[-1])

        return (
            IdentityWithLowerDiagonal(Ks),
            Diagonal(Bs),
            IdentityWithUpperDiagonal(Vs),
        )

    def to_tensor(self) -> torch.Tensor:
        L = LowerDiagonal(
            height_leading_zeros=self.diagonal_blocks[0].height,
            lower_blocks=self.lower_blocks,
            width_trailing_zeros=self.diagonal_blocks[-1].width,
        )
        U = UpperDiagonal(
            width_leading_zeros=self.diagonal_blocks[0].width,
            upper_blocks=self.upper_blocks,
            height_trailing_zeros=self.diagonal_blocks[-1].height,
        )
        D = Diagonal(self.diagonal_blocks)
        return D.to_tensor() + L.to_tensor() + U.to_tensor()


class SymmetricTriDiagonal(Tridiagonal):
    """
    A symmetric block tridiagonal matrix, with diagonal_blocks D and lower_blocks L,
    i.e. the blocks of the matrix look like

    [ D_0  L_0^T   0      0   ... ]
    [ L_0  D_1   L_1^T    0   ... ]
    [  0   L_1   D_2    L_2^T ... ]
    [ ...                        ]

    """

    def __init__(
        self, lower_blocks: Sequence[Matrix], diagonal_blocks: Sequence[Matrix]
    ):
        if len(lower_blocks) + 1 != len(diagonal_blocks):
            raise ValueError(
                "Number of lower blocks must be one less than the number of diagonal blocks"
            )

        super().__init__(diagonal_blocks, lower_blocks, [])

    @cached_property
    def upper_blocks(self) -> list[Matrix]:
        return [b.T for b in self.lower_blocks]

    @cached_property
    def __matmul__(self, other: Matrix) -> Matrix:
        raise NotImplementedError

    @singledispatchmethod
    def __add__(self, other: Matrix) -> Matrix:
        raise NotImplementedError

    def UDU_decomposition(self) -> tuple["UpperBiDiagonal", "Diagonal"]:
        "Factorize into U D U^T, where U is upper block-diagonal, and D is diagonal."

        # An illustration of block UDUᵗ decomposition:
        #
        #   T = U D Uᵗ
        #
        #    [ D₀    L₀ᵗ    0     ...               ]   =   [ I  U₀   0   ...           ] [ B₀  0   0  ...          ] [  I   0   0   ...          ]
        #    [ L₀    D₁   L₁ᵗ     ...               ]       [ 0   I   U₁  ...           ] [  0  B₁  0  ...          ] [ U₀ᵗ  I   0   ...          ]
        #    [ 0     L₁   D₂      ...               ]       [ 0   0    I  ...           ] [  0   0  B₂ ...          ] [  0  U₁ᵗ  I   ...          ]
        #                 ...                                                           ...
        #    [ 0     0    0       ... D_{n-2} L_{n-2}ᵗ ]    [ 0   0    0 ... I U_{n-2}  ] [ 0 0 0 ... B_{n-2} 0     ] [ 0 ... U_{n-3}ᵗ        I  0 ]
        #    [ 0     0    0       ... L_{n-2} D_{n-1}  ]    [ 0   0    0 ... 0      I   ] [ 0 0 0 ...    0  B_{n-1} ] [ 0 ...          U_{n-2}ᵗ  I ]
        #
        #
        # We have the base case
        #   B[n-1] = D[n-1]
        #
        # For i < n-1, we also have
        #   L[i-1] = B[i] @ U[i-1].T
        #   D[i] = B[i] + U[i] @ B[i+1] @ U[i].T
        #
        # These imply, respectively,
        #
        #   U[i] = (B[i+1] \ L[i]).T
        #   B[i] = D[i] - U[i] B[i+1] @ U[i].T

        Bs = [self.diagonal_blocks[-1]]
        Us = []
        for L, D in zip(self.lower_blocks[::-1], self.diagonal_blocks[-2::-1]):
            Us.insert(0, (Bs[0].solve(L)).T)
            Bs.insert(0, D - Us[0] @ Bs[0] @ Us[0].T)

        return UpperBiDiagonal(
            upper_blocks=Us,
            diagonal_blocks=[Identity(b.width) for b in Bs],
        ), Diagonal(Bs)


class LowerBiDiagonal(Tridiagonal):
    """Represents a block bi-diagonal matrix.

    [D_0    0
     L_1  D_2    0
          L_1  D_3
              ...    0
               L_N  D_{N - 1}]
    """

    def __init__(
        self, lower_blocks: Sequence[Matrix], diagonal_blocks: Sequence[Matrix]
    ):
        if len(lower_blocks) != len(diagonal_blocks) - 1:
            raise ValueError(
                "Number of lower blocks must be one less than the number of diagonal blocks"
            )
        diagonal_blocks = list(map(Tensor.wrap, diagonal_blocks))

        super().__init__(
            diagonal_blocks,
            lower_blocks,
            upper_blocks=[
                Zero((D.height, Dnext.width))
                for D, Dnext in zip(diagonal_blocks[:-1], diagonal_blocks[1:])
            ],
        )

    @singledispatchmethod
    def __matmul__(self, other: Matrix) -> Matrix:
        return Tridiagonal.__matmul__(self, other)

    @__matmul__.register
    def _(self, v: Vertical) -> Vertical:
        if v.num_blocks() != len(self.diagonal_blocks):
            raise ValueError("Number of blocks in vector and matrix must match")

        return Vertical(
            [self.diagonal_blocks[0] @ v.flat[0]]
            + [
                self.lower_blocks[i] @ v.flat[i]
                + self.diagonal_blocks[i + 1] @ v.flat[i + 1]
                for i in range(len(self.lower_blocks))
            ]
        )

    def solve(self, rhs: Vertical) -> Vertical:
        """Solve M x = rhs using back-substitution."""
        if rhs.num_blocks() != len(self.diagonal_blocks):
            raise ValueError("Number of blocks in vector and matrix must match")

        # We have that
        #   diagonal_blocks[0] @ x[0] = rhs[0]
        # so
        #   x[0] = diagonal_block[0] \ rhs[0]
        # For i>0, we have
        #   lower_blocks[i] @ x[i] + diagonal_blocks[i+1] @ x[i+1] = rhs[i+1]
        # so
        #   x[i+1] = diagonal_blocks[i+1] \ (rhs[i+1] - lower_blocks[i] @ x[i])
        result_blocks = [self.diagonal_blocks[0].solve(rhs.flat[0])]
        for i in range(len(self.lower_blocks)):
            result_blocks.append(
                self.diagonal_blocks[i + 1].solve(
                    rhs.flat[i + 1] - self.lower_blocks[i] @ result_blocks[i]
                )
            )
        return Vertical(result_blocks)

    @property
    def T(self) -> "UpperBiDiagonal":
        return UpperBiDiagonal(
            upper_blocks=[b.T for b in self.lower_blocks],
            diagonal_blocks=[b.T for b in self.diagonal_blocks],
        )

    def to_tensor(self) -> torch.Tensor:
        return (
            LowerDiagonal(
                height_leading_zeros=self.diagonal_blocks[0].height,
                lower_blocks=self.lower_blocks,
                width_trailing_zeros=self.diagonal_blocks[-1].width,
            ).to_tensor()
            + Diagonal(self.diagonal_blocks).to_tensor()
        )


class IdentityWithLowerDiagonal(LowerBiDiagonal):
    def __init__(self, lower_blocks: Sequence[Matrix]):
        lower_blocks = [Tensor.wrap(b) for b in lower_blocks]
        super().__init__(
            lower_blocks=lower_blocks,
            diagonal_blocks=[Identity(L.width) for L in lower_blocks]
            + [Identity(lower_blocks[-1].height)],
        )


class UpperBiDiagonal(Tridiagonal):
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
                "Number of upper blocks must be the number of diagonal blocks minus one"
            )
        diagonal_blocks = list(map(Tensor.wrap, diagonal_blocks))
        super().__init__(
            diagonal_blocks,
            lower_blocks=[
                Zero((Dnext.height, D.width))
                for D, Dnext in zip(diagonal_blocks[:-1], diagonal_blocks[1:])
            ],
            upper_blocks=upper_blocks,
        )

    @singledispatchmethod
    def __matmul__(self, v: Matrix) -> Matrix:
        raise NotImplementedError  # Special cases implemented below.

    @__matmul__.register
    def __matmul__(self, v: Vertical) -> Vertical:
        if v.num_blocks() != len(self.diagonal_blocks):
            raise ValueError("Number of blocks in vector and matrix must match")

        return Vertical(
            [
                self.diagonal_blocks[i] @ v.flat[i]
                + self.upper_blocks[i] @ v.flat[i + 1]
                for i in range(len(self.upper_blocks))
            ]
            + [self.diagonal_blocks[-1] @ v.flat[-1]]
        )

    def solve(self, rhs: Vertical) -> Vertical:
        """Solve M x = rhs using forward-substitution."""
        if rhs.num_blocks() != len(self.diagonal_blocks):
            raise ValueError("Number of blocks in vector and matrix must match")

        # We have for i<N,
        #   diagonal_blocks[i] @ x[i] + upper_blocks[i] @ x[i+1] = rhs[i]
        # so
        #   x[i] = diagonal_blocks[i] \ (rhs[i] - upper_blocks[i] @ x[i+1])
        # For i=N, we have
        #   diagonal_blocks[N] @ x[N] = rhs[N]
        # so
        #   x[N] = diagonal_blocks[N] \ rhs[N]
        result_blocks = [self.diagonal_blocks[-1].solve(rhs.flat[-1])]
        for i in range(len(self.upper_blocks) - 1, -1, -1):
            result_blocks.insert(
                0,
                self.diagonal_blocks[i].solve(
                    rhs.flat[i] - self.upper_blocks[i] @ result_blocks[0]
                ),
            )
        return Vertical(result_blocks)

    @property
    def T(self) -> LowerBiDiagonal:
        return LowerBiDiagonal(
            lower_blocks=[b.T for b in self.upper_blocks],
            diagonal_blocks=[b.T for b in self.diagonal_blocks],
        )

    def to_tensor(self) -> torch.Tensor:
        return (
            UpperDiagonal(
                width_leading_zeros=self.diagonal_blocks[0].width,
                upper_blocks=self.upper_blocks,
                height_trailing_zeros=self.diagonal_blocks[-1].height,
            ).to_tensor()
            + Diagonal(self.diagonal_blocks).to_tensor()
        )


class IdentityWithUpperDiagonal(UpperBiDiagonal):
    def __init__(self, upper_blocks: Sequence[Matrix]):
        upper_blocks = [Tensor.wrap(b) for b in upper_blocks]
        super().__init__(
            upper_blocks=upper_blocks,
            diagonal_blocks=[Identity(L.height) for L in upper_blocks]
            + [Identity(upper_blocks[-1].width)],
        )


class Diagonal(Tridiagonal):
    @singledispatchmethod
    def __init__(self, diagonal_blocks: Sequence[Matrix]):
        super().__init__(diagonal_blocks, [], [])

    @__init__.register
    def _(self, diagonal_blocks: Ragged):
        self.__init__(diagonal_blocks.flat)

    @property
    def upper_blocks(self) -> list[Matrix]:
        raise NotImplementedError

    @property
    def lower_blocks(self) -> list[Matrix]:
        raise NotImplementedError

    def invert(self) -> "Diagonal":
        return Diagonal([b.invert() for b in self.diagonal_blocks])

    def solve(self, rhs: Vertical | "Diagonal") -> Vertical | "Diagonal":
        if self.num_blocks() != rhs.num_blocks():
            raise ValueError("Number of blocks in matrix and vector must match")

        return rhs.__class__(
            [
                block.solve(rhs_block)
                for block, rhs_block in zip(self.flatten(), rhs.flatten())
            ]
        )

    def to_tensor(self) -> torch.Tensor:
        return torch.block_diag(*[b.to_tensor() for b in self.flatten()])

    @singledispatchmethod
    def __matmul__(self, other: Matrix) -> Matrix:
        raise NotImplementedError  # Special cases implemented below.

    @singledispatchmethod
    def __add__(self, other: Matrix) -> Matrix:
        raise NotImplementedError  # Special cases implemented below.

    @singledispatchmethod
    def __sub__(self, other: Matrix) -> Matrix:
        raise NotImplementedError  # Special cases implemented below.

    @property
    def T(self) -> "Diagonal":
        return self.apply_unary_operation(lambda block: block.T)


@Diagonal.__matmul__.register
def _(self, other: Diagonal) -> Diagonal:
    return self.apply_binary_operation(other, lambda m1, m2: m1 @ m2)


@Diagonal.__matmul__.register
def _(self, other: Vertical) -> Vertical:
    return Vertical(self @ Diagonal(other))


@Diagonal.__add__.register
def _(self, other: Diagonal) -> Diagonal:
    return Ragged.__add__(self, other)  # Forwards to the parent.


@Diagonal.__add__.register
def _(self, _: Identity) -> Diagonal:
    "Diagonal + I"
    return Diagonal([b + torch.eye(b.shape[0]) for b in self.flatten()])


@Diagonal.__sub__.register
def _(self, other: Diagonal) -> Diagonal:
    return Ragged.__sub__(self, other)  # Forwards to the parent.


@Diagonal.__sub__.register
def _(self, _: Identity) -> Diagonal:
    "Diagonal - I"
    return Diagonal([b - torch.eye(b.shape[0]) for b in self.flatten()])


@SymmetricTriDiagonal.__add__.register
def _(self, other: Diagonal) -> "SymmetricTriDiagonal":
    if len(self.diagonal_blocks) != len(other.blocks):
        raise ValueError("Number of blocks in the operands must match")

    # T + D just adds to the diagonal blocks.
    return SymmetricTriDiagonal(
        lower_blocks=self.lower_blocks,
        diagonal_blocks=(Diagonal(self.diagonal_blocks) + other).flat,
    )


class LowerDiagonal(LowerBiDiagonal):
    def __init__(
        self,
        height_leading_zeros: int,
        lower_blocks: Sequence[Matrix],
        width_trailing_zeros: int,
    ):
        lower_blocks = list(map(Tensor.wrap, lower_blocks))
        diagonal_blocks = []
        h = height_leading_zeros
        for L in lower_blocks:
            diagonal_blocks.append(Zero((h, L.width)))
            h = L.height
        diagonal_blocks.append(Zero((h, width_trailing_zeros)))

        super().__init__(diagonal_blocks=diagonal_blocks, lower_blocks=lower_blocks)
        self.height_leading_zeros = height_leading_zeros
        self.width_trailing_zeros = width_trailing_zeros

    @property
    def upper_blocks(self) -> list[Matrix]:
        raise NotImplementedError

    @singledispatchmethod
    def __matmul__(self, other: Matrix) -> Matrix:
        raise NotImplementedError

    @__matmul__.register
    def _(self, other: Vertical) -> Vertical:
        if other.num_blocks() != len(self.diagonal_blocks):
            raise ValueError("Number of blocks in the operands must match")

        # L v amounts to L[:] v[:-1], with a zero tacked on at the start.
        return Vertical(
            [Zero((self.height_leading_zeros, other.width))]
            + (Diagonal(self.lower_blocks) @ Vertical(other.flat[:-1])).flat,
        )

    def to_tensor(self) -> torch.Tensor:
        t = torch.zeros(self.height, self.width)

        row, col = self.height_leading_zeros, 0
        for L in self.lower_blocks:
            t[row : row + L.height, col : col + L.width] = L.to_tensor()
            row += L.height
            col += L.width

        return t

    @property
    def T(self) -> "UpperDiagonal":
        return UpperDiagonal(
            width_leading_zeros=self.height_leading_zeros,
            upper_blocks=[b.T for b in self.lower_blocks],
            height_trailing_zeros=self.width_trailing_zeros,
        )


@LowerDiagonal.__matmul__.register
def _(self, other: Diagonal) -> "LowerDiagonal":
    if len(other.num_blocks()) != len(self.diagonal_blocks):
        raise ValueError("Number of blocks in the operands must match")

    # L D is lower diagonal, with entries L[:] D[:-1].
    return LowerDiagonal(
        height_leading_zeros=self.height_leading_zeros,
        lower_blocks=(Diagonal(self.lower_blocks) @ Diagonal(other.flat[:-1])).flat,
        width_trailing_zeros=other.flat[-1].shape[1],
    )


@Diagonal.__matmul__.register
def _(self, other: LowerDiagonal) -> Diagonal:
    if self.num_blocks() != len(other.diagonal_blocks):
        raise ValueError("Number of blocks in the operands must match")

    # D L is lower diagonal, with entries D[1:] L[:], and zero tacked on at
    return LowerDiagonal(
        height_leading_zeros=self.flat[0].height,
        lower_blocks=(Diagonal(self.flat[1:]) @ Diagonal(other.lower_blocks)).flat,
        width_trailing_zeros=other.width_trailing_zeros,
    )


def downshifting_matrix(
    height_leading_zeros, v_heights: Sequence[int]
) -> LowerDiagonal:
    return LowerDiagonal(
        height_leading_zeros=height_leading_zeros,
        lower_blocks=list(map(Identity, v_heights[:-1])),
        width_trailing_zeros=v_heights[-1],
    )


class UpperDiagonal(UpperBiDiagonal):
    def __init__(
        self,
        width_leading_zeros: int,
        upper_blocks: Sequence[Matrix],
        height_trailing_zeros: int,
    ):
        upper_blocks = list(map(Tensor.wrap, upper_blocks))

        # TODO: use list comprehension to make this tighter.
        diagonal_blocks = []
        w = width_leading_zeros
        for U in upper_blocks:
            diagonal_blocks.append(Zero((U.height, w)))
            w = U.width
        diagonal_blocks.append(Zero((height_trailing_zeros, w)))

        super().__init__(diagonal_blocks=diagonal_blocks, upper_blocks=upper_blocks)
        self.height_trailing_zeros = height_trailing_zeros
        self.width_leading_zeros = width_leading_zeros

    @singledispatchmethod
    def __matmul__(self, other: Matrix) -> Matrix:
        raise NotImplementedError

    @__matmul__.register
    def _(self, other: Diagonal) -> Matrix:
        if other.num_blocks() != len(self.diagonal_blocks):
            raise ValueError("Number of blocks in the operands must match")

        # U D is upper diagonal, with entries U[:] D[1:]
        return UpperDiagonal(
            width_leading_zeros=other.flat[0].shape[1],
            upper_blocks=(Diagonal(self.upper_blocks) @ Diagonal(other.flat[1:])).flat,
            height_trailing_zeros=self.height_trailing_zeros,
        )

    @__matmul__.register
    def _(self, other: Vertical) -> Vertical:
        if other.num_blocks() != len(self.diagonal_blocks):
            raise ValueError("Number of blocks in the operands must match")

        # U v amounts U[:] v[1:], with a zero tacked on at the end.
        return Vertical(
            (Diagonal(self.upper_blocks) @ Vertical(other.flat[1:])).flat
            + [Zero((self.height_trailing_zeros, other.width))]
        )

    @__matmul__.register
    def _(self, other: LowerDiagonal) -> Diagonal:
        if other.num_blocks() != self.num_blocks():
            raise ValueError("Number of blocks in the operands must match")

        # U L is lower diagonal, with entries U[:] L[:], and zero tacked on at
        # the end.
        return Diagonal(
            (Diagonal(self.upper_blocks) @ Diagonal(other.lower_blocks)).flat
            + [Zero((self.height_trailing_zeros, other.width_trailing_zeros))]
        )

    def to_tensor(self) -> torch.Tensor:
        t = torch.zeros(self.height, self.width)
        row, col = 0, self.width_leading_zeros
        for U in self.upper_blocks:
            t[row : row + U.height, col : col + U.width] = U.to_tensor()
            row += U.height
            col += U.width
        return t

    @property
    def T(self) -> LowerDiagonal:
        return LowerDiagonal(
            height_leading_zeros=self.width_leading_zeros,
            lower_blocks=[b.T for b in self.upper_blocks],
            width_trailing_zeros=self.height_trailing_zeros,
        )


@Diagonal.__matmul__.register
def _(self, other: UpperDiagonal) -> Diagonal:
    if self.num_blocks() != len(other.diagonal_blocks):
        raise ValueError("Number of blocks in the operands must match")

    # D U is upper diagonal, with entries D[:-1] U[:].
    return UpperDiagonal(
        width_leading_zeros=other.width_leading_zeros,
        upper_blocks=(Diagonal(self.flat[:-1]) @ Diagonal(other.upper_blocks)).flat,
        height_trailing_zeros=self.flat[-1].height,
    )


@LowerDiagonal.__matmul__.register
def _(self, other: UpperDiagonal) -> Diagonal:
    if len(other.blocks) != len(self.blocks):
        raise ValueError("Number of blocks in the operands must match")

    # L U is diagonal, with entries L[:] U[:].
    return Diagonal(
        [Zero((self.height_leading_zeros, other.width_leading_zeros))]
        + (Diagonal(self.lower_blocks) @ Diagonal(other.upper_blocks)).flat
    )
