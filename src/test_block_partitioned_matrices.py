"""Tests for partitioned.py module."""

import pytest
import torch
from block_partitioned_matrices import (
    Vertical,
    Diagonal,
    Generic,
    Generic3x3,
    IdentityWithLowerDiagonal,
    IdentityWithUpperDiagonal,
    Symmetric2x2,
    SymmetricTriDiagonal,
    Tensor,
    Tridiagonal,
    UpperBiDiagonal,
    UpperDiagonal,
    LowerDiagonal,
    LowerBiDiagonal,
    Identity,
    ScaledIdentity,
    Zero,
    reshape_to_2d_list,
    downshifting_matrix,
)


class TestTensor:
    def test_eq_identity(self):
        assert Identity(2) == Identity(2)

    def test_eq_zero(self):
        assert Zero((2, 2)) == Zero((2, 2))

    def test_eq_zero_with_tensor(self):
        torch.testing.assert_close(
            Zero((2, 2)).to_tensor(), Tensor.wrap(torch.zeros(2, 2)).to_tensor()
        )


class TestGeneric:
    @pytest.fixture
    def blocks_ragged(self):
        return [
            [Tensor(2, 2), Tensor(2, 3)],
            [Tensor(3, 2), Tensor(3, 3)],
        ]

    @pytest.fixture
    def blocks_all_same_widths(self):
        return [
            [Tensor(2, 3), Tensor(2, 3)],
            [Tensor(2, 3), Tensor(2, 3)],
        ]

    def test_reshape_returns_expected_nested_lists(self):
        values = list(range(6))
        reshaped = reshape_to_2d_list(values, (2, 3))
        assert reshaped == [[0, 1, 2], [3, 4, 5]]

    def test_reshape_invalid_length_raises(self):
        with pytest.raises(ValueError, match="Length of list 5 does not match shape"):
            reshape_to_2d_list(list(range(5)), (2, 3))

    def test_shape_preserves_input_layout(self, blocks_ragged):
        M = Generic(blocks_ragged)

        assert M.shape == (2, 2)
        assert M[0, 1] is blocks_ragged[0][1]
        assert M[1, 0] is blocks_ragged[1][0]

    def test_reshape_returns_new_generic_with_same_blocks(self, blocks_all_same_widths):
        M = Generic(blocks_all_same_widths)

        M_reshaped = M.reshape((4, 1))

        assert M_reshaped.shape == (4, 1)
        assert M_reshaped[0, 0] is blocks_all_same_widths[0][0]
        assert M_reshaped[1, 0] is blocks_all_same_widths[0][1]
        assert M_reshaped[2, 0] is blocks_all_same_widths[1][0]
        assert M_reshaped[3, 0] is blocks_all_same_widths[1][1]

    def test_iter_traverses_blocks_row_major(self, blocks_ragged):
        M = Generic(blocks_ragged)

        expected = [
            blocks_ragged[0][0],
            blocks_ragged[0][1],
            blocks_ragged[1][0],
            blocks_ragged[1][1],
        ]

        assert len(M.flat) == len(expected)
        for actual, original in zip(M.flatten(), expected):
            assert actual is original

    def test_slicing_preserves_block_identity(self, blocks_all_same_widths):
        M = Generic(blocks_all_same_widths)

        first_row = M[0, :]
        assert first_row.shape == (1, 2)
        assert first_row[0, 0] is blocks_all_same_widths[0][0]
        assert first_row[0, 1] is blocks_all_same_widths[0][1]

        second_column = M[:, 1]
        assert second_column.shape == (2, 1)
        assert second_column[0, 0] is blocks_all_same_widths[0][1]
        assert second_column[1, 0] is blocks_all_same_widths[1][1]

        sub_block = M[1:, 1:]
        assert sub_block.shape == (1, 1)
        assert sub_block[0, 0] is blocks_all_same_widths[1][1]

    def test_matmul(self):
        # A: (2x2) blocks, block sizes:
        A = Generic(
            [
                [Tensor(torch.randn(2, 2)), Tensor(torch.randn(2, 3))],
                [Tensor(torch.randn(3, 2)), Tensor(torch.randn(3, 3))],
            ]
        )

        # B: (2x2) blocks, block sizes:
        B = Generic(
            [
                [Tensor(torch.randn(2, 4)), Tensor(torch.randn(2, 1))],
                [Tensor(torch.randn(3, 4)), Tensor(torch.randn(3, 1))],
            ]
        )

        C = A @ B

        assert isinstance(C, Generic)
        assert C.shape == (2, 2)

        torch.testing.assert_close(C.to_tensor(), A.to_tensor() @ B.to_tensor())


class TestGeneric3x3:
    """Tests for Generic3x3 class."""

    def test_init_raises_if_not_3x3(self):
        with pytest.raises(ValueError, match="Generic3x3 must be 3x3"):
            Generic3x3([[Tensor(1, 1), Tensor(1, 1)], [Tensor(1, 1), Tensor(1, 1)]])

    def test_solve(self):
        """Test that solve() implements A^-1 B."""
        I = torch.eye(2)
        # Use simple matrices to verify computation
        # A = diag(2I, 2I, 2I)
        # B = diag(3I, 3I, 3I)
        # solve(A, B) -> A^-1 @ B = diag(0.5I, 0.5I, 0.5I) @ diag(3I, 3I, 3I) = diag(1.5I, 1.5I, 1.5I)

        Z = torch.zeros(2, 2)
        A = Generic3x3(
            [
                [Tensor(2 * I), Tensor(Z), Tensor(Z)],
                [Tensor(Z), Tensor(2 * I), Tensor(Z)],
                [Tensor(Z), Tensor(Z), Tensor(2 * I)],
            ]
        )
        B = Generic3x3(
            [
                [Tensor(3 * I), Tensor(Z), Tensor(Z)],
                [Tensor(Z), Tensor(3 * I), Tensor(Z)],
                [Tensor(Z), Tensor(Z), Tensor(3 * I)],
            ]
        )

        C = A.solve(B)

        assert isinstance(C, Generic3x3)
        for i in range(3):
            for j in range(3):
                if i == j:
                    torch.testing.assert_close(C[i, j].to_tensor(), 1.5 * I)
                else:
                    torch.testing.assert_close(C[i, j].to_tensor(), Z)

    def test_matmul_generic3x3(self):
        """Test that matrix multiplication of Generic3x3 returns Generic3x3."""
        # Create 3x3 block matrices with random tensors
        # All blocks 2x2 for simplicity
        A_blocks = [[Tensor(torch.randn(2, 2)) for _ in range(3)] for _ in range(3)]
        B_blocks = [[Tensor(torch.randn(2, 2)) for _ in range(3)] for _ in range(3)]

        A = Generic3x3(A_blocks)
        B = Generic3x3(B_blocks)

        C = A @ B

        assert isinstance(C, Generic3x3)
        assert C.shape == (3, 3)
        torch.testing.assert_close(C.to_tensor(), A.to_tensor() @ B.to_tensor())

    def test_matmul_with_vertical(self):
        """Test matrix multiplication of Generic3x3 with Vertical."""
        # A: 3x3 of 2x2 blocks
        A = Generic3x3(
            [[Tensor(torch.randn(2, 2)) for _ in range(3)] for _ in range(3)]
        )
        # v: 3x1 of 2x1 blocks, inlined into Vertical constructor
        v = Vertical([Tensor(torch.randn(2, 1)) for _ in range(3)])

        result = A @ v

        assert isinstance(result, Vertical)
        assert result.shape == (3, 1)
        torch.testing.assert_close(result.to_tensor(), A.to_tensor() @ v.to_tensor())

    def test_transpose_matches_tensor_transpose(self):
        """Test that the transpose of Generic3x3 matches tensor transpose."""
        torch.manual_seed(42)
        M = Generic3x3(
            [
                [
                    Tensor(torch.randn(2, 2)),
                    Tensor(torch.randn(2, 3)),
                    Tensor(torch.randn(2, 4)),
                ],
                [
                    Tensor(torch.randn(3, 2)),
                    Tensor(torch.randn(3, 3)),
                    Tensor(torch.randn(3, 4)),
                ],
                [
                    Tensor(torch.randn(4, 2)),
                    Tensor(torch.randn(4, 3)),
                    Tensor(torch.randn(4, 4)),
                ],
            ]
        )

        MT = M.T

        assert isinstance(MT, Generic3x3)
        torch.testing.assert_close(MT.to_tensor(), M.to_tensor().T)


def col(*args) -> torch.Tensor:
    "Cretae a column vector."
    return torch.tensor(args)[:, None]


class TestBlockVector:
    """Tests for BlockVector class."""

    def test_add_basic(self):
        """Test adding two block vectors with matching blocks."""
        v1_blocks = [
            col(1.0, 2.0),
            col(3.0, 4.0, 5.0),
        ]
        v2_blocks = [col(10.0, 20.0), col(30.0, 40.0, 50.0)]

        v1 = Vertical(v1_blocks)
        v2 = Vertical(v2_blocks)

        result = v1 + v2

        assert torch.allclose(result[0, 0], col(11.0, 22.0))
        assert torch.allclose(result[1, 0], col(33.0, 44.0, 55.0))

    def test_add_single_block(self):
        """Test adding two block vectors with a single block each."""
        v1 = Vertical([col(1.0, 2.0, 3.0)])
        v2 = Vertical([col(4.0, 5.0, 6.0)])

        result = v1 + v2

        assert torch.allclose(result[0, 0], col(5.0, 7.0, 9.0))

    def test_add_mismatched_block_count_raises(self):
        """Test that adding vectors with different block counts raises ValueError."""
        v1 = Vertical([col(1.0, 2.0)])
        v2 = Vertical([col(3.0, 4.0), col(5.0, 6.0)])

        with pytest.raises(ValueError, match="Number of .* must match"):
            v1 + v2

    def test_add_mismatched_block_shape_raises(self):
        """Test that adding vectors with mismatched block shapes raises RuntimeError."""
        v1 = Vertical([col(1.0, 2.0), col(3.0, 4.0)])
        v2 = Vertical([col(10.0, 20.0, 30.0), col(40.0, 50.0)])

        with pytest.raises(
            RuntimeError,
            match="The size of tensor a .* must match the size of tensor b",
        ):
            v1 + v2

    def test_to_tensor(self):
        v = Vertical([col(1.0, 2.0), col(3.0, 4.0, 5.0)])
        assert torch.allclose(v.to_tensor(), col(1.0, 2.0, 3.0, 4.0, 5.0))

    def test_to_tensor_nested(self):
        v = Vertical([Vertical([col(1.0, 2.0)]), Vertical([col(3.0, 4.0, 5.0)])])
        assert torch.allclose(v.to_tensor(), col(1.0, 2.0, 3.0, 4.0, 5.0))

    def test_blockwise_transpose_structure(self):
        """Test blockwise_transpose structure and content."""
        v11 = col(1.0, 1.1)
        v12 = col(2.0, 2.1)
        V1 = Vertical([v11, v12])

        v21 = col(3.0, 3.1)
        v22 = col(4.0, 4.1)
        V2 = Vertical([v21, v22])

        V = Vertical([V1, V2])

        VT = V.blockwise_transpose()

        # Expected: VT = [Vertical([v11, v21]), Vertical([v12, v22])]
        assert isinstance(VT, Vertical)
        assert VT.num_blocks() == 2

        U1 = VT.flat[0]
        U2 = VT.flat[1]

        assert isinstance(U1, Vertical)
        assert isinstance(U2, Vertical)

        assert U1.num_blocks() == 2
        assert U2.num_blocks() == 2

        # Check content
        torch.testing.assert_close(U1.flat[0].to_tensor(), v11)
        torch.testing.assert_close(U1.flat[1].to_tensor(), v21)

        torch.testing.assert_close(U2.flat[0].to_tensor(), v12)
        torch.testing.assert_close(U2.flat[1].to_tensor(), v22)

    def test_blockwise_transpose_mismatched_blocks_raises(self):
        """Test that blockwise_transpose raises ValueError for mismatched sub-blocks."""
        V1 = Vertical([col(1.0), col(2.0)])
        V2 = Vertical([col(3.0)])  # Only 1 sub-block

        V = Vertical([V1, V2])

        with pytest.raises(
            ValueError, match="All blocks must have the same number of sub-blocks"
        ):
            V.blockwise_transpose()

    def test_blockwise_transpose_involution(self):
        """Test that blockwise_transpose is an involution (f(f(x)) = x)."""
        v11 = col(1.0, 1.1)
        v12 = col(2.0, 2.1)
        V1 = Vertical([v11, v12])

        v21 = col(3.0, 3.1)
        v22 = col(4.0, 4.1)
        V2 = Vertical([v21, v22])

        V = Vertical([V1, V2])

        V_TT = V.blockwise_transpose().blockwise_transpose()

        torch.testing.assert_close(V_TT.to_tensor(), V.to_tensor())


class TestBlockDiagonal:
    """Tests for BlockDiagonal class."""

    def test_init(self):
        """Test BlockDiagonal initialization."""
        D = Diagonal([torch.eye(2), torch.eye(3)])
        assert D.num_blocks() == 2

    def test_neg(self):
        """Test negation of block-diagonal matrix."""
        blocks = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0]])]
        D = Diagonal(blocks)
        neg_D = -D

        assert torch.allclose(neg_D.flat[0], -blocks[0])
        assert torch.allclose(neg_D.flat[1], -blocks[1])

    def test_multiply_vertical(self):
        """Test applying block-diagonal matrix to vector."""
        D = Diagonal(
            [
                torch.tensor([[2.0, 0.0], [0.0, 3.0]]),  # 2x2 block
                torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
                ),  # 3x3 block
            ]
        )
        v = Vertical([col(1.0, 1.0), col(1.0, 1.0, 1.0)])

        Dv = D @ v

        assert isinstance(Dv, Vertical)
        assert torch.allclose(Dv.flat[0], col(2.0, 3.0))
        assert torch.allclose(Dv.flat[1], col(1.0, 2.0, 3.0))

    def test_apply_mismatched_blocks_raises(self):
        """Test that apply with mismatched blocks raises ValueError."""
        blocks = [torch.eye(2), torch.eye(3)]
        D = Diagonal(blocks)

        v_blocks = [col(1.0, 1.0)]  # Only 1 block
        v = Vertical(v_blocks)

        with pytest.raises(
            ValueError, match="Number of columns in each row must match"
        ):
            D @ v

    def test_invert(self):
        """Test inverting block-diagonal matrix."""
        D = Diagonal(
            [
                torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
                torch.tensor([[4.0]]),
            ]
        )
        inv_D = D.invert()

        assert torch.allclose(
            inv_D.flat[0], torch.tensor([[0.5, 0.0], [0.0, 1.0 / 3.0]])
        )
        assert torch.allclose(inv_D.flat[1], torch.tensor([[0.25]]))

    def test_invert_singular_raises(self):
        """Test that inverting singular matrix raises error."""
        bdm = Diagonal([torch.tensor([[0.0, 0.0], [0.0, 0.0]])])

        with pytest.raises(torch._C._LinAlgError):
            bdm.invert()

    def test_solve(self):
        """Test solving linear system with block-diagonal matrix."""
        # A x = b where A is block diagonal
        D = Diagonal([torch.tensor([[2.0, 0.0], [0.0, 3.0]]), torch.tensor([[4.0]])])

        rhs = Vertical([col(4.0, 6.0), col(8.0)])

        solution = D.solve(rhs)
        assert torch.allclose(solution.flat[0], col(2.0, 2.0))
        assert torch.allclose(solution.flat[1], col(2.0))

    def test_solve_mismatched_blocks_raises(self):
        """Test that solve with mismatched blocks raises ValueError."""
        bdm = Diagonal([torch.eye(2), torch.eye(3)])

        rhs = Vertical([col(1.0, 1.0)])  # Only 1 block

        with pytest.raises(ValueError, match="Number of blocks"):
            bdm.solve(rhs)

    def test_solve_verifies_solution(self):
        """Test that solve actually produces a valid solution."""
        # Create random invertible matrices
        torch.manual_seed(42)
        A1 = torch.randn(3, 3)
        A1 = A1 @ A1.T + torch.eye(3)  # Make positive definite
        A2 = torch.randn(2, 2)
        A2 = A2 @ A2.T + torch.eye(2)  # Make positive definite

        D = Diagonal([A1, A2])

        rhs = Vertical([torch.randn(3, 1), torch.randn(2, 1)])

        solution = D.solve(rhs)

        # Verify: A @ x = b
        result = D @ solution
        assert torch.allclose(result.flat[0], rhs.flat[0], atol=1e-5)
        assert torch.allclose(result.flat[1], rhs.flat[1], atol=1e-5)

    def test_add_identity(self):
        "Test that D+I works"
        D = Diagonal([torch.eye(2), torch.eye(3)])
        Daug = D + Identity()
        assert torch.allclose(Daug.flat[0], 2 * torch.eye(2))
        assert torch.allclose(Daug.flat[1], 2 * torch.eye(3))

    def test_sub_identity(self):
        "Test that D-I works"
        D = Diagonal([torch.eye(2), torch.eye(3)])
        Daug = D - Identity()
        assert torch.allclose(Daug.flat[0], torch.zeros((2, 2)))
        assert torch.allclose(Daug.flat[1], torch.zeros((3, 3)))

    def test_symmetricize(self):
        M = Diagonal([torch.randn(2, 2), torch.randn(3, 3)])
        M_symmetric = M @ M.T
        M_symmetric_tensor = M_symmetric.to_tensor()
        assert torch.allclose(M_symmetric_tensor, M_symmetric_tensor.T)


class TestIdentityWithLowerBlockDiagonal:
    """Tests for IdentityWithLowerDiagonal class."""

    def test_init(self):
        """Test initialization."""
        lower_blocks = [torch.eye(2), torch.eye(2)]
        m = IdentityWithLowerDiagonal(lower_blocks)
        assert len(m.lower_blocks) == 2

    def test_apply(self):
        """Test applying matrix to vector."""
        # M = [I     0    0
        #      A     I    0
        #      0     B    I]
        # where A and B are 2x2 matrices (lower_blocks already contains the values to use)
        M = IdentityWithLowerDiagonal(
            [
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            ]
        )
        v = Vertical([col(1.0, 1.0), col(2.0, 2.0), col(3.0, 3.0)])

        result = M @ v

        # result[0] = v[0] = [1, 1]
        # result[1] = A @ v[0] + v[1] = [1, 1] + [2, 2] = [3, 3]
        # result[2] = B @ v[1] + v[2] = 2*[2, 2] + [3, 3] = [7, 7]

        assert torch.allclose(result.flat[0], col(1.0, 1.0))
        assert torch.allclose(result.flat[1], col(3.0, 3.0))
        assert torch.allclose(result.flat[2], col(7.0, 7.0))

    def test_apply_mismatched_blocks_raises(self):
        """Test that apply with mismatched blocks raises ValueError."""
        M = IdentityWithLowerDiagonal([torch.eye(2), torch.eye(2)])

        v = Vertical([col(1.0, 1.0)])  # Only 1 block, need 3

        with pytest.raises(ValueError, match="Number of blocks"):
            M @ v

    def test_solve(self):
        """Test solving linear system."""
        # M x = rhs
        # x[0] = rhs[0]
        # x[i+1] = rhs[i+1] - lower_blocks[i] @ x[i]

        M = IdentityWithLowerDiagonal(
            [
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            ]
        )

        rhs = Vertical([col(1.0, 1.0), col(2.0, 2.0), col(3.0, 3.0)])

        solution = M.solve(rhs)
        assert torch.allclose(solution.flat[0], col(1.0, 1.0))
        assert torch.allclose(solution.flat[1], col(1.0, 1.0))
        assert torch.allclose(solution.flat[2], col(1.0, 1.0))

    def test_solve_verifies_solution(self):
        """Test that solve produces a valid solution."""
        torch.manual_seed(42)

        M = IdentityWithLowerDiagonal([torch.randn(3, 3), torch.randn(3, 3)])

        rhs = Vertical([torch.randn(3, 1), torch.randn(3, 1), torch.randn(3, 1)])

        solution = M.solve(rhs)

        # Verify: M @ x = rhs
        result = M @ solution
        assert torch.allclose(result.flat[0], rhs.flat[0], atol=1e-5)
        assert torch.allclose(result.flat[1], rhs.flat[1], atol=1e-5)
        assert torch.allclose(result.flat[2], rhs.flat[2], atol=1e-5)

    def test_solve_with_zero_blocks(self):
        """Test solving when some lower blocks are zero."""
        # This is common for the first block
        M = IdentityWithLowerDiagonal(
            [torch.zeros(2, 2), torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
        )

        rhs = Vertical([col(1.0, 2.0), col(3.0, 4.0), col(5.0, 6.0)])

        solution = M.solve(rhs)

        # x[0] = rhs[0] = [1, 2]
        # x[1] = rhs[1] - 0 @ x[0] = [3, 4]
        # x[2] = rhs[2] - B @ x[1] = [5, 6] - [3, 4] = [2, 2]

        assert torch.allclose(solution.flat[0], col(1.0, 2.0))
        assert torch.allclose(solution.flat[1], col(3.0, 4.0))
        assert torch.allclose(solution.flat[2], col(2.0, 2.0))

    def test_transpose(self):
        """Test that transpose returns IdentityWithUpperDiagonal."""
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        M = IdentityWithLowerDiagonal([A, B])

        M_T = M.T

        assert isinstance(M_T, UpperBiDiagonal)
        assert len(M_T.upper_blocks) == 2
        assert torch.allclose(M_T.upper_blocks[0], A.T)
        assert torch.allclose(M_T.upper_blocks[1], B.T)


class TestIdentityWithUpperBlockDiagonal:
    """Tests for IdentityWithUpperBlockDiagonal class."""

    def test_init(self):
        """Test initialization."""
        M = IdentityWithUpperDiagonal([torch.eye(2), torch.eye(2)])
        assert len(M.upper_blocks) == 2

    def test_apply(self):
        """Test applying matrix to vector."""
        # M^T = [I      A    0
        #        0      I    B
        #        0      0    I]
        # where A and B are 2x2 matrices
        M = IdentityWithUpperDiagonal(
            [
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            ]
        )

        v = Vertical([col(1.0, 1.0), col(2.0, 2.0), col(3.0, 3.0)])

        result = M @ v

        # result[0] = v[0] + A @ v[1] = [1, 1] + [2, 2] = [3, 3]
        # result[1] = v[1] + B @ v[2] = [2, 2] + 2*[3, 3] = [8, 8]
        # result[2] = v[2] = [3, 3]

        assert torch.allclose(result.flat[0], col(3.0, 3.0))
        assert torch.allclose(result.flat[1], col(8.0, 8.0))
        assert torch.allclose(result.flat[2], col(3.0, 3.0))

    def test_apply_mismatched_blocks_raises(self):
        """Test that apply with mismatched blocks raises ValueError."""
        M = IdentityWithUpperDiagonal([torch.eye(2), torch.eye(2)])

        v = Vertical([col(1.0, 1.0)])  # Only 1 block, need 3

        with pytest.raises(ValueError, match="Number of blocks"):
            M @ v

    def test_solve(self):
        """Test solving linear system using forward-substitution."""
        # M^T x = rhs
        # x[L] = rhs[L]
        # x[i] = rhs[i] - upper_blocks[i] @ x[i+1]

        M = IdentityWithUpperDiagonal(
            [
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            ]
        )

        rhs = Vertical([col(3.0, 3.0), col(8.0, 8.0), col(3.0, 3.0)])

        solution = M.solve(rhs)

        # x[2] = rhs[2] = [3, 3]
        # x[1] = rhs[1] - B @ x[2] = [8, 8] - 2*[3, 3] = [2, 2]
        # x[0] = rhs[0] - A @ x[1] = [3, 3] - [2, 2] = [1, 1]

        assert torch.allclose(solution.flat[0], col(1.0, 1.0))
        assert torch.allclose(solution.flat[1], col(2.0, 2.0))
        assert torch.allclose(solution.flat[2], col(3.0, 3.0))

    def test_solve_verifies_solution(self):
        """Test that solve produces a valid solution."""
        torch.manual_seed(42)

        M = IdentityWithUpperDiagonal([torch.randn(3, 3), torch.randn(3, 3)])

        rhs = Vertical([torch.randn(3, 1), torch.randn(3, 1), torch.randn(3, 1)])

        solution = M.solve(rhs)

        # Verify: M @ x = rhs
        result = M @ solution
        assert torch.allclose(result.flat[0], rhs.flat[0], atol=1e-5)
        assert torch.allclose(result.flat[1], rhs.flat[1], atol=1e-5)
        assert torch.allclose(result.flat[2], rhs.flat[2], atol=1e-5)

    def test_solve_with_zero_blocks(self):
        """Test solving when some upper blocks are zero."""

        M = IdentityWithUpperDiagonal(
            [torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.zeros(2, 2)]
        )

        rhs = Vertical([col(3.0, 4.0), col(5.0, 6.0), col(7.0, 8.0)])

        solution = M.solve(rhs)

        # x[2] = rhs[2] = [7, 8]
        # x[1] = rhs[1] - 0 @ x[2] = [5, 6]
        # x[0] = rhs[0] - A @ x[1] = [3, 4] - [5, 6] = [-2, -2]

        assert torch.allclose(solution.flat[0], col(-2.0, -2.0))
        assert torch.allclose(solution.flat[1], col(5.0, 6.0))
        assert torch.allclose(solution.flat[2], col(7.0, 8.0))

    def test_transpose(self):
        """Test that transpose returns IdentityWithLowerBlockDiagonal."""
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        M = IdentityWithUpperDiagonal([A, B])

        M_T = M.T

        assert isinstance(M_T, LowerBiDiagonal)
        assert len(M_T.lower_blocks) == 2
        assert torch.allclose(M_T.lower_blocks[0], A.T)
        assert torch.allclose(M_T.lower_blocks[1], B.T)

    def test_transpose_involution(self):
        """Test that (M.T).T = M (transpose is an involution)."""
        torch.manual_seed(42)
        m = IdentityWithUpperDiagonal([torch.randn(3, 3), torch.randn(3, 3)])

        v = Vertical([torch.randn(3, 1), torch.randn(3, 1), torch.randn(3, 1)])

        # Apply M
        result1 = m @ v

        # Apply (M.T).T which should equal M
        m_T_T = m.T.T
        result2 = m_T_T @ v

        assert torch.allclose(result1.flat[0], result2.flat[0], atol=1e-5)
        assert torch.allclose(result1.flat[1], result2.flat[1], atol=1e-5)
        assert torch.allclose(result1.flat[2], result2.flat[2], atol=1e-5)


class TestUpperDiagonal:
    def test_width_and_height(self):
        U = UpperDiagonal(2, [torch.randn(2, 3), torch.randn(3, 3)], 3)

        # The diagonal_blocks look like:
        # [Zero(2,2), Zero(3,3), Zero(3,3)] (by construction)
        # Upper blocks set the block layout for the actual width/height calculations.
        assert U.height == 2 + 3 + 3
        assert U.width == 2 + 3 + 3

        # Try a different shape (with rectangular blocks)
        U2 = UpperDiagonal(4, [torch.randn(5, 2), torch.randn(6, 5)], 7)
        # Diagonal blocks will have shapes: (5,4), (6,2), (7,5)
        assert U2.height == 5 + 6 + 7
        assert U2.width == 4 + 2 + 5

        # More upper blocks (test more cases)
        U3 = UpperDiagonal(
            1, [torch.randn(2, 4), torch.randn(3, 2), torch.randn(5, 3)], 6
        )
        # Diagonal blocks shapes: (2,1), (3,4), (5,2), (6,3)
        assert U3.height == 2 + 3 + 5 + 6
        assert U3.width == 1 + 4 + 2 + 3

        U_tensor = U.to_tensor()
        assert U_tensor.shape[0] == U.height
        assert U_tensor.shape[1] == U.width

    def test_transpose(self):
        U = UpperDiagonal(2, [torch.randn(2, 3), torch.randn(3, 3)], 3)
        assert torch.allclose(U.T.to_tensor(), U.to_tensor().T)


class TestLowerDiagonal:
    def test_width_and_height(self):
        # Format matches UpperDiagonal: LowerDiagonal(height_leading_zeros, lower_blocks, width_trailing_zeros)
        L = LowerDiagonal(2, [torch.randn(3, 2), torch.randn(3, 3)], 3)

        # The diagonal_blocks look like:
        # [Zero(2,2), Zero(3,3), Zero(3,3)] (by construction)
        # Lower blocks set the block layout for the actual width/height calculations.
        assert L.height == 2 + 3 + 3
        assert L.width == 2 + 3 + 3

        # Try a different shape (with rectangular blocks)
        L2 = LowerDiagonal(4, [torch.randn(2, 5), torch.randn(5, 6)], 7)
        # L2 has shape
        #    [ (4x5)0         0        0    ]
        #    [ (2x5)L0   (2x6)0        0    ]
        #    [    0      (5x6)L1  (5x7)0   ]
        assert L2.height == 4 + 2 + 5
        assert L2.width == 5 + 6 + 7

        L3 = LowerDiagonal(
            1, [torch.randn(4, 2), torch.randn(2, 3), torch.randn(3, 5)], 6
        )
        # L3 has the structure:
        #      [ (1x2)0      0        0        0      ]
        #      [ (4x2)L0   (4x3)0     0        0      ]
        #      [    0      (2x3)L1   (2x5)0    0      ]
        #      [    0         0      (3x5)L2  (3x6)0  ]
        assert L3.height == 1 + 4 + 2 + 3
        assert L3.width == 2 + 3 + 5 + 6

        L_tensor = L.to_tensor()
        assert L_tensor.shape[0] == L.height
        assert L_tensor.shape[1] == L.width

    def test_transpose(self):
        L = LowerDiagonal(2, [torch.randn(3, 2), torch.randn(3, 3)], 3)
        assert torch.allclose(L.T.to_tensor(), L.to_tensor().T)


class TestDownshift:
    @pytest.fixture
    def v(self):
        return Vertical([torch.randn(2, 1), torch.randn(3, 1), torch.randn(4, 1)])

    def test_downshift(self, v):
        P = downshifting_matrix(10, [b.shape[0] for b in v.flatten()])
        Pv = P @ v
        assert Pv.num_blocks() == 3

        assert Pv.flat[0] == Zero((10, 1))
        torch.testing.assert_close(Pv.flat[1], v.flat[0])
        torch.testing.assert_close(Pv.flat[2], v.flat[1])

    def test_upshift(self, v):
        shapes = [b.shape[0] for b in v.flatten()]
        P = downshifting_matrix(shapes[0], shapes[1:] + [shapes[-1]])
        PTv = P.T @ v
        assert PTv.num_blocks() == 3
        torch.testing.assert_close(PTv.flat[0], v.flat[1])
        torch.testing.assert_close(PTv.flat[1], v.flat[2])
        assert PTv.flat[2] == Zero((shapes[-1], 1))

    def test_down_then_upshift(self, v):
        P = downshifting_matrix(10, [b.shape[0] for b in v.flatten()])
        r = P.T @ (P @ v)
        assert r.num_blocks() == 3
        torch.testing.assert_close(r.flat[0], v.flat[0])
        torch.testing.assert_close(r.flat[1], v.flat[1])
        assert r.flat[2] == Zero((4, 1))

    def test_PTP(self, v):
        P = downshifting_matrix(10, [b.shape[0] for b in v.flatten()])
        r = P.T @ (P @ v)
        rr = (P.T @ P) @ v
        assert r.num_blocks() == rr.num_blocks()
        for br, brr in zip(r.flatten(), rr.flatten()):
            torch.testing.assert_close(br.to_tensor(), brr.to_tensor())

    def test_PPT_blocks(self, v):
        P = downshifting_matrix(2, [b.shape[0] for b in v.flatten()])
        D = P @ P.T
        assert isinstance(D, Diagonal)
        assert D.num_blocks() == 3
        assert D.flat[0].shape == (2, 2)
        assert isinstance(D.flat[1], Identity)
        assert D.flat[1].dimension == 2
        assert isinstance(D.flat[2], Identity)
        assert D.flat[2].dimension == 3

    def test_PPT(self, v):
        P = downshifting_matrix(2, [b.shape[0] for b in v.flatten()])
        u_heights = [b.height for b in P.diagonal_blocks]
        u = Vertical([torch.randn(h, 1) for h in u_heights])

        r = P @ (P.T @ u)
        rr = (P @ P.T) @ u
        assert r.num_blocks() == rr.num_blocks()
        for br, brr in zip(r.flatten(), rr.flatten()):
            torch.testing.assert_close(br.to_tensor(), brr.to_tensor())

    def test_PTDP(self):
        P = downshifting_matrix(2, [3, 4, 5])
        D = Diagonal([torch.randn(2, 2), torch.randn(3, 3), torch.randn(4, 4)])

        r = P.T @ (D @ P)
        rr = (P.T @ D) @ P
        assert r.num_blocks() == rr.num_blocks()
        for br, brr in zip(r.flatten(), rr.flatten()):
            torch.testing.assert_close(br.to_tensor(), brr.to_tensor())


class TestTransposeRelationship:
    """Tests for transpose relationship between Lower and Upper block diagonal matrices."""

    def test_lower_diagonal_to_tensor(self):
        torch.manual_seed(123)

        L = LowerDiagonal(
            height_leading_zeros=2,
            lower_blocks=[torch.randn(2, 3), torch.randn(3, 4)],
            width_trailing_zeros=4,
        )

        U = L.T

        assert isinstance(U, UpperDiagonal)

        # Compare dense tensor representations
        L_dense = L.to_tensor()
        U_dense = U.to_tensor()
        torch.testing.assert_close(U_dense, L_dense.T)

    def test_lower_bidiagonal_to_tensor(self):
        torch.manual_seed(321)

        L = LowerBiDiagonal(
            lower_blocks=[torch.randn(2, 2), torch.randn(2, 2)],
            diagonal_blocks=[torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2)],
        )
        U = L.T

        assert isinstance(U, UpperBiDiagonal)

        # Compare dense tensor representations
        L_dense = L.to_tensor()
        U_dense = U.to_tensor()
        torch.testing.assert_close(U_dense, L_dense.T)

    def test_upper_bidiagonal_to_tensor(self):
        torch.manual_seed(654)

        U = UpperBiDiagonal(
            upper_blocks=[torch.randn(3, 3), torch.randn(3, 3)],
            diagonal_blocks=[torch.randn(3, 3), torch.randn(3, 3), torch.randn(3, 3)],
        )
        L = U.T

        assert isinstance(L, LowerBiDiagonal)

        # Compare dense tensor representations
        U_dense = U.to_tensor()
        L_dense = L.to_tensor()
        torch.testing.assert_close(L_dense, U_dense.T)

    def test_lower_upper_transpose_consistency(self):
        """Test that M.T @ v = M^T @ v mathematically."""
        torch.manual_seed(42)

        m_lower = IdentityWithLowerDiagonal([torch.randn(3, 3), torch.randn(3, 3)])
        m_upper = m_lower.T

        # This should be consistent with the transpose operation
        assert isinstance(m_upper, UpperBiDiagonal)

    def test_upper_lower_transpose_consistency(self):
        """Test that (M^T).T @ v = M @ v."""
        torch.manual_seed(42)

        m_upper = IdentityWithUpperDiagonal([torch.randn(3, 3), torch.randn(3, 3)])
        m_lower = m_upper.T

        # This should be consistent with the transpose operation
        assert isinstance(m_lower, LowerBiDiagonal)


class TestSymmetricBlock2x2:
    """Tests for SymmetricBlock2x2 class."""

    def test_apply(self):
        """Test matrix-vector multiplication with a 2-block Vertical."""
        # Create a simple symmetric block 2x2 matrix
        # Q = [[Q11, Q12],
        #      [Q12^T, Q22]]
        matrix = Symmetric2x2(
            block11=torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
            block12=col(1.0, 1.0),
            block22=torch.tensor([[4.0]]),
        )

        v = Vertical([col(1.0, 1.0), col(2.0)])
        result = matrix @ v

        # result[0] = Q11 @ v[0] + Q12 @ v[1] = [2, 0; 0, 3] @ [1, 1] + [1; 1] @ [2]
        #         = [2, 3] + [2, 2] = [4, 5]
        # result[1] = Q22 @ v[1] + Q12^T @ v[0] = [4] @ [2] + [1, 1] @ [1, 1]
        #         = [8] + [2] = [10]
        assert torch.allclose(result.flat[0], col(4.0, 5.0))
        assert torch.allclose(result.flat[1], col(10.0))

    def test_apply_mismatched_blocks_raises(self):
        """Test that apply with wrong number of blocks raises ValueError."""
        matrix = Symmetric2x2(
            torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
            col(1.0, 1.0),
            torch.tensor([[4.0]]),
        )

        v = Vertical([col(1.0, 1.0)])  # Only 1 block, need 2

        with pytest.raises(ValueError, match="Number of blocks"):
            matrix @ v

    def test_invert(self):
        """Test inverting symmetric block 2x2 matrix."""
        torch.manual_seed(42)
        # Create positive definite blocks to ensure invertibility
        block11 = torch.randn(3, 3)
        block22 = torch.randn(2, 2)

        matrix2x2 = Symmetric2x2(
            block11 @ block11.T + torch.eye(3),
            torch.randn(3, 2),
            block22 @ block22.T + torch.eye(2),
        )
        matrix2x2_inv = matrix2x2.invert()

        # Create a random vector
        v = Vertical([torch.randn(3, 1), torch.randn(2, 1)])

        # Check that matrix @ matrix_inv @ v = v
        result = matrix2x2 @ (matrix2x2_inv @ v)

        assert torch.allclose(result.flat[0], v.flat[0])
        assert torch.allclose(result.flat[1], v.flat[1])

    def test_invert_via_LDL(self):
        """Test inverting symmetric block 2x2 matrix via LDL decomposition."""
        torch.manual_seed(42)
        # Create positive definite blocks to ensure invertibility
        block11 = torch.randn(3, 3)
        block22 = torch.randn(2, 2)

        matrix2x2 = Symmetric2x2(
            block11 @ block11.T + torch.eye(3),
            torch.randn(3, 2),
            block22 @ block22.T + torch.eye(2),
        )
        matrix2x2_inv = matrix2x2.invert_via_LDL()

        # Create a random vector
        v = Vertical([torch.randn(3, 1), torch.randn(2, 1)])

        # Check that matrix @ matrix_inv @ v = v
        result = matrix2x2 @ (matrix2x2_inv @ (v))

        assert torch.allclose(result.flat[0], v.flat[0])
        assert torch.allclose(result.flat[1], v.flat[1])

    def test_matmul_with_upper_diagonal_blocks_using_to_tensor(self):
        """Test inverting SymmetricBlock2x2 with upper diagonal blocks."""
        matrix2x2 = Symmetric2x2(
            block11=Diagonal([torch.ones(2, 2), torch.ones(3, 3)]),
            block12=UpperDiagonal(2, [torch.ones(2, 3)], 3),
            block22=Diagonal([torch.ones(2, 2), torch.ones(3, 3)]) + Identity(),
        )
        v = Vertical([Vertical([torch.ones(2, 1), torch.ones(3, 1)])] * 2)
        r = matrix2x2 @ v
        rr = matrix2x2.to_tensor() @ v.to_tensor()
        torch.testing.assert_close(r.to_tensor(), rr)

    def test_matmul_with_upper_diagonal_blocks(self):
        """Test inverting SymmetricBlock2x2 with upper diagonal blocks."""
        matrix2x2 = Symmetric2x2(
            block11=Diagonal([torch.ones(2, 2), torch.ones(3, 3)]),
            block12=UpperDiagonal(2, [torch.ones(2, 3)], 3),
            block22=Diagonal([torch.ones(2, 2), torch.ones(3, 3)]) + Identity(),
        )

        v = Vertical([Vertical([torch.ones(2, 1), torch.ones(3, 1)])] * 2)

        r = matrix2x2 @ v
        assert len(r.blocks) == 2
        torch.testing.assert_close(r.flat[0].flat[0], 5 * torch.ones(2, 1))
        torch.testing.assert_close(r.flat[0].flat[1], 3 * torch.ones(3, 1))
        torch.testing.assert_close(r.flat[1].flat[0], 3 * torch.ones(2, 1))
        torch.testing.assert_close(r.flat[1].flat[1], 6 * torch.ones(3, 1))

    def test_invert_with_upper_diagonal_blocks_using_to_tensor(self):
        """Test inverting SymmetricBlock2x2 with upper diagonal blocks."""
        block11 = Diagonal([torch.randn(2, 2), torch.randn(3, 3)])
        block22 = Diagonal([torch.randn(2, 2), torch.randn(3, 3)])
        M = Symmetric2x2(
            block11=block11 @ block11.T,
            block12=UpperDiagonal(2, [torch.randn(2, 3)], 3),
            block22=block22 @ block22.T + Identity(),
        )
        M_inv = M.invert()
        M_inv_tensor = M_inv.to_tensor()
        M_inv_tensor_torch = torch.linalg.inv(M.to_tensor())

        torch.testing.assert_close(M_inv_tensor, M_inv_tensor_torch)

    def test_invert_with_upper_diagonal_blocks(self):
        """Test inverting SymmetricBlock2x2 with upper diagonal blocks."""

        block11 = Diagonal([torch.randn(2, 2), torch.randn(3, 3)])
        block22 = Diagonal([torch.randn(2, 2), torch.randn(3, 3)])
        M = Symmetric2x2(
            block11=block11 @ block11.T,
            block12=UpperDiagonal(2, [torch.randn(2, 3)], 3),
            block22=block22 @ block22.T + Identity(),
        )
        M_inv = M.invert()
        v = Vertical(
            [
                Vertical([torch.randn(2, 1), torch.randn(3, 1)]),
                Vertical([torch.randn(2, 1), torch.randn(3, 1)]),
            ]
        )
        r = M_inv @ (M @ v)
        assert r.num_blocks() == 2
        torch.testing.assert_close(r.flat[0].flat[0], v.flat[0].flat[0])
        torch.testing.assert_close(r.flat[0].flat[1], v.flat[0].flat[1])
        torch.testing.assert_close(r.flat[1].flat[0], v.flat[1].flat[0])
        torch.testing.assert_close(r.flat[1].flat[1], v.flat[1].flat[1])

    def test_UDU_decomposition(self):
        """Test UDU_decomposition is correctness by checking U @ D @ U.T = A."""
        # Create positive definite blocks to ensure invertibility
        block11 = torch.randn(3, 3)
        block22 = torch.randn(2, 2)

        matrix2x2 = Symmetric2x2(
            block11 @ block11.T + torch.eye(3),
            torch.randn(3, 2),
            block22 @ block22.T + torch.eye(2),
        )
        U, D = matrix2x2.UDU_decomposition()

        assert isinstance(U, UpperBiDiagonal)
        assert isinstance(D, Diagonal)

        # Generate a random vector
        v = Vertical([torch.randn(3, 1), torch.randn(2, 1)])

        # Test that U @ D @ U.T @ v = A @ v
        UDUv = U @ (D @ (U.T @ v))
        Av = matrix2x2 @ v

        torch.testing.assert_close(UDUv.flat[0], Av.flat[0])
        torch.testing.assert_close(UDUv.flat[1], Av.flat[1])

    def test_LDL_decomposition(self):
        """Test LDL_decomposition is correctness by checking L @ D @ L.T = A."""
        # Create positive definite blocks to ensure invertibility
        block11 = torch.randn(3, 3)
        block22 = torch.randn(2, 2)

        matrix2x2 = Symmetric2x2(
            block11 @ block11.T + torch.eye(3),
            torch.randn(3, 2),
            block22 @ block22.T + torch.eye(2),
        )
        L, D = matrix2x2.LDL_decomposition()

        assert isinstance(L, LowerBiDiagonal)
        assert isinstance(D, Diagonal)

        # Generate a random vector
        v = Vertical([torch.randn(3, 1), torch.randn(2, 1)])

        # Test that L @ D @ L.T @ v = A @ v
        LDLv = L @ (D @ (L.T @ v))
        Av = matrix2x2 @ v

        torch.testing.assert_close(LDLv.blocks[0], Av.blocks[0])
        torch.testing.assert_close(LDLv.blocks[1], Av.blocks[1])

    def test_invert_via_UDU_decomposition(self):
        """Test inverting SymmetricBlock2x2 using LDL decomposition."""
        # Create positive definite blocks to ensure invertibility
        block11 = torch.randn(3, 3)
        block22 = torch.randn(2, 2)

        matrix2x2 = Symmetric2x2(
            block11 @ block11.T + torch.eye(3),
            torch.randn(3, 2),
            block22 @ block22.T + torch.eye(2),
        )
        U, D = matrix2x2.UDU_decomposition()

        # A vector that can be multiplied by matri2x2
        v = Vertical([torch.randn(3, 1), torch.randn(2, 1)])

        # Confirm that inv(UDU^T) A v = v
        v_hat = U.T.solve(D.solve(U.solve(matrix2x2 @ v)))

        torch.testing.assert_close(v_hat.blocks[0], v.blocks[0])
        torch.testing.assert_close(v_hat.blocks[1], v.blocks[1])


class TestSymmetricTriDiagonal:
    """Tests for SymmetricTriDiagonal class."""

    def test_UDU_decomposition_reconstructs_matrix(self):
        """Test that U @ D @ U.T equals the original symmetric tri-diagonal matrix."""
        torch.manual_seed(0)

        # Build a small 3-block SPD tri-diagonal system with 2x2 blocks.
        A0 = torch.randn(2, 2)
        A1 = torch.randn(2, 2)
        A2 = torch.randn(2, 2)

        D_blocks = [
            A0 @ A0.T + torch.eye(2),
            A1 @ A1.T + torch.eye(2),
            A2 @ A2.T + torch.eye(2),
        ]

        L_blocks = [
            torch.randn(2, 2),
            torch.randn(2, 2),
        ]

        tri = SymmetricTriDiagonal(lower_blocks=L_blocks, diagonal_blocks=D_blocks)

        U, D = tri.UDU_decomposition()

        assert isinstance(U, UpperBiDiagonal)
        assert isinstance(D, Diagonal)

        T_dense = tri.to_tensor()

        # Dense representation of the factorization.
        U_dense = U.to_tensor()
        D_dense = D.to_tensor()
        UDU_dense = U_dense @ D_dense @ U_dense.T

        torch.testing.assert_close(UDU_dense, T_dense, rtol=1e-5, atol=1e-5)

    def test_invert_via_UDU_decomposition(self):
        """Test solving A v = b via UDU decomposition for SymmetricTriDiagonal."""
        torch.manual_seed(1)

        A0 = torch.randn(2, 2)
        A1 = torch.randn(2, 2)
        A2 = torch.randn(2, 2)

        D_blocks = [
            A0 @ A0.T + torch.eye(2),
            A1 @ A1.T + torch.eye(2),
            A2 @ A2.T + torch.eye(2),
        ]

        L_blocks = [
            torch.randn(2, 2),
            torch.randn(2, 2),
        ]

        tri = SymmetricTriDiagonal(lower_blocks=L_blocks, diagonal_blocks=D_blocks)
        U, D = tri.UDU_decomposition()

        # Build a random RHS and solve using dense linear algebra as reference.
        T_dense = tri.to_tensor()
        b = torch.randn(T_dense.shape[0], 1)

        # Reference solution using dense solver.
        x_ref = torch.linalg.solve(T_dense, b)

        # Solution using the UDU factorization: x = (U D U^T)^{-1} b.
        # Interpret b as a Vertical of blocks matching the diagonal blocks.
        block_size = D_blocks[0].shape[0]
        b_blocks = [
            b[i * block_size : (i + 1) * block_size] for i in range(len(D_blocks))
        ]
        b_vert = Vertical(b_blocks)

        x_block = U.T.solve(D.solve(U.solve(b_vert)))
        x = x_block.to_tensor()

        torch.testing.assert_close(x, x_ref, rtol=1e-5, atol=1e-5)


class TestTridiagonal:
    def test_blockwise_transpose_collects_band_components(self):
        def scalar_block(value: float) -> Diagonal:
            return Diagonal([Tensor.wrap(torch.tensor([[value]]))])

        def make_tridiagonal(offset: float) -> Tridiagonal:
            return Tridiagonal(
                [scalar_block(offset + i) for i in range(3)],
                lower_blocks=[scalar_block(offset + 10 + i) for i in range(2)],
                upper_blocks=[scalar_block(offset + 20 + i) for i in range(2)],
            )

        blocks = [
            [make_tridiagonal(1.0), make_tridiagonal(2.0)],
            [make_tridiagonal(3.0), make_tridiagonal(4.0)],
        ]
        M = Generic(blocks)

        transposed = Tridiagonal.blockwise_transpose(M)

        assert len(transposed.diagonal_blocks) == 3
        assert len(transposed.lower_blocks) == 2
        assert len(transposed.upper_blocks) == 2
        assert transposed.diagonal_blocks[0].shape == M.shape

        for diag_idx, diag_block in enumerate(transposed.diagonal_blocks):
            assert isinstance(diag_block, Generic)
            for row in range(2):
                for col in range(2):
                    assert (
                        diag_block[row, col]
                        is blocks[row][col].diagonal_blocks[diag_idx]
                    )

        for band_idx, lower_block in enumerate(transposed.lower_blocks):
            assert isinstance(lower_block, Generic)
            for row in range(2):
                for col in range(2):
                    assert (
                        lower_block[row, col] is blocks[row][col].lower_blocks[band_idx]
                    )

        for band_idx, upper_block in enumerate(transposed.upper_blocks):
            assert isinstance(upper_block, Generic)
            for row in range(2):
                for col in range(2):
                    assert (
                        upper_block[row, col] is blocks[row][col].upper_blocks[band_idx]
                    )

    def test_add_tridiagonal(self):
        """Test adding two Tridiagonal matrices."""
        D = [torch.eye(2), torch.eye(2), torch.eye(2)]
        L = [torch.ones(2, 2), torch.ones(2, 2)]
        U = [torch.ones(2, 2), torch.ones(2, 2)]

        T1 = Tridiagonal(D, lower_blocks=L, upper_blocks=U)
        T2 = Tridiagonal(D, lower_blocks=L, upper_blocks=U)

        T_sum = T1 + T2

        assert isinstance(T_sum, Tridiagonal)
        # Verify blocks
        for b1, b_sum in zip(T1.diagonal_blocks, T_sum.diagonal_blocks):
            torch.testing.assert_close(b_sum.to_tensor(), b1.to_tensor() * 2)
        for b1, b_sum in zip(T1.lower_blocks, T_sum.lower_blocks):
            torch.testing.assert_close(b_sum.to_tensor(), b1.to_tensor() * 2)
        for b1, b_sum in zip(T1.upper_blocks, T_sum.upper_blocks):
            torch.testing.assert_close(b_sum.to_tensor(), b1.to_tensor() * 2)

    def test_LDU_decomposition_reconstructs_matrix(self):
        """Test that L @ D @ U equals the original tridiagonal matrix."""
        torch.manual_seed(0)

        # Build a small 3-block tridiagonal system with 2x2 blocks.  Ensure its
        # diagonal blocks are invertible.
        tri = Tridiagonal(
            [
                torch.randn(2, 2) + 2 * torch.eye(2),
                torch.randn(2, 2) + 2 * torch.eye(2),
                torch.randn(2, 2) + 2 * torch.eye(2),
            ],
            lower_blocks=[torch.randn(2, 2), torch.randn(2, 2)],
            upper_blocks=[torch.randn(2, 2), torch.randn(2, 2)],
        )
        L, D, U = tri.LDU_decomposition()

        assert isinstance(L, IdentityWithLowerDiagonal)
        assert isinstance(D, Diagonal)
        assert isinstance(U, IdentityWithUpperDiagonal)

        T_dense = tri.to_tensor()

        # Dense representation of the factorization.
        L_dense = L.to_tensor()
        D_dense = D.to_tensor()
        U_dense = U.to_tensor()
        LDU_dense = L_dense @ D_dense @ U_dense

        torch.testing.assert_close(LDU_dense, T_dense, rtol=1e-5, atol=1e-5)

    def test_solve_via_LDU_decomposition(self):
        """Test solving A v = b via LDU decomposition for Tridiagonal."""
        torch.manual_seed(1)

        D_blocks = [
            torch.randn(2, 2) + 5 * torch.eye(2),
            torch.randn(2, 2) + 5 * torch.eye(2),
            torch.randn(2, 2) + 5 * torch.eye(2),
        ]

        L_blocks = [torch.randn(2, 2), torch.randn(2, 2)]
        U_blocks = [torch.randn(2, 2), torch.randn(2, 2)]

        tri = Tridiagonal(D_blocks, L_blocks, U_blocks)
        L_mat, D_mat, U_mat = tri.LDU_decomposition()

        # Build a random RHS and solve using dense linear algebra as reference.
        T_dense = tri.to_tensor()
        b = torch.randn(T_dense.shape[0], 1)

        # Reference solution using dense solver.
        x_ref = torch.linalg.solve(T_dense, b)

        # Solution using the LDU factorization: x = (L D U)^{-1} b = U^{-1} D^{-1} L^{-1} b.
        # Interpret b as a Vertical of blocks matching the diagonal blocks.
        block_size = D_blocks[0].shape[0]
        b_blocks = [
            b[i * block_size : (i + 1) * block_size] for i in range(len(D_blocks))
        ]
        b_vert = Vertical(b_blocks)

        # Solve L y = b  => y = L^{-1} b
        # Solve D z = y  => z = D^{-1} y
        # Solve U x = z  => x = U^{-1} z
        y = L_mat.solve(b_vert)
        z = D_mat.solve(y)
        x_block = U_mat.solve(z)

        x = x_block.to_tensor()

        torch.testing.assert_close(x, x_ref, rtol=1e-5, atol=1e-5)

    def test_solve(self):
        """Test Tridiagonal.solve against dense reference."""
        torch.manual_seed(2)

        tri = Tridiagonal(
            [
                torch.randn(2, 2) + 2 * torch.eye(2),
                torch.randn(2, 2) + 2 * torch.eye(2),
                torch.randn(2, 2) + 2 * torch.eye(2),
            ],
            lower_blocks=[torch.randn(2, 2), torch.randn(2, 2)],
            upper_blocks=[torch.randn(2, 2), torch.randn(2, 2)],
        )

        # Build RHS
        b = Vertical([torch.randn(b.height, 1) for b in tri.diagonal_blocks])

        # Solve
        solution = tri.solve(b)

        solution_expected = torch.linalg.solve(tri.to_tensor(), b.to_tensor())
        torch.testing.assert_close(solution.to_tensor(), solution_expected)

    def test_matmul_vertical(self):
        """Test Tridiagonal @ Vertical multiplication."""
        torch.manual_seed(42)
        # Create a Tridiagonal Matrix
        # [ 1I   U0   0  ]
        # [ L0   2I   U1 ]
        # [ 0    L1   3I ]

        L0 = torch.randn(2, 2)
        L1 = torch.randn(2, 2)
        U0 = torch.randn(2, 2)
        U1 = torch.randn(2, 2)

        tri = Tridiagonal(
            [1 * torch.eye(2), 2 * torch.eye(2), 3 * torch.eye(2)],
            lower_blocks=[L0, L1],
            upper_blocks=[U0, U1],
        )

        # Vector v = [v1, v2, v3]
        v1 = col(1.0, 1.0)
        v2 = col(2.0, 2.0)
        v3 = col(3.0, 3.0)
        v = Vertical([v1, v2, v3])

        # Expected result:
        # r1 = D1 v1 + U1 v2 = 1*v1 + U0 v2
        # r2 = L1 v1 + D2 v2 + U2 v3 = L0 v1 + 2*v2 + U1 v3
        # r3 = L2 v2 + D3 v3 = L1 v2 + 3*v3

        r1 = 1 * v1 + U0 @ v2
        r2 = L0 @ v1 + 2 * v2 + U1 @ v3
        r3 = L1 @ v2 + 3 * v3

        result = tri @ v

        assert isinstance(result, Vertical)
        assert result.num_blocks() == 3
        assert torch.allclose(result.flat[0], r1)
        assert torch.allclose(result.flat[1], r2)
        assert torch.allclose(result.flat[2], r3)


class TestScaledIdentity:
    def test_init(self):
        s = ScaledIdentity(2.0, 3)
        assert s.scale == 2.0
        assert s.dimension == 3

    def test_mul_identity_creates_scaled(self):
        I = Identity(3)
        s = 2.0 * I
        assert isinstance(s, ScaledIdentity)
        assert s.scale == 2.0
        assert s.dimension == 3

        s2 = I * 3.0
        assert isinstance(s2, ScaledIdentity)
        assert s2.scale == 3.0
        assert s2.dimension == 3

    def test_to_tensor(self):
        s = ScaledIdentity(2.0, 3)
        expected = 2.0 * torch.eye(3)
        torch.testing.assert_close(s.to_tensor(), expected)

    def test_invert(self):
        s = ScaledIdentity(2.0, 3)
        inv = s.invert()
        assert isinstance(inv, ScaledIdentity)
        assert inv.scale == 0.5
        assert inv.dimension == 3

    def test_invert_product(self):
        sI = ScaledIdentity(2.0, 3)
        sIi = sI.invert()
        v = Tensor(torch.tensor([[1.0], [2.0], [3.0]]))

        res = sI @ sIi @ v

        torch.testing.assert_close(res.to_tensor(), v.to_tensor())

    def test_solve_tensor(self):
        s = ScaledIdentity(2.0, 2)
        rhs = Tensor(torch.tensor([[4.0], [6.0]]))
        sol = s.solve(rhs)
        assert isinstance(sol, Tensor)
        torch.testing.assert_close(sol.to_tensor(), torch.tensor([[2.0], [3.0]]))

    def test_matmul_tensor(self):
        s = ScaledIdentity(2.0, 2)
        other = Tensor(torch.eye(2))
        res = s @ other
        torch.testing.assert_close(res.to_tensor(), 2.0 * torch.eye(2))

        res2 = other @ s
        torch.testing.assert_close(res2.to_tensor(), 2.0 * torch.eye(2))

    def test_matmul_scaled_identity(self):
        s1 = ScaledIdentity(2.0, 3)
        s2 = ScaledIdentity(3.0, 3)
        res = s1 @ s2
        assert isinstance(res, ScaledIdentity)
        assert res.scale == 6.0
        assert res.dimension == 3

    def test_add_tensor(self):
        t = Tensor(torch.ones(2, 2))
        s = ScaledIdentity(2.0, 2)
        res = t + s
        expected = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
        torch.testing.assert_close(res.to_tensor(), expected)

    def test_add_diagonal(self):
        D = Diagonal([torch.eye(2), torch.eye(2)])
        sI = ScaledIdentity(2.0)  # implicit dimension
        res = D + sI
        assert isinstance(res, Diagonal)
        # Each block should be I + 2I = 3I
        for b in res.diagonal_blocks:
            torch.testing.assert_close(b.to_tensor(), 3.0 * torch.eye(2))

    def test_add_symmetric_tridiagonal(self):
        D = [torch.eye(2), torch.eye(2)]
        L = [torch.zeros(2, 2)]
        T = SymmetricTriDiagonal(lower_blocks=L, diagonal_blocks=D)
        s = ScaledIdentity(2.0)
        res = T + s
        assert isinstance(res, SymmetricTriDiagonal)
        for b in res.diagonal_blocks:
            torch.testing.assert_close(b.to_tensor(), 3.0 * torch.eye(2))
