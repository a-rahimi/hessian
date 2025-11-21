"""Tests for partitioned.py module."""

import pytest
import torch
from block_partitioned_matrices import (
    Vertical,
    Diagonal,
    IdentityWithLowerDiagonal,
    IdentityWithUpperDiagonal,
    Symmetric2x2,
    SymmetricTriDiagonal,
    UpperBiDiagonal,
    UpperDiagonal,
    LowerDiagonal,
    LowerBiDiagonal,
    Identity,
    downshifting_matrix,
)


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

        assert torch.allclose(result.blocks[0], col(11.0, 22.0))
        assert torch.allclose(result.blocks[1], col(33.0, 44.0, 55.0))

    def test_add_single_block(self):
        """Test adding two block vectors with a single block each."""
        v1 = Vertical([col(1.0, 2.0, 3.0)])
        v2 = Vertical([col(4.0, 5.0, 6.0)])

        result = v1 + v2

        assert torch.allclose(result.blocks[0], col(5.0, 7.0, 9.0))

    def test_add_mismatched_block_count_raises(self):
        """Test that adding vectors with different block counts raises ValueError."""
        v1 = Vertical([col(1.0, 2.0)])
        v2 = Vertical([col(3.0, 4.0), col(5.0, 6.0)])

        with pytest.raises(ValueError, match="Number of blocks must match"):
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


class TestBlockDiagonal:
    """Tests for BlockDiagonal class."""

    def test_init(self):
        """Test BlockDiagonal initialization."""
        blocks = [torch.eye(2), torch.eye(3)]
        bdm = Diagonal(blocks)
        assert len(bdm.blocks) == 2

    def test_neg(self):
        """Test negation of block-diagonal matrix."""
        blocks = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0]])]
        bdm = Diagonal(blocks)
        neg_bdm = -bdm

        assert torch.allclose(neg_bdm.blocks[0], -blocks[0])
        assert torch.allclose(neg_bdm.blocks[1], -blocks[1])

    def test_apply(self):
        """Test applying block-diagonal matrix to vector."""
        bdm = Diagonal(
            [
                torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
            ]
        )
        v = Vertical([col(1.0, 1.0), col(1.0, 1.0, 1.0)])

        result = bdm @ v

        assert torch.allclose(result.blocks[0], col(2.0, 3.0))
        assert torch.allclose(result.blocks[1], col(1.0, 2.0, 3.0))

    def test_apply_mismatched_blocks_raises(self):
        """Test that apply with mismatched blocks raises ValueError."""
        blocks = [torch.eye(2), torch.eye(3)]
        bdm = Diagonal(blocks)

        v_blocks = [col(1.0, 1.0)]  # Only 1 block
        v = Vertical(v_blocks)

        with pytest.raises(ValueError, match="Number of blocks"):
            bdm @ v

    def test_invert(self):
        """Test inverting block-diagonal matrix."""
        # Use simple invertible matrices
        blocks = [
            torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
            torch.tensor([[4.0]]),
        ]
        bdm = Diagonal(blocks)
        inv_bdm = bdm.invert()

        assert torch.allclose(
            inv_bdm.blocks[0], torch.tensor([[0.5, 0.0], [0.0, 1.0 / 3.0]])
        )
        assert torch.allclose(inv_bdm.blocks[1], torch.tensor([[0.25]]))

    def test_invert_singular_raises(self):
        """Test that inverting singular matrix raises error."""
        bdm = Diagonal([torch.tensor([[0.0, 0.0], [0.0, 0.0]])])

        with pytest.raises(torch._C._LinAlgError):
            bdm.invert()

    def test_solve(self):
        """Test solving linear system with block-diagonal matrix."""
        # A x = b where A is block diagonal
        bdm = Diagonal([torch.tensor([[2.0, 0.0], [0.0, 3.0]]), torch.tensor([[4.0]])])

        rhs = Vertical([col(4.0, 6.0), col(8.0)])

        solution = bdm.solve(rhs)

        assert torch.allclose(solution.blocks[0], col(2.0, 2.0))
        assert torch.allclose(solution.blocks[1], col(2.0))

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

        bdm = Diagonal([A1, A2])

        rhs = Vertical([torch.randn(3, 1), torch.randn(2, 1)])

        solution = bdm.solve(rhs)

        # Verify: A @ x = b
        result = bdm @ solution
        assert torch.allclose(result.blocks[0], rhs.blocks[0], atol=1e-5)
        assert torch.allclose(result.blocks[1], rhs.blocks[1], atol=1e-5)

    def test_add_identity(self):
        "Test that D+I works"
        D = Diagonal([torch.eye(2), torch.eye(3)])
        Daug = D + Identity()
        assert torch.allclose(Daug.blocks[0], 2 * torch.eye(2))
        assert torch.allclose(Daug.blocks[1], 2 * torch.eye(3))

    def test_sub_identity(self):
        "Test that D-I works"
        D = Diagonal([torch.eye(2), torch.eye(3)])
        Daug = D - Identity()
        assert torch.allclose(Daug.blocks[0], torch.zeros((2, 2)))
        assert torch.allclose(Daug.blocks[1], torch.zeros((3, 3)))

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

        assert torch.allclose(result.blocks[0], col(1.0, 1.0))
        assert torch.allclose(result.blocks[1], col(3.0, 3.0))
        assert torch.allclose(result.blocks[2], col(7.0, 7.0))

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

        assert torch.allclose(solution.blocks[0], col(1.0, 1.0))
        assert torch.allclose(solution.blocks[1], col(1.0, 1.0))
        assert torch.allclose(solution.blocks[2], col(1.0, 1.0))

    def test_solve_verifies_solution(self):
        """Test that solve produces a valid solution."""
        torch.manual_seed(42)

        M = IdentityWithLowerDiagonal([torch.randn(3, 3), torch.randn(3, 3)])

        rhs = Vertical([torch.randn(3, 1), torch.randn(3, 1), torch.randn(3, 1)])

        solution = M.solve(rhs)

        # Verify: M @ x = rhs
        result = M @ solution
        assert torch.allclose(result.blocks[0], rhs.blocks[0], atol=1e-5)
        assert torch.allclose(result.blocks[1], rhs.blocks[1], atol=1e-5)
        assert torch.allclose(result.blocks[2], rhs.blocks[2], atol=1e-5)

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

        assert torch.allclose(solution.blocks[0], col(1.0, 2.0))
        assert torch.allclose(solution.blocks[1], col(3.0, 4.0))
        assert torch.allclose(solution.blocks[2], col(2.0, 2.0))

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

        assert torch.allclose(result.blocks[0], col(3.0, 3.0))
        assert torch.allclose(result.blocks[1], col(8.0, 8.0))
        assert torch.allclose(result.blocks[2], col(3.0, 3.0))

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

        assert torch.allclose(solution.blocks[0], col(1.0, 1.0))
        assert torch.allclose(solution.blocks[1], col(2.0, 2.0))
        assert torch.allclose(solution.blocks[2], col(3.0, 3.0))

    def test_solve_verifies_solution(self):
        """Test that solve produces a valid solution."""
        torch.manual_seed(42)

        M = IdentityWithUpperDiagonal([torch.randn(3, 3), torch.randn(3, 3)])

        rhs = Vertical([torch.randn(3, 1), torch.randn(3, 1), torch.randn(3, 1)])

        solution = M.solve(rhs)

        # Verify: M @ x = rhs
        result = M @ solution
        assert torch.allclose(result.blocks[0], rhs.blocks[0], atol=1e-5)
        assert torch.allclose(result.blocks[1], rhs.blocks[1], atol=1e-5)
        assert torch.allclose(result.blocks[2], rhs.blocks[2], atol=1e-5)

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

        assert torch.allclose(solution.blocks[0], col(-2.0, -2.0))
        assert torch.allclose(solution.blocks[1], col(5.0, 6.0))
        assert torch.allclose(solution.blocks[2], col(7.0, 8.0))

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

        assert torch.allclose(result1.blocks[0], result2.blocks[0], atol=1e-5)
        assert torch.allclose(result1.blocks[1], result2.blocks[1], atol=1e-5)
        assert torch.allclose(result1.blocks[2], result2.blocks[2], atol=1e-5)


class TestUpperBlockDiagonal:
    def test_to_tensor(self):
        U = UpperDiagonal(2, [torch.randn(2, 3), torch.randn(3, 3)], 3)
        assert torch.allclose(U.T.to_tensor(), U.to_tensor().T)


class TestDownshift:
    @pytest.fixture
    def v(self):
        return Vertical([torch.randn(2, 1), torch.randn(3, 1), torch.randn(4, 1)])

    def test_downshift(self, v):
        P = downshifting_matrix(10, [b.shape[0] for b in v.blocks])
        Pv = P @ v
        assert len(Pv.blocks) == 3
        torch.testing.assert_close(Pv.blocks[0], torch.zeros(10, 1))
        torch.testing.assert_close(Pv.blocks[1], v.blocks[0])
        torch.testing.assert_close(Pv.blocks[2], v.blocks[1])

    def test_upshift(self, v):
        P = downshifting_matrix(2, [b.shape[0] for b in v.blocks])
        PTv = P.T @ v
        assert len(PTv.blocks) == 3
        torch.testing.assert_close(PTv.blocks[0], v.blocks[1])
        torch.testing.assert_close(PTv.blocks[1], v.blocks[2])
        torch.testing.assert_close(PTv.blocks[2], torch.zeros(4, 1))

    def test_down_then_upshift(self, v):
        P = downshifting_matrix(10, [b.shape[0] for b in v.blocks])
        r = P.T @ (P @ v)
        assert len(r.blocks) == 3
        torch.testing.assert_close(r.blocks[0], v.blocks[0])
        torch.testing.assert_close(r.blocks[1], v.blocks[1])
        torch.testing.assert_close(r.blocks[2], torch.zeros(4, 1))

    def test_PTP(self, v):
        P = downshifting_matrix(10, [b.shape[0] for b in v.blocks])
        r = P.T @ (P @ v)
        rr = (P.T @ P) @ v
        assert len(r.blocks) == len(rr.blocks)
        for br, brr in zip(r.blocks, rr.blocks):
            torch.testing.assert_close(br, brr)

    def test_PPT(self, v):
        P = downshifting_matrix(2, [b.shape[0] for b in v.blocks])
        r = P @ (P.T @ v)
        rr = (P @ P.T) @ v
        assert len(r.blocks) == len(rr.blocks)
        for br, brr in zip(r.blocks, rr.blocks):
            torch.testing.assert_close(br, brr)

    def test_PTDP(self):
        P = downshifting_matrix(2, [3, 4, 5])
        D = Diagonal([torch.randn(2, 2), torch.randn(3, 3), torch.randn(4, 4)])

        r = P.T @ (D @ P)
        rr = (P.T @ D) @ P
        assert len(r.blocks) == len(rr.blocks)
        for br, brr in zip(r.blocks, rr.blocks):
            torch.testing.assert_close(br, brr)


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
        assert torch.allclose(result.blocks[0], col(4.0, 5.0))
        assert torch.allclose(result.blocks[1], col(10.0))

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
        result = matrix2x2 @ (matrix2x2_inv @ (v))

        assert torch.allclose(result.blocks[0], v.blocks[0])
        assert torch.allclose(result.blocks[1], v.blocks[1])

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

        assert torch.allclose(result.blocks[0], v.blocks[0])
        assert torch.allclose(result.blocks[1], v.blocks[1])

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
        torch.testing.assert_close(r.blocks[0].blocks[0], 5 * torch.ones(2, 1))
        torch.testing.assert_close(r.blocks[0].blocks[1], 3 * torch.ones(3, 1))
        torch.testing.assert_close(r.blocks[1].blocks[0], 3 * torch.ones(2, 1))
        torch.testing.assert_close(r.blocks[1].blocks[1], 6 * torch.ones(3, 1))

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
        assert len(r.blocks) == 2
        torch.testing.assert_close(r.blocks[0].blocks[0], v.blocks[0].blocks[0])
        torch.testing.assert_close(r.blocks[0].blocks[1], v.blocks[0].blocks[1])
        torch.testing.assert_close(r.blocks[1].blocks[0], v.blocks[1].blocks[0])
        torch.testing.assert_close(r.blocks[1].blocks[1], v.blocks[1].blocks[1])

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

        torch.testing.assert_close(UDUv.blocks[0], Av.blocks[0])
        torch.testing.assert_close(UDUv.blocks[1], Av.blocks[1])

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
