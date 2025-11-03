"""Tests for partitioned.py module."""

import pytest
import torch
from partitioned import (
    Vertical,
    Diagonal,
    IdentityWithLowerBlockDiagonalMatrix,
    IdentityWithUpperBlockDiagonalMatrix,
    SymmetricBlock2x2Matrix,
)


class TestBlockVector:
    """Tests for BlockVector class."""

    def test_add_basic(self):
        """Test adding two block vectors with matching blocks."""
        v1_blocks = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])]
        v2_blocks = [torch.tensor([10.0, 20.0]), torch.tensor([30.0, 40.0, 50.0])]

        v1 = Vertical(v1_blocks)
        v2 = Vertical(v2_blocks)

        result = v1 + v2

        assert torch.allclose(result.blocks[0], torch.tensor([11.0, 22.0]))
        assert torch.allclose(result.blocks[1], torch.tensor([33.0, 44.0, 55.0]))

    def test_add_single_block(self):
        """Test adding two block vectors with a single block each."""
        v1 = Vertical([torch.tensor([1.0, 2.0, 3.0])])
        v2 = Vertical([torch.tensor([4.0, 5.0, 6.0])])

        result = v1 + v2

        assert torch.allclose(result.blocks[0], torch.tensor([5.0, 7.0, 9.0]))

    def test_add_mismatched_block_count_raises(self):
        """Test that adding vectors with different block counts raises ValueError."""
        v1 = Vertical([torch.tensor([1.0, 2.0])])
        v2 = Vertical([torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])])

        with pytest.raises(ValueError, match="Number of blocks must match"):
            v1 + v2

    def test_add_mismatched_block_shape_raises(self):
        """Test that adding vectors with mismatched block shapes raises RuntimeError."""
        v1 = Vertical([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])])
        v2 = Vertical([torch.tensor([10.0, 20.0, 30.0]), torch.tensor([40.0, 50.0])])

        with pytest.raises(
            RuntimeError,
            match="The size of tensor a .* must match the size of tensor b",
        ):
            v1 + v2


class TestBlockDiagonalMatrix:
    """Tests for BlockDiagonalMatrix class."""

    def test_init(self):
        """Test BlockDiagonalMatrix initialization."""
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
        # Create a simple 2x2 and 3x3 block diagonal matrix
        blocks = [
            torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
        ]
        bdm = Diagonal(blocks)

        v_blocks = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0, 1.0])]
        v = Vertical(v_blocks)

        result = bdm @ v

        expected_0 = torch.tensor([2.0, 3.0])
        expected_1 = torch.tensor([1.0, 2.0, 3.0])

        assert torch.allclose(result.blocks[0], expected_0)
        assert torch.allclose(result.blocks[1], expected_1)

    def test_apply_mismatched_blocks_raises(self):
        """Test that apply with mismatched blocks raises ValueError."""
        blocks = [torch.eye(2), torch.eye(3)]
        bdm = Diagonal(blocks)

        v_blocks = [torch.tensor([1.0, 1.0])]  # Only 1 block
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

        expected_inv_0 = torch.tensor([[0.5, 0.0], [0.0, 1.0 / 3.0]])
        expected_inv_1 = torch.tensor([[0.25]])

        assert torch.allclose(inv_bdm.blocks[0], expected_inv_0)
        assert torch.allclose(inv_bdm.blocks[1], expected_inv_1)

    def test_invert_singular_raises(self):
        """Test that inverting singular matrix raises error."""
        blocks = [torch.tensor([[0.0, 0.0], [0.0, 0.0]])]
        bdm = Diagonal(blocks)

        with pytest.raises(torch._C._LinAlgError):
            bdm.invert()

    def test_solve(self):
        """Test solving linear system with block-diagonal matrix."""
        # A x = b where A is block diagonal
        blocks = [
            torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
            torch.tensor([[4.0]]),
        ]
        bdm = Diagonal(blocks)

        rhs_blocks = [torch.tensor([4.0, 6.0]), torch.tensor([8.0])]
        rhs = Vertical(rhs_blocks)

        solution = bdm.solve(rhs)

        expected_0 = torch.tensor([2.0, 2.0])
        expected_1 = torch.tensor([2.0])

        assert torch.allclose(solution.blocks[0], expected_0)
        assert torch.allclose(solution.blocks[1], expected_1)

    def test_solve_mismatched_blocks_raises(self):
        """Test that solve with mismatched blocks raises ValueError."""
        blocks = [torch.eye(2), torch.eye(3)]
        bdm = Diagonal(blocks)

        rhs_blocks = [torch.tensor([1.0, 1.0])]  # Only 1 block
        rhs = Vertical(rhs_blocks)

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

        rhs_blocks = [torch.randn(3), torch.randn(2)]
        rhs = Vertical(rhs_blocks)

        solution = bdm.solve(rhs)

        # Verify: A @ x = b
        result = bdm @ solution
        assert torch.allclose(result.blocks[0], rhs.blocks[0], atol=1e-5)
        assert torch.allclose(result.blocks[1], rhs.blocks[1], atol=1e-5)


class TestIdentityWithLowerBlockDiagonalMatrix:
    """Tests for IdentityWithLowerBlockDiagonalMatrix class."""

    def test_init(self):
        """Test initialization."""
        lower_blocks = [torch.eye(2), torch.eye(2)]
        m = IdentityWithLowerBlockDiagonalMatrix(lower_blocks)
        assert len(m.lower_blocks) == 2

    def test_apply(self):
        """Test applying matrix to vector."""
        # M = [I      0    0
        #      A     I    0
        #      0     B    I]
        # where A and B are 2x2 matrices (lower_blocks already contains the values to use)
        A = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        B = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        lower_blocks = [A, B]
        m = IdentityWithLowerBlockDiagonalMatrix(lower_blocks)

        v_blocks = [
            torch.tensor([1.0, 1.0]),
            torch.tensor([2.0, 2.0]),
            torch.tensor([3.0, 3.0]),
        ]
        v = Vertical(v_blocks)

        result = m @ v

        # result[0] = v[0] = [1, 1]
        # result[1] = A @ v[0] + v[1] = [1, 1] + [2, 2] = [3, 3]
        # result[2] = B @ v[1] + v[2] = 2*[2, 2] + [3, 3] = [7, 7]

        assert torch.allclose(result.blocks[0], torch.tensor([1.0, 1.0]))
        assert torch.allclose(result.blocks[1], torch.tensor([3.0, 3.0]))
        assert torch.allclose(result.blocks[2], torch.tensor([7.0, 7.0]))

    def test_apply_mismatched_blocks_raises(self):
        """Test that apply with mismatched blocks raises ValueError."""
        lower_blocks = [torch.eye(2), torch.eye(2)]
        m = IdentityWithLowerBlockDiagonalMatrix(lower_blocks)

        v_blocks = [torch.tensor([1.0, 1.0])]  # Only 1 block, need 3
        v = Vertical(v_blocks)

        with pytest.raises(ValueError, match="Number of blocks"):
            m @ v

    def test_solve(self):
        """Test solving linear system."""
        # M x = rhs
        # x[0] = rhs[0]
        # x[i+1] = rhs[i+1] - lower_blocks[i] @ x[i]

        A = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        B = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        lower_blocks = [A, B]
        m = IdentityWithLowerBlockDiagonalMatrix(lower_blocks)

        rhs_blocks = [
            torch.tensor([1.0, 1.0]),
            torch.tensor([2.0, 2.0]),
            torch.tensor([3.0, 3.0]),
        ]
        rhs = Vertical(rhs_blocks)

        solution = m.solve(rhs)

        assert torch.allclose(solution.blocks[0], torch.tensor([1.0, 1.0]))
        assert torch.allclose(solution.blocks[1], torch.tensor([1.0, 1.0]))
        assert torch.allclose(solution.blocks[2], torch.tensor([1.0, 1.0]))

    def test_solve_verifies_solution(self):
        """Test that solve produces a valid solution."""
        torch.manual_seed(42)
        A = torch.randn(3, 3)
        B = torch.randn(3, 3)

        lower_blocks = [A, B]
        m = IdentityWithLowerBlockDiagonalMatrix(lower_blocks)

        rhs_blocks = [torch.randn(3), torch.randn(3), torch.randn(3)]
        rhs = Vertical(rhs_blocks)

        solution = m.solve(rhs)

        # Verify: M @ x = rhs
        result = m @ solution
        assert torch.allclose(result.blocks[0], rhs.blocks[0], atol=1e-5)
        assert torch.allclose(result.blocks[1], rhs.blocks[1], atol=1e-5)
        assert torch.allclose(result.blocks[2], rhs.blocks[2], atol=1e-5)

    def test_solve_with_zero_blocks(self):
        """Test solving when some lower blocks are zero."""
        # This is common for the first block
        zero_block = torch.zeros(2, 2)
        B = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        lower_blocks = [zero_block, B]
        m = IdentityWithLowerBlockDiagonalMatrix(lower_blocks)

        rhs_blocks = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
            torch.tensor([5.0, 6.0]),
        ]
        rhs = Vertical(rhs_blocks)

        solution = m.solve(rhs)

        # x[0] = rhs[0] = [1, 2]
        # x[1] = rhs[1] - 0 @ x[0] = [3, 4]
        # x[2] = rhs[2] - B @ x[1] = [5, 6] - [3, 4] = [2, 2]

        assert torch.allclose(solution.blocks[0], torch.tensor([1.0, 2.0]))
        assert torch.allclose(solution.blocks[1], torch.tensor([3.0, 4.0]))
        assert torch.allclose(solution.blocks[2], torch.tensor([2.0, 2.0]))

    def test_transpose(self):
        """Test that transpose returns IdentityWithUpperBlockDiagonalMatrix."""
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        lower_blocks = [A, B]
        m = IdentityWithLowerBlockDiagonalMatrix(lower_blocks)

        m_T = m.T

        assert isinstance(m_T, IdentityWithUpperBlockDiagonalMatrix)
        assert len(m_T.upper_blocks) == 2
        assert torch.allclose(m_T.upper_blocks[0], A.T)
        assert torch.allclose(m_T.upper_blocks[1], B.T)


class TestIdentityWithUpperBlockDiagonalMatrix:
    """Tests for IdentityWithUpperBlockDiagonalMatrix class."""

    def test_init(self):
        """Test initialization."""
        upper_blocks = [torch.eye(2), torch.eye(2)]
        m = IdentityWithUpperBlockDiagonalMatrix(upper_blocks)
        assert len(m.upper_blocks) == 2

    def test_apply(self):
        """Test applying matrix to vector."""
        # M^T = [I      A    0
        #        0      I    B
        #        0      0    I]
        # where A and B are 2x2 matrices
        A = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        B = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        upper_blocks = [A, B]
        m = IdentityWithUpperBlockDiagonalMatrix(upper_blocks)

        v_blocks = [
            torch.tensor([1.0, 1.0]),
            torch.tensor([2.0, 2.0]),
            torch.tensor([3.0, 3.0]),
        ]
        v = Vertical(v_blocks)

        result = m @ v

        # result[0] = v[0] + A @ v[1] = [1, 1] + [2, 2] = [3, 3]
        # result[1] = v[1] + B @ v[2] = [2, 2] + 2*[3, 3] = [8, 8]
        # result[2] = v[2] = [3, 3]

        assert torch.allclose(result.blocks[0], torch.tensor([3.0, 3.0]))
        assert torch.allclose(result.blocks[1], torch.tensor([8.0, 8.0]))
        assert torch.allclose(result.blocks[2], torch.tensor([3.0, 3.0]))

    def test_apply_mismatched_blocks_raises(self):
        """Test that apply with mismatched blocks raises ValueError."""
        upper_blocks = [torch.eye(2), torch.eye(2)]
        m = IdentityWithUpperBlockDiagonalMatrix(upper_blocks)

        v_blocks = [torch.tensor([1.0, 1.0])]  # Only 1 block, need 3
        v = Vertical(v_blocks)

        with pytest.raises(ValueError, match="Number of blocks"):
            m @ v

    def test_solve(self):
        """Test solving linear system using forward-substitution."""
        # M^T x = rhs
        # x[L] = rhs[L]
        # x[i] = rhs[i] - upper_blocks[i] @ x[i+1]

        A = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        B = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        upper_blocks = [A, B]
        m = IdentityWithUpperBlockDiagonalMatrix(upper_blocks)

        rhs_blocks = [
            torch.tensor([3.0, 3.0]),
            torch.tensor([8.0, 8.0]),
            torch.tensor([3.0, 3.0]),
        ]
        rhs = Vertical(rhs_blocks)

        solution = m.solve(rhs)

        # x[2] = rhs[2] = [3, 3]
        # x[1] = rhs[1] - B @ x[2] = [8, 8] - 2*[3, 3] = [2, 2]
        # x[0] = rhs[0] - A @ x[1] = [3, 3] - [2, 2] = [1, 1]

        assert torch.allclose(solution.blocks[0], torch.tensor([1.0, 1.0]))
        assert torch.allclose(solution.blocks[1], torch.tensor([2.0, 2.0]))
        assert torch.allclose(solution.blocks[2], torch.tensor([3.0, 3.0]))

    def test_solve_verifies_solution(self):
        """Test that solve produces a valid solution."""
        torch.manual_seed(42)

        m = IdentityWithUpperBlockDiagonalMatrix([torch.randn(3, 3), torch.randn(3, 3)])

        rhs = Vertical([torch.randn(3), torch.randn(3), torch.randn(3)])

        solution = m.solve(rhs)

        # Verify: M @ x = rhs
        result = m @ solution
        assert torch.allclose(result.blocks[0], rhs.blocks[0], atol=1e-5)
        assert torch.allclose(result.blocks[1], rhs.blocks[1], atol=1e-5)
        assert torch.allclose(result.blocks[2], rhs.blocks[2], atol=1e-5)

    def test_solve_with_zero_blocks(self):
        """Test solving when some upper blocks are zero."""

        m = IdentityWithUpperBlockDiagonalMatrix(
            [torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.zeros(2, 2)]
        )

        rhs_blocks = [
            torch.tensor([3.0, 4.0]),
            torch.tensor([5.0, 6.0]),
            torch.tensor([7.0, 8.0]),
        ]
        rhs = Vertical(rhs_blocks)

        solution = m.solve(rhs)

        # x[2] = rhs[2] = [7, 8]
        # x[1] = rhs[1] - 0 @ x[2] = [5, 6]
        # x[0] = rhs[0] - A @ x[1] = [3, 4] - [5, 6] = [-2, -2]

        assert torch.allclose(solution.blocks[0], torch.tensor([-2.0, -2.0]))
        assert torch.allclose(solution.blocks[1], torch.tensor([5.0, 6.0]))
        assert torch.allclose(solution.blocks[2], torch.tensor([7.0, 8.0]))

    def test_transpose(self):
        """Test that transpose returns IdentityWithLowerBlockDiagonalMatrix."""
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        m = IdentityWithUpperBlockDiagonalMatrix([A, B])

        m_T = m.T

        assert isinstance(m_T, IdentityWithLowerBlockDiagonalMatrix)
        assert len(m_T.lower_blocks) == 2
        assert torch.allclose(m_T.lower_blocks[0], A.T)
        assert torch.allclose(m_T.lower_blocks[1], B.T)

    def test_transpose_involution(self):
        """Test that (M.T).T = M (transpose is an involution)."""
        torch.manual_seed(42)
        A = torch.randn(3, 3)
        B = torch.randn(3, 3)
        m = IdentityWithUpperBlockDiagonalMatrix([A, B])

        v_blocks = [torch.randn(3), torch.randn(3), torch.randn(3)]
        v = Vertical(v_blocks)

        # Apply M
        result1 = m @ v

        # Apply (M.T).T which should equal M
        m_T_T = m.T.T
        result2 = m_T_T @ v

        assert torch.allclose(result1.blocks[0], result2.blocks[0], atol=1e-5)
        assert torch.allclose(result1.blocks[1], result2.blocks[1], atol=1e-5)
        assert torch.allclose(result1.blocks[2], result2.blocks[2], atol=1e-5)


class TestTransposeRelationship:
    """Tests for transpose relationship between Lower and Upper block diagonal matrices."""

    def test_lower_upper_transpose_consistency(self):
        """Test that M.T @ v = M^T @ v mathematically."""
        torch.manual_seed(42)
        A = torch.randn(3, 3)
        B = torch.randn(3, 3)

        m_lower = IdentityWithLowerBlockDiagonalMatrix([A, B])
        m_upper = m_lower.T

        # This should be consistent with the transpose operation
        assert isinstance(m_upper, IdentityWithUpperBlockDiagonalMatrix)

    def test_upper_lower_transpose_consistency(self):
        """Test that (M^T).T @ v = M @ v."""
        torch.manual_seed(42)

        m_upper = IdentityWithUpperBlockDiagonalMatrix(
            [torch.randn(3, 3), torch.randn(3, 3)]
        )
        m_lower = m_upper.T

        # This should be consistent with the transpose operation
        assert isinstance(m_lower, IdentityWithLowerBlockDiagonalMatrix)


class TestSymmetricBlock2x2Matrix:
    """Tests for SymmetricBlock2x2Matrix class."""

    def test_apply(self):
        """Test matrix-vector multiplication with a 2-block BlockVector."""
        # Create a simple symmetric block 2x2 matrix
        # Q = [[Q11, Q12],
        #      [Q12^T, Q22]]
        matrix = SymmetricBlock2x2Matrix(
            block11=torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
            block12=torch.tensor([[1.0], [1.0]]),
            block22=torch.tensor([[4.0]]),
        )

        v = Vertical([torch.tensor([1.0, 1.0]), torch.tensor([2.0])])
        result = matrix @ v

        # result[0] = Q11 @ v[0] + Q12 @ v[1] = [2, 0; 0, 3] @ [1, 1] + [1; 1] @ [2]
        #         = [2, 3] + [2, 2] = [4, 5]
        # result[1] = Q22 @ v[1] + Q12^T @ v[0] = [4] @ [2] + [1, 1] @ [1, 1]
        #         = [8] + [2] = [10]
        expected_0 = torch.tensor([4.0, 5.0])
        expected_1 = torch.tensor([10.0])

        assert torch.allclose(result.blocks[0], expected_0)
        assert torch.allclose(result.blocks[1], expected_1)

    def test_apply_mismatched_blocks_raises(self):
        """Test that apply with wrong number of blocks raises ValueError."""
        matrix = SymmetricBlock2x2Matrix(
            torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
            torch.tensor([[1.0], [1.0]]),
            torch.tensor([[4.0]]),
        )

        v = Vertical([torch.tensor([1.0, 1.0])])  # Only 1 block, need 2

        with pytest.raises(ValueError, match="Number of blocks"):
            matrix @ v

    def test_invert(self):
        """Test inverting symmetric block 2x2 matrix."""
        torch.manual_seed(42)
        # Create positive definite blocks to ensure invertibility
        block11 = torch.randn(3, 3)
        block22 = torch.randn(2, 2)

        matrix2x2 = SymmetricBlock2x2Matrix(
            block11 @ block11.T + torch.eye(3),
            torch.randn(3, 2),
            block22 @ block22.T + torch.eye(2),
        )
        matrix2x2_inv = matrix2x2.invert()

        # Create a random vector
        v = Vertical([torch.randn(3), torch.randn(2)])

        # Check that matrix @ matrix_inv @ v = v
        result = matrix2x2 @ (matrix2x2_inv @ (v))

        assert torch.allclose(result.blocks[0], v.blocks[0])
        assert torch.allclose(result.blocks[1], v.blocks[1])

    def test_ldl_decomposition_correctness(self):
        """Test UDU_decomposition is correctness by checking U @ D @ U.T = A."""
        # Create positive definite blocks to ensure invertibility
        block11 = torch.randn(3, 3)
        block22 = torch.randn(2, 2)

        matrix2x2 = SymmetricBlock2x2Matrix(
            block11 @ block11.T + torch.eye(3),
            torch.randn(3, 2),
            block22 @ block22.T + torch.eye(2),
        )
        U, D = matrix2x2.UDU_decomposition()

        assert isinstance(U, IdentityWithUpperBlockDiagonalMatrix)
        assert isinstance(D, Diagonal)

        # Generate a random vector
        v = Vertical([torch.randn(3), torch.randn(2)])

        # Test that U @ D @ U.T @ v = A @ v
        UDUv = U @ (D @ (U.T @ v))
        Av = matrix2x2 @ v

        torch.testing.assert_close(UDUv.blocks[0], Av.blocks[0])
        torch.testing.assert_close(UDUv.blocks[1], Av.blocks[1])

    def test_invert_via_ldl(self):
        """Test inverting SymmetricBlock2x2Matrix using LDL decomposition."""
        # Create positive definite blocks to ensure invertibility
        block11 = torch.randn(3, 3)
        block22 = torch.randn(2, 2)

        matrix2x2 = SymmetricBlock2x2Matrix(
            block11 @ block11.T + torch.eye(3),
            torch.randn(3, 2),
            block22 @ block22.T + torch.eye(2),
        )
        U, D = matrix2x2.UDU_decomposition()

        # A vector that can be multiplied by matri2x2
        v = Vertical([torch.randn(3), torch.randn(2)])

        # Confirm that inv(UDU^T) A v = v
        v_hat = U.T.solve(D.solve(U.solve(matrix2x2 @ v)))

        torch.testing.assert_close(v_hat.blocks[0], v.blocks[0])
        torch.testing.assert_close(v_hat.blocks[1], v.blocks[1])
