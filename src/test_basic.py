"""Tests to confirm basic understanding of Python and some libraries."""

import pytest


class TestPytextFixtureScope:
    """Ensure fixtures are recreated for each test in a class.

    The pytest documentation is circumspect about this. It mentions the scope
    for functions, but not for methods, so we verify that fixtures are
    re-executed for each method in the class.
    """

    @pytest.fixture
    def stuff(self):
        return []

    def test_stuff_is_empty(self, stuff):
        assert stuff == []

        # Mutate it. We'll check if this test can affect other tests in the
        # class.
        stuff.append(1)

    def test_stuff_is_still_empty(self, stuff):
        assert stuff == []


class TestMatmulOperatorAssociativity:
    """Test the order of operations for chained @ operators."""

    def test_matmul_chain_is_left_associative(self):
        "Test that A @ B @ C is evaluated as (A @ B) @ C (left-to-right)."

        class TrackedMatrix(str):
            def __matmul__(self, other):
                return TrackedMatrix(f"({self} @ {other})")

        A = TrackedMatrix("A")
        B = TrackedMatrix("B")
        C = TrackedMatrix("C")

        # Some sanity checks on our custom @ operator.
        assert A @ B == "(A @ B)"
        assert (A @ B) @ C == "((A @ B) @ C)"
        assert A @ (B @ C) == "(A @ (B @ C))"

        # The actual test of left-associativity.
        assert A @ B @ C == "((A @ B) @ C)", "@ is not left-associative"
