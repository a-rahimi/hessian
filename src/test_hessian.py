"""
Unit tests for hessian.py
"""

import torch
import torch.nn as nn
from hessian import DenseBlock, LossLayer, SequenceOfBlocks


class TestDenseBlock:
    """Test BasicBlock derivatives with analytical solutions."""

    def test_derivatives_identity_activation(self):
        """
        Test BasicBlock.derivatives() with nn.Identity activation.

        For f(z; W) = flat(z W') = W z', compute all derivatives analytically and
        compare against the deriviates returned by the derivatives() method. We
        use flat() instead of vec() throughout the code.  Unlike the paper,
        which makes heavy use of vec(), we'll use flat() in this test. The
        two important properties of flat are:
          - flat(X) = vec(X').
          - flat(ABC) = (A ⊗ C') flat(B)

        For a tensor-valued function f and a tensor X, ∇_X f(X) is the matrix J
        such that flat(f(X+dX) - f(X)) = J flat(dX). In particular, for linear
        functions f(X) = J flat(X), J is the deriviatve of f wrt X.

        First-order:
        - ∇_W f = ∇_W flat(W z') = ∇_W (I ⊗ z) flat(W) = (I ⊗ z)
        - ∇_z f = ∇_z W flat(z) = W

        Second-order partials of v'f = v'Wz', where v is some constant vector:
        - ∇²_WW v'f = 0
        - ∇²_zz v'f = 0
        - ∇²_zW v'f = ∇_z ∇_W flat(v' W z') = ∇_z ∇_W (v' ⊗ z) flat(W) = ∇_z (v' ⊗ z) = (v' ⊗ 1)
        """
        batch_size = 1
        input_dim = 3
        output_dim = 4
        num_params = input_dim * output_dim

        torch.manual_seed(42)

        z_in = torch.randn(batch_size, input_dim, requires_grad=True)

        # Create a random model and populate its internal cache.
        block = DenseBlock(input_dim, output_dim, nn.Identity())
        block(z_in)

        dloss_dz = torch.randn(batch_size, output_dim)

        assert block.linear.weight.shape == (output_dim, input_dim)

        expected_Dx = torch.kron(torch.eye(output_dim), z_in)
        expected_Dz = block.linear.weight
        expected_DD_Dzx = torch.kron(dloss_dz.reshape(-1, 1), torch.eye(input_dim))
        assert expected_DD_Dzx.shape == (num_params, input_dim)

        derivs = block.derivatives(dloss_dz)

        assert derivs.Dx.shape == (output_dim, num_params)
        torch.testing.assert_close(derivs.Dx, expected_Dx)

        assert derivs.Dz.shape == (output_dim, input_dim)
        torch.testing.assert_close(derivs.Dz, expected_Dz)

        torch.testing.assert_close(derivs.DD_Dxx, torch.zeros_like(derivs.DD_Dxx))
        torch.testing.assert_close(derivs.DM_Dzz, torch.zeros_like(derivs.DM_Dzz))

        assert derivs.DD_Dzx.shape == (num_params, input_dim)
        torch.testing.assert_close(derivs.DD_Dzx, expected_DD_Dzx)

    def test_derivatives_square_activation(self):
        """
        Test BasicBlock.derivatives() with square activation.

        For f(z; W) = flat(z W')^2, compute all derivatives
        analytically and compare against the deriviates returned by the
        derivatives() method.

        To simplify the derivations, define e = flat(zW') so that
        f(z; W) = e^2, and v'f = e'diag(v)e. We showed above that
        de/dW = I ⊗ z, and de/dz = W.

        First-order:
        - ∇_W f = ∇_W e^2 = 2 diag(e) de/dW  = 2 diag(e) ⊗ z
        - ∇_z f = 2 diag(e) ∇_z de/dz =  2 diag(e) W

        Second-order partials of v'f = v'Wz', where v is some constant vector:
        - ∇²_WW v'f = ∇²_WW e'diag(v)e = ∇_W flat(2 e'diag(v) de/dW)
                    = ∇_W flat(2 e'diag(v) (I ⊗ z))
                    = ∇_W 2 (diag(v) ⊗ z)' e
                    = 2 (diag(v) ⊗ z') de/dW
                    = 2 (diag(v) ⊗ z') (I ⊗ z)
        - ∇²_zz v'f = ∇²_zz e'diag(v)e = ∇_z flat(2 e'diag(v) de/dz)
                    = ∇_z flat(2 e'diag(v) W)
                    = 2 W' diag(v) ∇_z e
                    = 2 W' diag(v) W

        """
        batch_size = 1
        input_dim = 3
        output_dim = 4
        num_params = input_dim * output_dim

        torch.manual_seed(42)

        z_in = torch.randn(batch_size, input_dim, requires_grad=True)

        # Create a random model and populate its internal cache.
        block = DenseBlock(input_dim, output_dim, lambda x: x**2)
        block(z_in)
        z_linear = block.linear(z_in).flatten()

        dloss_dz = torch.randn(batch_size, output_dim)

        assert block.linear.weight.shape == (output_dim, input_dim)

        derivs = block.derivatives(dloss_dz)

        expected_Dx = 2 * torch.kron(torch.diag(z_linear), z_in)
        assert derivs.Dx.shape == (output_dim, num_params)
        torch.testing.assert_close(derivs.Dx, expected_Dx)

        expected_Dz = 2 * z_linear.reshape(-1, 1) * block.linear.weight
        assert derivs.Dz.shape == (output_dim, input_dim)
        torch.testing.assert_close(derivs.Dz, expected_Dz)

        expected_DD_Dxx = (
            2
            * torch.kron(torch.diag(dloss_dz.flatten()), z_in.reshape(-1, 1))
            @ torch.kron(torch.eye(output_dim), z_in)
        )
        assert expected_DD_Dxx.shape == (num_params, num_params)
        assert derivs.DD_Dxx.shape == (num_params, num_params)
        torch.testing.assert_close(derivs.DD_Dxx, expected_DD_Dxx)

        expected_DD_Dzz = (
            2
            * block.linear.weight.T
            @ torch.diag(dloss_dz.flatten())
            @ block.linear.weight
        )
        assert expected_DD_Dzz.shape == (input_dim, input_dim)
        assert derivs.DM_Dzz.shape == (input_dim, input_dim)
        torch.testing.assert_close(derivs.DM_Dzz, expected_DD_Dzz)

    def test_derivatives_linear_activation_numerical(self):
        batch_size = 1
        input_dim = 3
        output_dim = 4

        torch.manual_seed(42)

        z_in = torch.randn(batch_size, input_dim, requires_grad=True)

        # Create a random model and populate its internal cache.
        block = DenseBlock(input_dim, output_dim, nn.Identity())
        block(z_in)

        dloss_dz = torch.randn(batch_size, output_dim)

        derivs = block.derivatives(dloss_dz)

        dz = 1e-4 * torch.randn_like(z_in)

        params = dict(block.named_parameters())
        dparams = {
            param_name: 1e-4 * torch.randn_like(param)
            for param_name, param in params.items()
        }
        params_perturbed = {
            param_name: param + dparams[param_name]
            for param_name, param in params.items()
        }
        dx = torch.cat([dparam.flatten() for dparam in dparams.values()])

        def f(x, z):
            return torch.func.functional_call(block, x, (z,))

        torch.testing.assert_close(
            derivs.Dx @ dx,
            (f(params_perturbed, z_in) - f(params, z_in)).flatten(),
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            derivs.Dz @ dz.flatten(),
            (f(params, z_in + dz) - f(params, z_in)).flatten(),
            rtol=1e-6,
            atol=1e-6,
        )
        assert not torch.allclose(
            derivs.Dz @ dz.flatten(),
            (f(params, z_in + 2 * dz) - f(params, z_in)).flatten(),
            rtol=1e-6,
            atol=1e-6,
        ), "Tolerances are not tight enough."

        def dloss_dz_f(x, z):
            return torch.func.functional_call(block, x, (z,)) @ dloss_dz.T

        assert not torch.allclose(
            (dx[None, :] @ derivs.DD_Dzx @ dz.T).flatten(),
            (
                dloss_dz_f(params_perturbed, z_in + 2 * dz) - dloss_dz_f(params, z_in)
            ).flatten(),
            rtol=1e-5,
            atol=1e-5,
        ), "Tolerances are not tight enough."

        torch.testing.assert_close(
            (dx[None, :] @ derivs.DD_Dxx @ dx[:, None]).flatten(),
            (dloss_dz_f(params_perturbed, z_in) - dloss_dz_f(params, z_in)).flatten(),
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            (dx[None, :] @ derivs.DD_Dzx @ dz.T).flatten(),
            (
                dloss_dz_f(params_perturbed, z_in + dz) - dloss_dz_f(params, z_in)
            ).flatten(),
            rtol=1e-4,
            atol=1e-4,
        )


class TestLossLayer:
    """Test LossLayer with num_classes=10."""

    def test_forward_logging(self):
        """
        Test that the base class's forward() method properly logs input and output.
        """
        batch_size = 1
        input_dim = 5
        num_classes = 10

        torch.manual_seed(42)

        # Create input and targets
        z_in = torch.randn(batch_size, input_dim, requires_grad=True)

        # Create LossLayer and call forward with targets
        loss_layer = LossLayer(input_dim, num_classes)
        output = loss_layer(z_in, torch.tensor([3]))

        # Verify that input and output are logged
        assert loss_layer.input is not None, "Input should be logged"
        assert loss_layer.output is not None, "Output should be logged"
        torch.testing.assert_close(loss_layer.input, z_in)
        torch.testing.assert_close(loss_layer.output, output)

    def test_forward_returns_scalar(self):
        """
        Test that forward() returns a scalar despite num_classes=10.
        """
        batch_size = 1
        input_dim = 5
        num_classes = 10

        torch.manual_seed(42)

        # Create input and targets
        z_in = torch.randn(batch_size, input_dim, requires_grad=True)

        # Create LossLayer
        loss_layer = LossLayer(input_dim, num_classes)

        # Compute forward pass with targets (computes loss)
        output = loss_layer(z_in, torch.tensor([7]))

        # Verify output is a scalar
        assert (
            output.ndim == 0
        ), f"Output should be scalar (0-dim), got {output.ndim}-dim"
        assert (
            output.numel() == 1
        ), f"Output should have 1 element, got {output.numel()}"

        # Verify it's a valid loss value (non-negative for cross-entropy)
        assert output.item() >= 0, "Cross-entropy loss should be non-negative"

    def test_functional_call_forward(self):
        """
        Test calling LossLayer.forward() via torch.func.functional_call.

        Verifies that the functional call returns the same scalar loss as the
        regular call and that input/output logging still occurs on the module.
        """
        batch_size = 1
        input_dim = 5
        num_classes = 10

        torch.manual_seed(42)

        z_in = torch.randn(batch_size, input_dim, requires_grad=True)
        target = torch.tensor([2])

        loss_layer = LossLayer(input_dim, num_classes)

        # Regular forward call
        out_regular = loss_layer(z_in, target)
        assert out_regular.ndim == 0 and out_regular.numel() == 1

        # functional_call with current parameters
        params = dict(loss_layer.named_parameters())
        out_func = torch.func.functional_call(loss_layer, params, (z_in, target))

        # Outputs should match
        torch.testing.assert_close(out_func, out_regular)

        # Logging should reflect the latest call (functional_call above)
        assert loss_layer.input is not None and loss_layer.output is not None
        torch.testing.assert_close(loss_layer.input, z_in)
        torch.testing.assert_close(loss_layer.output, out_func)

    def test_backward_grad_shape(self):
        """
        Test that gradients can be computed for LossLayer parameters and that
        weight.grad has the expected shape (num_classes, input_dim).
        """
        batch_size = 1
        input_dim = 5
        num_classes = 10

        torch.manual_seed(0)

        z_in = torch.randn(batch_size, input_dim, requires_grad=True)
        target = torch.tensor([3])

        loss_layer = LossLayer(input_dim, num_classes)

        loss = loss_layer(z_in, target)
        assert loss.ndim == 0
        loss.backward()

        assert (
            loss_layer.linear.weight.grad is not None
        ), "weight.grad should exist after backward()"
        assert loss_layer.linear.weight.grad.shape == (
            num_classes,
            input_dim,
        ), f"Expected weight.grad shape ({num_classes}, {input_dim}), got {loss_layer.linear.weight.grad.shape}"

    def test_functional_call_jacrev_linear_weight_shape(self):
        """
        Compute jacrev via torch.func.jacrev on a functional_call and confirm
        that the jacobian for linear.weight has the expected shape.
        """
        batch_size = 1
        input_dim = 5
        num_classes = 10

        torch.manual_seed(123)

        z_in = torch.randn(batch_size, input_dim)
        target = torch.tensor([4])

        loss_layer = LossLayer(input_dim, num_classes)

        def loss_fn(p):
            return torch.func.functional_call(
                loss_layer, p, (z_in.flatten(), target.squeeze())
            )

        jac = torch.func.jacrev(loss_fn)(dict(loss_layer.named_parameters()))

        assert "linear.weight" in jac
        assert jac["linear.weight"].shape == (num_classes, input_dim)

    def test_derivatives_shapes(self):
        """
        Test that derivatives() returns derivatives with the correct shapes.

        Note: derivatives() computes derivatives with respect to the logits output
        (before applying cross-entropy loss), so output_dim = num_classes.
        """
        batch_size = 1
        input_dim = 5
        num_classes = 10
        num_params = input_dim * num_classes

        torch.manual_seed(42)

        z_in = torch.randn(batch_size, input_dim, requires_grad=True)

        loss_layer = LossLayer(input_dim, num_classes)
        target = torch.tensor([7])
        loss_layer(z_in, target)

        # Compute derivatives
        derivs = loss_layer.derivatives(1.0, target)

        # Verify shapes
        assert derivs.Dx.shape == (
            num_params,
        ), f"Dx shape should be ({num_params}), got {derivs.Dx.shape}"

        assert derivs.Dz.shape == (
            input_dim,
        ), f"Dz shape should be ({input_dim}), got {derivs.Dz.shape}"

        assert derivs.DD_Dxx.shape == (
            num_params,
            num_params,
        ), f"DD_Dxx shape should be ({num_params}, {num_params}), got {derivs.DD_Dxx.shape}"

        assert derivs.DD_Dzx.shape == (
            num_params,
            input_dim,
        ), f"DD_Dzx shape should be ({num_params}, {input_dim}), got {derivs.DD_Dzx.shape}"

        assert derivs.DM_Dzz.shape == (
            input_dim,
            input_dim,
        ), f"DM_Dzz shape should be ({input_dim}, {input_dim}), got {derivs.DM_Dzz.shape}"


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


class TestSequenceOfBlocksForward:
    """Tests for SequenceOfBlocks.forward behavior."""

    def test_forward_basic(self):
        torch.manual_seed(7)
        batch_size = 1
        input_dim = 4
        hidden_dim = 5
        num_classes = 10
        num_layers = 6

        model = SequenceOfBlocks(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
        )

        z_in = torch.randn(batch_size, input_dim, requires_grad=True)
        target = torch.randint(0, num_classes, (batch_size,))

        loss = model(z_in, target)

        # Output is scalar
        assert loss.ndim == 0 and loss.numel() == 1

    def test_gradient_of_loss_wrt_layer_outputs(self):
        """Test that ∂z_L/∂z_ℓ (gradient of loss w.r.t. each layer output) has correct shape."""
        torch.manual_seed(11)
        batch_size = 1
        input_dim = 4
        hidden_dim = 5
        num_classes = 6
        num_layers = 4

        model = SequenceOfBlocks(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
        )

        x = torch.randn(batch_size, input_dim, requires_grad=True)
        target = torch.randint(0, num_classes, (batch_size,))

        with model.save_dloss_douts():
            loss = model(x, target)
            loss.backward()

        # Check each intermediate layer's hook gradient shape matches its output
        for layer_idx, layer in enumerate(model.layers):
            assert hasattr(layer, "dloss_dout"), f"Layer {layer_idx} missing dloss_dout"
            assert layer.dloss_dout is not None, f"Layer {layer_idx} dloss_dout is None"
            assert layer.output is not None, f"Layer {layer_idx} output not cached"
            assert (
                layer.dloss_dout.shape == layer.output.shape
            ), f"Layer {layer_idx} dloss_dout shape {layer.dloss_dout.shape} != output shape {layer.output.shape}"

        # Check loss layer as well (scalar loss)
        assert hasattr(model.loss_layer, "dloss_dout")
        assert model.loss_layer.dloss_dout is not None
        # loss is scalar; grad shape should match loss shape
        assert (
            model.loss_layer.dloss_dout.shape == loss.shape
        ), f"Loss layer dloss_dout shape {model.loss_layer.dloss_dout.shape} != loss shape {loss.shape}"

        # It should in fact be 1.
        assert torch.allclose(model.loss_layer.dloss_dout, torch.ones_like(loss))


if __name__ == "__main__":
    # Run test directly
    test = TestDenseBlock()
    test.test_derivatives_identity_activation()
    test.test_derivatives_square_activation()
    test.test_derivatives_linear_activation_numerical()

    loss_test = TestLossLayer()
    loss_test.test_forward_logging()
    loss_test.test_forward_returns_scalar()
    loss_test.test_derivatives_shapes()

    print("✓ All tests passed!")
