"""
Unit tests for hessian.py
"""

import pytest
import torch
import torch.nn as nn

import hessian
from hessian import DenseBlock, LossLayer, SequenceOfDenseBlocks
import partitioned


class TestDenseBlock:
    """Test BasicBlock derivatives with analytical solutions."""

    @pytest.fixture
    def batch_size(self):
        return 1

    @pytest.fixture
    def input_dim(self):
        return 3

    @pytest.fixture
    def output_dim(self):
        return 4

    @pytest.fixture
    def num_params(self, input_dim, output_dim):
        return input_dim * output_dim

    @pytest.fixture
    def z_in(self, batch_size, input_dim):
        return torch.randn(batch_size, input_dim, requires_grad=True)

    @pytest.fixture
    def dloss_dz(self, batch_size, output_dim):
        return torch.randn(batch_size, output_dim)

    @pytest.fixture
    def block(self, input_dim, output_dim, z_in):
        """Create a DenseBlock with Identity activation and populate its cache."""
        block = DenseBlock(input_dim, output_dim, nn.Identity())
        block(z_in)
        return block

    def test_derivatives_identity_activation(
        self, input_dim, output_dim, num_params, z_in, block, dloss_dz
    ):
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

    @pytest.fixture
    def block_square(self, input_dim, output_dim, z_in):
        """Create a DenseBlock with square activation and populate its cache."""
        block = DenseBlock(input_dim, output_dim, lambda x: x**2)
        block(z_in)
        return block

    def test_derivatives_square_activation(
        self,
        input_dim,
        output_dim,
        num_params,
        z_in,
        block_square,
        dloss_dz,
    ):
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
        z_linear = block_square.linear(z_in).flatten()

        assert block_square.linear.weight.shape == (output_dim, input_dim)

        derivs = block_square.derivatives(dloss_dz)

        expected_Dx = 2 * torch.kron(torch.diag(z_linear), z_in)
        assert derivs.Dx.shape == (output_dim, num_params)
        torch.testing.assert_close(derivs.Dx, expected_Dx)

        expected_Dz = 2 * z_linear.reshape(-1, 1) * block_square.linear.weight
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
            * block_square.linear.weight.T
            @ torch.diag(dloss_dz.flatten())
            @ block_square.linear.weight
        )
        assert expected_DD_Dzz.shape == (input_dim, input_dim)
        assert derivs.DM_Dzz.shape == (input_dim, input_dim)
        torch.testing.assert_close(derivs.DM_Dzz, expected_DD_Dzz)

    def test_derivatives_linear_activation_numerical(
        self, z_in, block_square, dloss_dz
    ):
        derivs = block_square.derivatives(dloss_dz)
        dz = 1e-4 * torch.randn_like(z_in)

        params = dict(block_square.named_parameters())
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
            return torch.func.functional_call(block_square, x, (z,))

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
            return torch.func.functional_call(block_square, x, (z,)) @ dloss_dz.T

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

    @pytest.fixture
    def batch_size(self):
        return 1

    @pytest.fixture
    def input_dim(self):
        return 5

    @pytest.fixture
    def num_classes(self):
        return 10

    @pytest.fixture
    def z_in(self, batch_size, input_dim):
        return torch.randn(batch_size, input_dim, requires_grad=True)

    @pytest.fixture
    def target(self, batch_size, num_classes):
        return torch.randint(0, num_classes, (batch_size,))

    @pytest.fixture
    def loss_layer(self, input_dim, num_classes):
        return LossLayer(input_dim, num_classes)

    @pytest.fixture
    def output(self, loss_layer, z_in, target):
        return loss_layer(z_in, target)

    def test_forward_logging(self, z_in, loss_layer, output):
        """
        Test that the base class's forward() method properly logs input and output.
        """
        # Verify that input and output are logged
        assert loss_layer.input is not None, "Input should be logged"
        assert loss_layer.output is not None, "Output should be logged"
        torch.testing.assert_close(loss_layer.input, z_in)
        torch.testing.assert_close(loss_layer.output, output)

    def test_forward_returns_scalar(self, output):
        """
        Test that forward() returns a scalar despite num_classes=10.
        """
        # Verify output is a scalar
        assert (
            output.ndim == 0
        ), f"Output should be scalar (0-dim), got {output.ndim}-dim"
        assert (
            output.numel() == 1
        ), f"Output should have 1 element, got {output.numel()}"

        # Verify it's a valid loss value (non-negative for cross-entropy)
        assert output.item() >= 0, "Cross-entropy loss should be non-negative"

    def test_functional_call_forward(self, z_in, loss_layer, target, output):
        """
        Test calling LossLayer.forward() via torch.func.functional_call.

        Verifies that the functional call returns the same scalar loss as the
        regular call and that input/output logging still occurs on the module.
        """
        # Regular forward call already done by output fixture
        assert output.ndim == 0 and output.numel() == 1

        # functional_call with current parameters
        params = dict(loss_layer.named_parameters())
        out_func = torch.func.functional_call(loss_layer, params, (z_in, target))

        # Outputs should match
        torch.testing.assert_close(out_func, output)

        # Logging should reflect the latest call (functional_call above)
        assert loss_layer.input is not None and loss_layer.output is not None
        torch.testing.assert_close(loss_layer.input, z_in)
        torch.testing.assert_close(loss_layer.output, out_func)

    def test_backward_grad_shape(self, output, loss_layer, num_classes, input_dim):
        """
        Test that gradients can be computed for LossLayer parameters and that
        weight.grad has the expected shape (num_classes, input_dim).
        """
        assert output.ndim == 0
        output.backward()

        assert (
            loss_layer.linear.weight.grad is not None
        ), "weight.grad should exist after backward()"
        assert loss_layer.linear.weight.grad.shape == (
            num_classes,
            input_dim,
        ), f"Expected weight.grad shape ({num_classes}, {input_dim}), got {loss_layer.linear.weight.grad.shape}"

    def test_functional_call_jacrev_linear_weight_shape(
        self, z_in, loss_layer, num_classes, input_dim, target
    ):
        """
        Compute jacrev via torch.func.jacrev on a functional_call and confirm
        that the jacobian for linear.weight has the expected shape.
        """

        def loss_fn(p):
            return torch.func.functional_call(
                loss_layer, p, (z_in.flatten(), target.squeeze())
            )

        jac = torch.func.jacrev(loss_fn)(dict(loss_layer.named_parameters()))

        assert "linear.weight" in jac
        assert jac["linear.weight"].shape == (num_classes, input_dim)

    def test_derivatives_shapes(self, input_dim, num_classes, z_in, loss_layer, target):
        """
        Test that derivatives() returns derivatives with the correct shapes.

        Note: derivatives() computes derivatives with respect to the logits output
        (before applying cross-entropy loss), so output_dim = num_classes.
        """
        num_params = input_dim * num_classes

        loss_layer(z_in, target)

        # Compute derivatives
        derivs = loss_layer.derivatives(torch.tensor([1.0]), target)

        # Verify shapes
        assert derivs.Dx.shape == (
            1,
            num_params,
        ), f"Dx shape should be (1, {num_params}), got {derivs.Dx.shape}"

        assert derivs.Dz.shape == (
            1,
            input_dim,
        ), f"Dz shape should be (1, {input_dim}), got {derivs.Dz.shape}"

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


class TestSequenceOfBlocks:
    @pytest.fixture
    def model_config(self):
        """Create a fresh model configuration for each test."""
        return dict(input_dim=3, hidden_dim=4, num_classes=6, num_layers=4)

    @pytest.fixture
    def model(self, model_config):
        """Create a fresh model instance for each test."""
        return SequenceOfDenseBlocks(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_classes=model_config["num_classes"],
            num_layers=model_config["num_layers"],
            activation=torch.tanh,
        )

    @pytest.fixture
    def z_in(self, model_config):
        """Create a fresh input tensor for each test."""
        batch_size = 1
        return torch.randn(batch_size, model_config["input_dim"], requires_grad=True)

    @pytest.fixture
    def target(self, model_config):
        """Create a fresh target tensor for each test."""
        batch_size = 1
        return torch.randint(0, model_config["num_classes"], (batch_size,))

    def test_forward_basic(self, model, z_in, target):
        loss = model(z_in, target)

        # Output is scalar
        assert loss.ndim == 0 and loss.numel() == 1

    def test_gradient_of_loss_wrt_layer_outputs(self, model, z_in, target):
        """Test that ∂z_L/∂z_ℓ (gradient of loss w.r.t. each layer output) has correct shape."""
        with model.save_dloss_douts():
            loss = model(z_in, target)
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

    def test_gradient_wrt_parameters_vs_torch_func(self, model, z_in, target):
        """
        Test gradient_wrt_parameters by comparing it to torch.func.grad.

        Compares:
        1. Result from gradient_wrt_parameters()
        2. Result from torch.func.grad()
        """
        Dx, Dz, DD_Dxx, DD_Dzx, DM_Dzz = model.derivatives(z_in, target)

        M = partitioned.IdentityWithLowerBlockDiagonalMatrix((-Dz).blocks[1:])
        e_L = partitioned.Vertical(
            [torch.zeros(layer.output.numel(), 1) for layer in model]
        )
        assert e_L.blocks[-1].numel() == 1
        e_L.blocks[-1][:] = 1.0
        dloss_dx = Dx.T @ M.T.solve(e_L)
        grad_flat = dloss_dx.to_tensor().flatten()

        # Method 2: Use loss.backward
        grad_torch_flat = torch.cat([p.grad.flatten() for p in model.parameters()])

        # Compare the two methods
        assert grad_flat.shape == grad_torch_flat.shape
        torch.testing.assert_close(grad_flat, grad_torch_flat)

    def test_hessian_vector_product_vs_torch_func(self, model, z_in, target):
        """
        Test hessian_vector_product by comparing it to torch.func.hessian.

        Creates a random partitioned vector v with blocks matching the parameter
        count of each layer, then compares:
        1. Result from hessian_vector_product(v)
        2. Result from torch.func.hessian(...) @ v.flatten()
        """
        # A random partitioned vector  each of which has as many elements as the
        # corresponding layer has parameters.
        v = partitioned.Vertical(
            [
                torch.randn(sum(p.numel() for p in layer.parameters()), 1)
                for layer in model
            ]
        )

        # Method 1: Use hessian_vector_product
        hvp_result = model.hessian_vector_product(z_in, target, v)
        hvp_flat = hvp_result.to_tensor()

        # Method 2: Use torch.func.hessian
        def loss_fn(x):
            return torch.func.functional_call(model, x, (z_in, target))

        hessian_dict = torch.func.hessian(loss_fn)(dict(model.named_parameters()))
        hvp_torch = hessian.flatten_2d_pytree(hessian_dict) @ v.to_tensor()

        # Compare the two methods
        torch.testing.assert_close(hvp_flat, hvp_torch, rtol=1e-4, atol=1e-5)

    def test_torch_hessian_is_invertible(self, model, z_in, target):
        """
        Test that the Hessian is invertible by explicitly computing it using torch.func.hessian,
        then check its condition number.

        If this test fails, the whole adventure of efficiently inverting the Hessian is for naught.
        """

        def loss_fn(x):
            return torch.func.functional_call(model, x, (z_in, target))

        hessian_dict = torch.func.hessian(loss_fn)(dict(model.named_parameters()))
        H = hessian.flatten_2d_pytree(hessian_dict)
        singular_values = torch.linalg.svd(H).S
        assert (
            singular_values[0] / singular_values[-1] < 1e6
        ), "Hessian is not invertible: singular_values = " + str(singular_values)

    def test_hessian_inverse_is_inverse_of_hessian(self, model, z_in, target):
        """
        Test that H^{-1} is actually the inverse of H by verifying H @ H^{-1} @ g = g.

        Generate a random vector g, compute h_inv_g = H^{-1} @ g using hessian_inverse_product,
        then compute h_h_inv_g = H @ h_inv_g using hessian_vector_product,
        and verify that h_h_inv_g ≈ g.
        """
        # Create a random vector g matching the parameter structure
        g = partitioned.Vertical(
            [
                torch.randn(sum(p.numel() for p in layer.parameters()), 1)
                for layer in model
            ]
        )

        # Compute H^{-1} @ g
        h_inv_g = model.hessian_inverse_product(z_in, target, g)

        # Compute H @ (H^{-1} @ g)
        h_h_inv_g = model.hessian_vector_product(z_in, target, h_inv_g)

        # Verify that H @ H^{-1} @ g = g
        torch.testing.assert_close(
            h_h_inv_g.to_tensor(), g.to_tensor(), rtol=1e-3, atol=1e-4
        )

    def test_hessian_inverse_product_vs_torch_func(self, model, z_in, target):
        """
        Test hessian_inverse_product by comparing it to torch.func.hessian.

        Computes H^{-1} explicitly using torch.linalg.inv(torch.func.hessian(...))
        and compares the result of hessian_inverse_product(g) against H^{-1} @ g.
        """
        # Create a random vector g
        g = partitioned.Vertical(
            [
                torch.randn(sum(p.numel() for p in layer.parameters()), 1)
                for layer in model
            ]
        )

        # Method 1: Use hessian_inverse_product
        hinv_g_result = model.hessian_inverse_product(z_in, target, g)
        hinv_g_flat = hinv_g_result.to_tensor()

        # Method 2: Use torch.func.hessian and explicit inversion
        def loss_fn(x):
            return torch.func.functional_call(model, x, (z_in, target))

        hessian_dict = torch.func.hessian(loss_fn)(dict(model.named_parameters()))
        H = hessian.flatten_2d_pytree(hessian_dict)
        hinv_g_torch = torch.linalg.solve(H, g.to_tensor())

        # Compare the two methods
        torch.testing.assert_close(hinv_g_flat, hinv_g_torch, rtol=1e-3, atol=1e-4)
