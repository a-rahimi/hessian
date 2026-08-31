"""
Unit tests for hessian.py
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

import hessian
from hessian import (
    CrossEntropyLayer,
    DenseBlock,
    LossLayer,
    SequenceOfBlocks,
    SequenceOfDenseBlocks,
)
import block_partitioned_matrices as bpm


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

        # Dz/DM_Dzz are now computed per-sample and returned as a nested
        # bpm.Diagonal (see docs/batch-structure-plan.md Phase 1); unwrap to a
        # dense tensor to compare against the analytical reference.
        assert (derivs.Dz.height, derivs.Dz.width) == (output_dim, input_dim)
        torch.testing.assert_close(derivs.Dz.to_tensor(), expected_Dz)

        torch.testing.assert_close(
            derivs.DD_Dxx, torch.zeros_like(derivs.DD_Dxx)
        )
        torch.testing.assert_close(
            derivs.DM_Dzz.to_tensor(), torch.zeros(input_dim, input_dim)
        )

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
        assert (derivs.Dz.height, derivs.Dz.width) == (output_dim, input_dim)
        torch.testing.assert_close(derivs.Dz.to_tensor(), expected_Dz)

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
        assert (derivs.DM_Dzz.height, derivs.DM_Dzz.width) == (input_dim, input_dim)
        torch.testing.assert_close(derivs.DM_Dzz.to_tensor(), expected_DD_Dzz)


class TestBatchStructuredDerivatives:
    """Test the per-sample nested-Diagonal representation of Dz/DM_Dzz.

    See docs/batch-structure-plan.md Phase 1: for an ordinary (non-loss)
    layer, sample i's output depends only on sample i's input, so Dz and
    DM_Dzz should come back as a bpm.Diagonal of `batch` independent
    sub-blocks rather than a dense (batch*w)^2 tensor. The oracle is the
    pre-change dense computation (torch.func.jacrev/hessian over the
    batch-flattened input).
    """

    @pytest.fixture
    def batch_size(self):
        return 5

    @pytest.fixture
    def input_dim(self):
        return 3

    @pytest.fixture
    def output_dim(self):
        return 4

    @pytest.fixture
    def z_in(self, batch_size, input_dim):
        return torch.randn(batch_size, input_dim, requires_grad=True)

    @pytest.fixture
    def dloss_dz(self, batch_size, output_dim):
        return torch.randn(batch_size, output_dim)

    @pytest.fixture
    def block(self, input_dim, output_dim, z_in):
        block = DenseBlock(input_dim, output_dim, torch.tanh)
        block(z_in)
        return block

    def dense_reference(self, block, z_in, dloss_dz, input_dim, output_dim):
        """Recompute Dz/DM_Dzz the pre-change way: dense jacrev/hessian over
        the batch-flattened input."""
        params = dict(block.named_parameters())
        input_shape = z_in.shape
        z_flat = z_in.flatten().detach()

        def f(z):
            return torch.func.functional_call(block, params, (z.reshape(input_shape),))

        def dloss_dz_f(z):
            return dloss_dz.flatten() @ f(z).flatten()

        Dz = torch.func.jacrev(f)(z_flat).reshape(-1, z_flat.numel())
        DM_Dzz = torch.func.hessian(dloss_dz_f)(z_flat)
        return Dz, DM_Dzz

    def test_dz_is_nested_diagonal_with_one_block_per_sample(
        self, block, z_in, dloss_dz, batch_size, input_dim, output_dim
    ):
        derivs = block.derivatives(dloss_dz)

        assert isinstance(derivs.Dz, bpm.Diagonal)
        assert derivs.Dz.num_blocks() == batch_size
        for sub_block in derivs.Dz.diagonal_blocks:
            assert sub_block.shape == (output_dim, input_dim)

        assert isinstance(derivs.DM_Dzz, bpm.Diagonal)
        assert derivs.DM_Dzz.num_blocks() == batch_size
        for sub_block in derivs.DM_Dzz.diagonal_blocks:
            assert sub_block.shape == (input_dim, input_dim)

    def test_dz_matches_dense_reference(
        self, block, z_in, dloss_dz, input_dim, output_dim
    ):
        derivs = block.derivatives(dloss_dz)
        Dz_dense_expected, DM_Dzz_dense_expected = self.dense_reference(
            block, z_in, dloss_dz, input_dim, output_dim
        )

        torch.testing.assert_close(
            derivs.Dz.to_tensor(), Dz_dense_expected, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            derivs.DM_Dzz.to_tensor(), DM_Dzz_dense_expected, rtol=1e-5, atol=1e-6
        )

    def test_cross_sample_blocks_are_exactly_zero_in_dense_reference(
        self, block, z_in, dloss_dz, batch_size, input_dim, output_dim
    ):
        """Sanity-check the structural claim itself: the dense reference has
        zero cross-sample coupling, which is exactly what licenses discarding
        those entries and storing only the block-diagonal."""
        _, DM_Dzz_dense = self.dense_reference(block, z_in, dloss_dz, input_dim, output_dim)
        DM_Dzz_blocked = DM_Dzz_dense.reshape(batch_size, input_dim, batch_size, input_dim)
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    torch.testing.assert_close(
                        DM_Dzz_blocked[i, :, j, :], torch.zeros(input_dim, input_dim)
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
        assert output.ndim == 0, (
            f"Output should be scalar (0-dim), got {output.ndim}-dim"
        )
        assert output.numel() == 1, (
            f"Output should have 1 element, got {output.numel()}"
        )

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

        assert loss_layer.linear.weight.grad is not None, (
            "weight.grad should exist after backward()"
        )
        assert loss_layer.linear.weight.grad.shape == (
            num_classes,
            input_dim,
        ), (
            f"Expected weight.grad shape ({num_classes}, {input_dim}), got {loss_layer.linear.weight.grad.shape}"
        )

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
        ), (
            f"DD_Dxx shape should be ({num_params}, {num_params}), got {derivs.DD_Dxx.shape}"
        )

        assert derivs.DD_Dzx.shape == (
            num_params,
            input_dim,
        ), (
            f"DD_Dzx shape should be ({num_params}, {input_dim}), got {derivs.DD_Dzx.shape}"
        )

        assert derivs.DM_Dzz.shape == (
            input_dim,
            input_dim,
        ), (
            f"DM_Dzz shape should be ({input_dim}, {input_dim}), got {derivs.DM_Dzz.shape}"
        )


class TestParameterFreeLossLayer:
    """`SequenceOfDenseBlocks` ends in a loss layer that owns no parameters.

    The classifier is its own `DenseBlock` at the end of the chain rather than
    fused into the loss, so the pipeline's last activation is the logits. Nothing
    about the model changes, but every parameter now sits strictly upstream of
    the loss, which is what lets the last DM_Dzz block be read as the curvature
    of a convex function of the logits.
    """

    HIDDEN_DIM, NUM_CLASSES, INPUT_DIM, NUM_LAYERS = 4, 6, 3, 4

    @pytest.fixture
    def model(self):
        return SequenceOfDenseBlocks(
            input_dim=self.INPUT_DIM,
            hidden_dim=self.HIDDEN_DIM,
            num_classes=self.NUM_CLASSES,
            num_layers=self.NUM_LAYERS,
            activation=torch.tanh,
        )

    @pytest.fixture
    def z_in(self):
        return torch.randn(2, self.INPUT_DIM)

    @pytest.fixture
    def target(self):
        return torch.randint(0, self.NUM_CLASSES, (2,))

    def test_loss_layer_owns_no_parameters(self, model):
        assert isinstance(model.loss_layer, CrossEntropyLayer)
        assert list(model.loss_layer.parameters()) == []

    def test_layer_chain_outputs_the_logits(self, model, z_in, target):
        logits = model.layers(z_in)
        assert logits.shape == (z_in.shape[0], self.NUM_CLASSES)
        torch.testing.assert_close(
            model(z_in, target), nn.functional.cross_entropy(logits, target)
        )

    def test_parameters_match_the_fused_arrangement(self, model):
        """Moving the classifier out of the loss layer moved no parameters."""
        fused = SequenceOfBlocks(
            [DenseBlock(self.INPUT_DIM, self.HIDDEN_DIM, torch.tanh)]
            + [
                DenseBlock(self.HIDDEN_DIM, self.HIDDEN_DIM, torch.tanh)
                for _ in range(self.NUM_LAYERS - 2)
            ],
            LossLayer(self.HIDDEN_DIM, self.NUM_CLASSES),
        )
        assert [p.shape for p in model.parameters()] == [
            p.shape for p in fused.parameters()
        ]

    def test_loss_layer_contributes_a_zero_width_parameter_block(
        self, model, z_in, target
    ):
        """The loss layer still gets a block, just one with no columns.

        Keeping the block rather than dropping it is what holds the block count
        equal to the number of layers, which every caller that repacks a flat
        vector into per-layer blocks relies on.
        """
        Dx, _, DD_Dxx, DD_Dzx, _ = model.derivatives(z_in, target)
        assert Dx.num_blocks() == len(list(model))
        assert Dx.diagonal_blocks[-1].shape == (1, 0)
        assert DD_Dxx.diagonal_blocks[-1].shape == (0, 0)
        assert DD_Dzx.diagonal_blocks[-1].shape[0] == 0

    def test_loss_curvature_block_is_positive_semidefinite(
        self, model, z_in, target
    ):
        """The last DM_Dzz block is ∇_zz of cross-entropy in its own logits."""
        *_, DM_Dzz = model.derivatives(z_in, target)
        block = DM_Dzz.diagonal_blocks[-1].to_tensor().detach()
        eigenvalues = torch.linalg.eigvalsh(0.5 * (block + block.T))
        assert float(eigenvalues.min()) > -1e-6


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

    @pytest.fixture
    def epsilon(self):
        return 0.1

    @pytest.fixture
    def random_parameter_vector(self, model):
        """
        A random partitioned vector each of which has as many elements as the
        corresponding layer has parameters.
        """
        return bpm.Vertical(
            [
                torch.randn(sum(p.numel() for p in layer.parameters()), 1)
                for layer in model
            ]
        )

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
            assert layer.dloss_dout.shape == layer.output.shape, (
                f"Layer {layer_idx} dloss_dout shape {layer.dloss_dout.shape} != output shape {layer.output.shape}"
            )

        # Check loss layer as well (scalar loss)
        assert hasattr(model.loss_layer, "dloss_dout")
        assert model.loss_layer.dloss_dout is not None
        # loss is scalar; grad shape should match loss shape
        assert model.loss_layer.dloss_dout.shape == loss.shape, (
            f"Loss layer dloss_dout shape {model.loss_layer.dloss_dout.shape} != loss shape {loss.shape}"
        )

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

        # Dz is a per-sample nested Diagonal; densify it so M solves against the
        # dense e_L below, matching how hessian_vector_product builds M.
        Dz = hessian._densify_per_sample_blocks(Dz)
        M = bpm.IdentityWithLowerDiagonal((-Dz).flat[1:])
        e_L = bpm.Vertical([torch.zeros(layer.output.numel(), 1) for layer in model])
        assert e_L.flat[-1].numel() == 1
        e_L.flat[-1][:] = 1.0
        dloss_dx = Dx.T @ M.T.solve(e_L)
        grad_flat = dloss_dx.to_tensor().flatten()

        # Method 2: Use loss.backward
        grad_torch_flat = torch.cat([p.grad.flatten() for p in model.parameters()])

        # Compare the two methods
        assert grad_flat.shape == grad_torch_flat.shape
        torch.testing.assert_close(grad_flat, grad_torch_flat)

    @pytest.fixture
    def z_in_batch(self, model_config):
        """Create a fresh input tensor for each test with batch_size=3."""
        batch_size = 3
        return torch.randn(batch_size, model_config["input_dim"], requires_grad=True)

    @pytest.fixture
    def target_batch(self, model_config):
        """Create a fresh target tensor for each test with batch_size=3."""
        batch_size = 3
        return torch.randint(0, model_config["num_classes"], (batch_size,))

    def test_hessian_vector_product_vs_torch_func(
        self, model, z_in, target, random_parameter_vector
    ):
        """
        Test hessian_vector_product by comparing it to torch.func.hessian.

        Creates a random partitioned vector v with blocks matching the parameter
        count of each layer, then compares:
        1. Result from hessian_vector_product(v)
        2. Result from torch.func.hessian(...) @ v.flatten()
        """

        # Method 1: Use hessian_vector_product
        hvp_result = model.hessian_vector_product(z_in, target, random_parameter_vector)
        hvp_flat = hvp_result.to_tensor()

        # Method 2: Use torch.func.hessian
        def loss_fn(x):
            return torch.func.functional_call(model, x, (z_in, target))

        hessian_dict = torch.func.hessian(loss_fn)(dict(model.named_parameters()))
        hvp_torch = (
            hessian.flatten_2d_pytree(hessian_dict)
            @ random_parameter_vector.to_tensor()
        )

        # Compare the two methods
        torch.testing.assert_close(hvp_flat, hvp_torch, rtol=1e-4, atol=1e-5)

    def test_hessian_vector_product_vs_torch_func_batch(
        self, model, z_in_batch, target_batch, random_parameter_vector
    ):
        """
        Test hessian_vector_product by comparing it to torch.func.hessian with batch_size=3.

        Creates a random partitioned vector v with blocks matching the parameter
        count of each layer, then compares:
        1. Result from hessian_vector_product(v)
        2. Result from torch.func.hessian(...) @ v.flatten()
        """

        # Method 1: Use hessian_vector_product
        hvp_result = model.hessian_vector_product(
            z_in_batch, target_batch, random_parameter_vector
        )
        hvp_flat = hvp_result.to_tensor()

        # Method 2: Use torch.func.hessian
        def loss_fn(x):
            return torch.func.functional_call(model, x, (z_in_batch, target_batch))

        hessian_dict = torch.func.hessian(loss_fn)(dict(model.named_parameters()))
        hvp_torch = (
            hessian.flatten_2d_pytree(hessian_dict)
            @ random_parameter_vector.to_tensor()
        )

        # Compare the two methods
        torch.testing.assert_close(hvp_flat, hvp_torch, rtol=1e-4, atol=1e-5)

    def test_torch_hessian_is_invertible(self, model, z_in, target, epsilon):
        """
        Test that the Hessian is invertible by explicitly computing it using torch.func.hessian,
        then check its condition number.

        If this test fails, the whole adventure of efficiently inverting the Hessian is for naught.
        """

        # Compute and store the Hessian by running backpropagation on the
        # gradient of the loss.
        def loss_fn(x):
            return torch.func.functional_call(model, x, (z_in, target))

        hessian_dict = torch.func.hessian(loss_fn)(dict(model.named_parameters()))
        H = hessian.flatten_2d_pytree(hessian_dict)
        H = H + epsilon * torch.eye(H.shape[0])

        singular_values = torch.linalg.svd(H).S
        assert singular_values[0] / singular_values[-1] < 1e6, (
            "Hessian is not invertible: singular_values = " + str(singular_values)
        )

    def test_hessian_inverse_product_vs_torch_func(
        self, model, z_in, target, epsilon, random_parameter_vector
    ):
        """
        Test hessian_inverse_product by comparing it to torch.func.hessian.

        Computes H explicitly using torch.func.hessian(...), and solve Hx=b
        using torch.linalg.solve.  Compare this result against that of
        hessian_inverse_product.
        """
        # Method 1: Use hessian_inverse_product
        hinv_g_result = model.hessian_inverse_product(
            z_in, target, random_parameter_vector, epsilon
        )
        hinv_g_flat = hinv_g_result.to_tensor()

        # Method 2: Use torch.func.hessian and explicit inversion
        def loss_fn(x):
            return torch.func.functional_call(model, x, (z_in, target))

        hessian_dict = torch.func.hessian(loss_fn)(dict(model.named_parameters()))
        H = hessian.flatten_2d_pytree(hessian_dict)
        H = H + epsilon * torch.eye(H.shape[0])
        # Solve in float64 so the reference solve itself adds no error. The
        # remaining discrepancy is float32 noise between the two independently
        # computed Hessians, amplified by the conditioning of H, so the
        # tolerance cannot be much tighter than 1e-3.
        hinv_g_torch = torch.linalg.solve(
            H.detach().double(), random_parameter_vector.to_tensor().double()
        ).float()

        # Compare the two methods
        torch.testing.assert_close(hinv_g_flat, hinv_g_torch, rtol=1e-3, atol=1e-3)

    def test_hessian_inverse_product_vs_torch_func_batch(
        self, model, z_in_batch, target_batch, epsilon, random_parameter_vector
    ):
        """
        Test hessian_inverse_product by comparing it to torch.func.hessian with batch_size=3.

        Computes H explicitly using torch.func.hessian(...), and solve Hx=b
        using torch.linalg.solve.  Compare this result against that of
        hessian_inverse_product.
        """
        # Method 1: Use hessian_inverse_product
        hinv_g_result = model.hessian_inverse_product(
            z_in_batch, target_batch, random_parameter_vector, epsilon
        )
        hinv_g_flat = hinv_g_result.to_tensor()

        # Method 2: Use torch.func.hessian and explicit inversion
        def loss_fn(x):
            return torch.func.functional_call(model, x, (z_in_batch, target_batch))

        hessian_dict = torch.func.hessian(loss_fn)(dict(model.named_parameters()))
        H = hessian.flatten_2d_pytree(hessian_dict)
        H = H + epsilon * torch.eye(H.shape[0])
        hinv_g_torch = torch.linalg.solve(H, random_parameter_vector.to_tensor())

        # Compare the two methods
        torch.testing.assert_close(hinv_g_flat, hinv_g_torch, rtol=1e-2, atol=1e-2)

    def test_hessian_inverse_solve_splu_matches_block(
        self, model, z_in, target, epsilon, random_parameter_vector
    ):
        """The splu and block solvers should agree to float32 accuracy on a small model."""
        setup = model.hessian_inverse_setup(z_in, target)
        x_splu = model.hessian_inverse_solve(
            setup, random_parameter_vector, epsilon, solver="splu"
        )
        x_block = model.hessian_inverse_solve(
            setup, random_parameter_vector, epsilon, solver="block"
        )

        for splu_block, block_block in zip(x_splu.flat, x_block.flat):
            assert splu_block.shape == block_block.shape
            assert splu_block.dtype == block_block.dtype

        torch.testing.assert_close(
            x_splu.to_tensor(), x_block.to_tensor(), rtol=1e-3, atol=1e-3
        )

    def test_hessian_inverse_solve_splu_vs_dense_float64(
        self, model, z_in, target, epsilon, random_parameter_vector
    ):
        """The splu solver should match a dense float64 solve of the same augmented system."""
        setup = model.hessian_inverse_setup(z_in, target)
        x_splu = model.hessian_inverse_solve(
            setup, random_parameter_vector, epsilon, solver="splu"
        )

        Dx, DD_Dxx, DD_Dzx, DM_Dzz, M, P, zero_block = setup
        K = bpm.Generic(
            [
                [DD_Dxx + epsilon * bpm.Identity(DD_Dxx.height), DD_Dzx @ P, Dx.T],
                [-Dx, M, zero_block],
                [-P.T @ DD_Dzx.T, -P.T @ DM_Dzz @ P, M.T],
            ]
        )
        K_dense = K.to_tensor().detach().to(torch.float64)

        b_flat = random_parameter_vector.to_tensor().to(torch.float64)
        rhs = torch.zeros(K_dense.shape[0], 1, dtype=torch.float64)
        rhs[: b_flat.shape[0]] = b_flat

        xyz = torch.linalg.solve(K_dense, rhs)
        x_ref = xyz[: b_flat.shape[0]].to(torch.float32)

        torch.testing.assert_close(
            x_splu.to_tensor(), x_ref, rtol=1e-5, atol=1e-6
        )

    def test_hessian_inverse_solve_rejects_unknown_solver(
        self, model, z_in, target, epsilon, random_parameter_vector
    ):
        setup = model.hessian_inverse_setup(z_in, target)
        with pytest.raises(ValueError, match="solver"):
            model.hessian_inverse_solve(
                setup, random_parameter_vector, epsilon, solver="qr"
            )

    def test_hessian_inverse_is_inverse_of_hessian(
        self, model, z_in, target, epsilon, random_parameter_vector
    ):
        """
        Test H^{-1} is actually the inverse of H by verifying H @ H^{-1} @ g = g.
        """
        # Compute (H + epsilon I)^{-1} @ g
        h_inv_g = model.hessian_inverse_product(
            z_in, target, random_parameter_vector, epsilon
        )

        # Compute (H + epsilon I) @ (H + epsilon I)^{-1} @ g
        h_h_inv_g = (
            model.hessian_vector_product(z_in, target, h_inv_g)
            + bpm.ScaledIdentity(epsilon) @ h_inv_g
        )

        # Verify that H @ H^{-1} @ g = g
        torch.testing.assert_close(
            h_h_inv_g.to_tensor(),
            random_parameter_vector.to_tensor(),
            rtol=1e-4,
            atol=1e-4,
        )


class TestGaussNewton:
    """`gauss_newton_setup` puts G = J' Λ J through the same augmented system as H.

    Every reference here is built from autograd rather than from the block
    machinery, so the tests check the machinery instead of restating it.
    """

    INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, NUM_LAYERS, BATCH = 3, 4, 3, 4, 2

    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        return SequenceOfDenseBlocks(
            input_dim=self.INPUT_DIM,
            hidden_dim=self.HIDDEN_DIM,
            num_classes=self.NUM_CLASSES,
            num_layers=self.NUM_LAYERS,
            activation=torch.tanh,
        )

    @pytest.fixture
    def z_in(self):
        torch.manual_seed(1)
        return torch.randn(self.BATCH, self.INPUT_DIM)

    @pytest.fixture
    def target(self):
        torch.manual_seed(2)
        return torch.randint(0, self.NUM_CLASSES, (self.BATCH,))

    @pytest.fixture
    def rhs(self, model):
        torch.manual_seed(3)
        return bpm.Vertical(
            [
                torch.randn(sum(p.numel() for p in layer.parameters()), 1)
                for layer in model
            ]
        )

    def _flat_params(self, model):
        """The parameters as one flat vector, plus the inverse of that packing."""
        params = dict(model.named_parameters())
        names = list(params)

        def unflatten(v):
            out, offset = {}, 0
            for name in names:
                n = params[name].numel()
                out[name] = v[offset : offset + n].view_as(params[name])
                offset += n
            return out

        flat = torch.cat([params[name].detach().reshape(-1) for name in names])
        return flat, unflatten

    def _logits_fn(self, model, z_in):
        """The map from a flat parameter vector to the flattened logits.

        The loss layer holds no parameters, so the layer chain's output is the
        loss's input, which is what the Gauss-Newton matrix linearizes.
        """
        _, unflatten = self._flat_params(model)

        def logits(v):
            params = {
                name[len("layers.") :]: t for name, t in unflatten(v).items()
            }
            return torch.func.functional_call(model.layers, params, (z_in,)).reshape(-1)

        return logits

    def _loss_of_logits(self, target):
        return lambda r: nn.functional.cross_entropy(
            r.reshape(self.BATCH, self.NUM_CLASSES), target
        )

    def _dense_gauss_newton(self, model, z_in, target):
        flat, _ = self._flat_params(model)
        logits = self._logits_fn(model, z_in)
        J = torch.func.jacrev(logits)(flat)
        Lam = torch.func.hessian(self._loss_of_logits(target))(logits(flat).detach())
        return J.T @ Lam @ J

    def _dense_hessian(self, model, z_in, target):
        flat, unflatten = self._flat_params(model)
        return torch.func.hessian(
            lambda v: torch.func.functional_call(model, unflatten(v), (z_in, target))
        )(flat)

    @pytest.mark.parametrize("solver", ["splu", "block"])
    @pytest.mark.parametrize("epsilon", [0.1, 1.0])
    def test_solve_matches_a_dense_gauss_newton_solve(
        self, model, z_in, target, rhs, solver, epsilon
    ):
        G = self._dense_gauss_newton(model, z_in, target)
        want = torch.linalg.solve(
            G + epsilon * torch.eye(G.shape[0]), rhs.to_tensor().flatten()
        )
        got = model.gauss_newton_inverse_product(
            z_in, target, rhs, epsilon, solver=solver
        )
        torch.testing.assert_close(
            got.to_tensor().flatten(), want, rtol=1e-4, atol=1e-5
        )

    def test_gauss_newton_is_psd_where_the_hessian_is_indefinite(
        self, model, z_in, target
    ):
        """The reason to build G at all: G^-1 reverses no descent direction."""
        G = self._dense_gauss_newton(model, z_in, target)
        H = self._dense_hessian(model, z_in, target)
        assert float(torch.linalg.eigvalsh(0.5 * (G + G.T)).min()) > -1e-5
        assert float(torch.linalg.eigvalsh(0.5 * (H + H.T)).min()) < -1e-3

    def test_hessian_exceeds_gauss_newton_by_the_network_curvature(
        self, model, z_in, target
    ):
        """H = G + sum_i (dloss/dlogit_i) * grad_xx logit_i, the identity defining G."""
        flat, _ = self._flat_params(model)
        logits = self._logits_fn(model, z_in)
        dloss_dlogits = torch.func.grad(self._loss_of_logits(target))(
            logits(flat).detach()
        )
        logit_hessians = torch.func.jacrev(torch.func.jacrev(logits))(flat)
        network_curvature = torch.einsum("i,ijk->jk", dloss_dlogits, logit_hessians)

        torch.testing.assert_close(
            self._dense_hessian(model, z_in, target),
            self._dense_gauss_newton(model, z_in, target) + network_curvature,
            rtol=1e-4,
            atol=1e-5,
        )

    def test_hessian_setup_is_left_alone(self, model, z_in, target):
        """Masking blocks for G must not disturb the Hessian's own setup."""
        hessian_setup = model.hessian_inverse_setup(z_in, target)
        model.gauss_newton_setup(z_in, target)
        after = model.hessian_inverse_setup(z_in, target)
        torch.testing.assert_close(
            hessian_setup.DD_Dxx.to_tensor(), after.DD_Dxx.to_tensor()
        )
        torch.testing.assert_close(
            hessian_setup.DM_Dzz.to_tensor(), after.DM_Dzz.to_tensor()
        )


def test_section3_sanity_check_dense_vs_linear_inverse(capsys):
    """
    Section 3 sanity check from reproduce-published-results/report.md.

    Build a tiny SequenceOfDenseBlocks, materialize the full dense Hessian H via
    torch.func.hessian on the cross-entropy loss, and for several values of ε
    compare torch.linalg.solve(H + ε I, g) against
    model.hessian_inverse_product(x, y, g_pytree, ε). The test passes when the
    relative error is below 1e-4 for every ε. Per-ε relative errors are printed
    so the result is human-readable when the test is run with -s.

    The test runs in float64 because ε = 1e-3 makes H + ε I poorly
    conditioned, so float32 round-off alone can push the relative error above
    1e-4 even though the algorithm itself is correct. We confirmed in
    development that at float64 the error drops to 1e-13 for every ε, which
    pins the float32 hit on precision rather than on the linear-inverse code.
    """
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        _run_section3_check(capsys)
    finally:
        torch.set_default_dtype(prev_dtype)


def _run_section3_check(capsys):
    torch.manual_seed(42)

    image_size = 4
    hidden_dim = 4
    num_layers = 2
    num_classes = 3
    batch_size = 8

    model = SequenceOfDenseBlocks(
        input_dim=image_size,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        activation=torch.tanh,
    )

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 300, (
        f"Parameter count {total_params} exceeds the Section 3 budget of 300; "
        "shrink the model so the dense Hessian stays cheap."
    )

    x = torch.randn(batch_size, image_size, requires_grad=False)
    y = torch.randint(0, num_classes, (batch_size,))

    # Build the dense Hessian via torch.func.hessian on a flat-parameter loss.
    def loss_fn(params_dict):
        return torch.func.functional_call(model, params_dict, (x, y))

    hessian_dict = torch.func.hessian(loss_fn)(dict(model.named_parameters()))
    H = hessian.flatten_2d_pytree(hessian_dict)
    assert H.shape == (total_params, total_params)

    # Random gradient laid out as a per-layer Vertical block, matching the
    # layout that hessian_inverse_product expects.
    g_pytree = bpm.Vertical(
        [
            torch.randn(sum(p.numel() for p in layer.parameters()), 1)
            for layer in model
        ]
    )
    g_flat = g_pytree.to_tensor().flatten()
    assert g_flat.shape == (total_params,)

    epsilons = [1e-3, 1e-1, 1.0, 10.0]
    rel_errors = {}
    eye = torch.eye(total_params)
    for eps in epsilons:
        x_dense = torch.linalg.solve(H + eps * eye, g_flat)
        x_ours = (
            model.hessian_inverse_product(x, y, g_pytree, eps).to_tensor().flatten()
        )
        num = torch.linalg.vector_norm(x_dense - x_ours).item()
        den = torch.linalg.vector_norm(x_dense).item()
        rel_errors[eps] = num / den

    # Print per-ε relative errors so the test result is human-readable.
    with capsys.disabled():
        print()
        print(
            f"[section3] P={total_params}, batch={batch_size}, "
            f"layers={num_layers}, hidden={hidden_dim}, image_size={image_size}"
        )
        for eps in epsilons:
            print(f"[section3] eps={eps:<8g} rel_err={rel_errors[eps]:.3e}")

    for eps, rel in rel_errors.items():
        assert rel < 1e-4, (
            f"Section 3 sanity check failed at eps={eps}: rel_err={rel:.3e} >= 1e-4. "
            "Do not loosen the tolerance; investigate the algorithm first."
        )


def test_to_scipy_csc_matches_dense():
    torch.manual_seed(0)
    M = bpm.Generic(
        [
            [torch.randn(3, 3), bpm.Zero((3, 2)), torch.randn(3, 4)],
            [torch.randn(2, 3), bpm.Identity(2), bpm.Zero((2, 4))],
            [
                bpm.Zero((4, 3)),
                torch.randn(4, 2),
                bpm.Diagonal([torch.randn(2, 2), bpm.ScaledIdentity(2.5, 2)]),
            ],
        ]
    )

    sparse = M.to_scipy_csc()

    assert sparse.dtype == np.float64
    assert sparse.shape == (M.height, M.width)
    np.testing.assert_allclose(sparse.toarray(), M.to_tensor().numpy(), atol=1e-7)
