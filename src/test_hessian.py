"""
Unit tests for hessian.py
"""

import torch
import torch.nn as nn
from hessian import DenseBlock


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


if __name__ == "__main__":
    # Run test directly
    test = TestDenseBlock()
    test.test_derivatives_identity_activation()
    test.test_derivatives_square_activation()
    test.test_derivatives_numerical()
    print("✓ All tests passed!")
