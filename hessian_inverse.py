"""
Implementation of the Hessian-inverse-vector product algorithm
as described in "The Hessian of tall-skinny networks is easy to invert"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class LayerDerivatives:
    """Stores first and second-order derivatives for a single layer."""
    # First-order derivatives
    grad_x: torch.Tensor  # ∇_x f_ℓ: gradient w.r.t. parameters (a × p)
    grad_z: torch.Tensor  # ∇_z f_ℓ: gradient w.r.t. inputs (a × a)
    
    # Second-order derivatives
    hess_xx: torch.Tensor  # ∇_{xx} f_ℓ: Hessian w.r.t. parameters (ap × p)
    hess_zx: torch.Tensor  # ∇_{zx} f_ℓ: mixed partial (ap × a)
    hess_xz: torch.Tensor  # ∇_{xz} f_ℓ: mixed partial (a² × p)
    hess_zz: torch.Tensor  # ∇_{zz} f_ℓ: Hessian w.r.t. inputs (a² × a)
    
    # Backprop value
    b: torch.Tensor  # b_ℓ: ∂z_L/∂z_ℓ


class LayerWithActivation(nn.Module):
    """A single layer f_ℓ(z_{ℓ-1}; x_ℓ) = activation(W @ z_{ℓ-1} + bias)"""
    
    def __init__(self, input_dim: int, output_dim: int, activation: str = 'tanh'):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation
        
        # Use smooth activation for differentiability
        if activation == 'tanh':
            self.act_fn = torch.tanh
        elif activation == 'sigmoid':
            self.act_fn = torch.sigmoid
        elif activation == 'softplus':
            self.act_fn = nn.functional.softplus
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        return self.act_fn(self.linear(x))


class LossLayer(nn.Module):
    """Final layer that fuses the last linear layer with the loss computation."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x, targets):
        """
        Args:
            x: input activations
            targets: target labels
        Returns:
            scalar loss value
        """
        logits = self.linear(x)
        loss = F.cross_entropy(logits, targets)
        return loss


class DeepMLPWithTracking(nn.Module):
    """
    Deep MLP that tracks intermediate activations for Hessian computation.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, 
                 num_hidden_layers: int = 19, activation: str = 'tanh'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_hidden_layers = num_hidden_layers
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(LayerWithActivation(input_dim, hidden_dim, activation))
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(LayerWithActivation(hidden_dim, hidden_dim, activation))
        
        # Final layer (includes loss)
        self.loss_layer = LossLayer(hidden_dim, num_classes)
        
        # Storage for intermediate activations
        self.activations: List[torch.Tensor] = []
    
    def forward(self, x, targets=None, track_activations=True):
        """
        Forward pass with optional activation tracking.
        
        Args:
            x: input tensor (batch_size, input_dim)
            targets: target labels (batch_size,) for loss computation
            track_activations: if True, store intermediate activations
        
        Returns:
            loss if targets provided, else final activations
        """
        if track_activations:
            self.activations = [x]
        
        z = x
        for layer in self.layers:
            z = layer(z)
            if track_activations:
                self.activations.append(z)
        
        if targets is not None:
            loss = self.loss_layer(z, targets)
            return loss
        else:
            return self.loss_layer.linear(z)
    
    def get_num_layers(self):
        """Total number of layers including final layer."""
        return len(self.layers) + 1
    
    def get_layer_params(self, layer_idx: int):
        """Get parameters for layer at given index."""
        if layer_idx < len(self.layers):
            return list(self.layers[layer_idx].parameters())
        else:
            return list(self.loss_layer.parameters())


def compute_layer_derivatives(model: DeepMLPWithTracking, 
                              layer_idx: int,
                              loss: torch.Tensor,
                              activations: List[torch.Tensor]) -> LayerDerivatives:
    """
    Compute all first and second-order derivatives for a single layer.
    
    Args:
        model: the neural network
        layer_idx: index of the layer (0 to L-1)
        loss: the scalar loss z_L
        activations: list of intermediate activations [z_0, z_1, ..., z_L]
    
    Returns:
        LayerDerivatives object containing all derivatives
    """
    L = model.get_num_layers()
    
    # Get layer parameters and input activation
    params = model.get_layer_params(layer_idx)
    z_input = activations[layer_idx]  # z_{ℓ-1}
    
    # For the last layer, z_output is the loss (scalar)
    # For other layers, z_output is the activation
    if layer_idx == L - 1:
        # Last layer: output is the loss itself
        z_output = loss
        is_loss_layer = True
    else:
        z_output = activations[layer_idx + 1]  # z_ℓ
        is_loss_layer = False
    
    # Flatten parameters into a single vector
    param_sizes = [p.numel() for p in params]
    p = sum(param_sizes)  # total number of parameters
    
    # Get dimensions (accounting for batch dimension)
    if is_loss_layer:
        a = 1  # Loss layer outputs a scalar
    else:
        a = z_output.shape[-1]  # output dimension
    
    a_in = z_input.shape[-1]  # input dimension
    
    # Compute backprop value b_ℓ = ∂z_L/∂z_ℓ
    if is_loss_layer:
        # For loss layer, b_L = 1
        b = torch.ones(1, device=z_input.device)
    else:
        b_result = torch.autograd.grad(loss, z_output, create_graph=True, retain_graph=True)[0]
        # Average over batch if needed
        if b_result.dim() > 1:
            b = b_result.mean(dim=0)  # Average over batch
        else:
            b = b_result
    
    # Compute ∇_x f_ℓ: gradient of output w.r.t. parameters
    if is_loss_layer:
        # Gradient of scalar loss w.r.t. parameters
        grad_x_list = []
        for param in params:
            grad = torch.autograd.grad(z_output, param, create_graph=True, retain_graph=True)[0]
            grad_x_list.append(grad.flatten())
        grad_x = torch.cat(grad_x_list).unsqueeze(0)  # 1 × p
    else:
        # Gradient of vector output w.r.t. parameters
        # Average over batch dimension first
        z_out_mean = z_output.mean(dim=0) if z_output.dim() > 1 else z_output
        grad_x = torch.zeros(a, p, device=z_input.device)
        
        for i in range(a):
            grad_x_list = []
            for param in params:
                grad = torch.autograd.grad(z_out_mean[i], param, create_graph=True, retain_graph=True,
                                          allow_unused=True)[0]
                if grad is None:
                    grad = torch.zeros_like(param)
                grad_x_list.append(grad.flatten())
            grad_x[i] = torch.cat(grad_x_list)
    
    # Compute ∇_z f_ℓ: gradient of output w.r.t. input
    if is_loss_layer:
        # Gradient of scalar w.r.t. input
        grad_z_result = torch.autograd.grad(z_output, z_input, create_graph=True, retain_graph=True)[0]
        # Average over batch if needed
        if grad_z_result.dim() > 1:
            grad_z = grad_z_result.mean(dim=0).unsqueeze(0)  # 1 × a_in
        else:
            grad_z = grad_z_result.unsqueeze(0)
    else:
        # Gradient of vector w.r.t. input
        z_out_mean = z_output.mean(dim=0) if z_output.dim() > 1 else z_output
        grad_z = torch.zeros(a, a_in, device=z_input.device)
        for i in range(a):
            grad_z[i] = torch.autograd.grad(z_out_mean[i], z_input, create_graph=True, retain_graph=True)[0].mean(dim=0) if z_input.dim() > 1 else torch.autograd.grad(z_out_mean[i], z_input, create_graph=True, retain_graph=True)[0]
    
    # Second-order derivatives
    # ∇_{xx} f_ℓ: Hessian w.r.t. parameters (ap × p)
    grad_x_flat = grad_x.flatten()  # ap elements
    hess_xx = torch.zeros(grad_x_flat.shape[0], p, device=z_input.device)
    for i in range(grad_x_flat.shape[0]):
        hess_row = []
        for param in params:
            h = torch.autograd.grad(grad_x_flat[i], param, create_graph=True, retain_graph=True,
                                   allow_unused=True)[0]
            if h is None:
                h = torch.zeros_like(param)
            hess_row.append(h.flatten())
        hess_xx[i] = torch.cat(hess_row)
    
    # ∇_{zx} f_ℓ: mixed partial (ap × a_in)
    hess_zx = torch.zeros(grad_x_flat.shape[0], a_in, device=z_input.device)
    for i in range(grad_x_flat.shape[0]):
        h = torch.autograd.grad(grad_x_flat[i], z_input, create_graph=True, retain_graph=True,
                               allow_unused=True)[0]
        if h is not None:
            hess_zx[i] = h.mean(dim=0) if h.dim() > 1 else h
    
    # ∇_{xz} f_ℓ: mixed partial (a*a_in × p)
    grad_z_flat = grad_z.flatten()  # a * a_in elements
    hess_xz = torch.zeros(grad_z_flat.shape[0], p, device=z_input.device)
    for i in range(grad_z_flat.shape[0]):
        hess_row = []
        for param in params:
            h = torch.autograd.grad(grad_z_flat[i], param, create_graph=True, retain_graph=True,
                                   allow_unused=True)[0]
            if h is None:
                h = torch.zeros_like(param)
            hess_row.append(h.flatten())
        hess_xz[i] = torch.cat(hess_row)
    
    # ∇_{zz} f_ℓ: Hessian w.r.t. inputs (a*a_in × a_in)
    hess_zz = torch.zeros(grad_z_flat.shape[0], a_in, device=z_input.device)
    for i in range(grad_z_flat.shape[0]):
        h = torch.autograd.grad(grad_z_flat[i], z_input, create_graph=True, retain_graph=True,
                               allow_unused=True)[0]
        if h is not None:
            hess_zz[i] = h.mean(dim=0) if h.dim() > 1 else h
    
    return LayerDerivatives(
        grad_x=grad_x,
        grad_z=grad_z,
        hess_xx=hess_xx,
        hess_zx=hess_zx,
        hess_xz=hess_xz,
        hess_zz=hess_zz,
        b=b
    )


def compute_all_layer_derivatives(model: DeepMLPWithTracking,
                                  loss: torch.Tensor,
                                  activations: List[torch.Tensor]) -> List[LayerDerivatives]:
    """Compute derivatives for all layers."""
    L = model.get_num_layers()
    derivatives = []
    
    for layer_idx in range(L):
        derivs = compute_layer_derivatives(model, layer_idx, loss, activations)
        derivatives.append(derivs)
    
    return derivatives


# Block-diagonal matrix representation
class BlockDiagonalMatrix:
    """Represents a block-diagonal matrix as a list of blocks."""
    
    def __init__(self, blocks: List[torch.Tensor]):
        self.blocks = blocks
    
    def apply(self, v: torch.Tensor) -> torch.Tensor:
        """Apply block-diagonal matrix to vector v."""
        result_blocks = []
        offset = 0
        for block in self.blocks:
            size = block.shape[1]
            v_block = v[offset:offset + size]
            result_blocks.append(block @ v_block)
            offset += size
        return torch.cat(result_blocks)
    
    def apply_transpose(self, v: torch.Tensor) -> torch.Tensor:
        """Apply transpose of block-diagonal matrix to vector v."""
        result_blocks = []
        offset = 0
        for block in self.blocks:
            size = block.shape[0]
            v_block = v[offset:offset + size]
            result_blocks.append(block.T @ v_block)
            offset += size
        return torch.cat(result_blocks)


class BlockBidiagonalMatrix:
    """Represents a block bi-diagonal matrix with diagonal and lower-diagonal blocks."""
    
    def __init__(self, diag_blocks: List[torch.Tensor], lower_blocks: List[torch.Tensor]):
        """
        Args:
            diag_blocks: diagonal blocks (L blocks)
            lower_blocks: lower diagonal blocks (L-1 blocks, with first being dummy/None)
        """
        self.diag_blocks = diag_blocks
        self.lower_blocks = lower_blocks
    
    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        """
        Solve M x = rhs using back-substitution.
        This is equivalent to backpropagation.
        """
        # Split rhs into blocks
        n_blocks = len(self.diag_blocks)
        block_sizes = [block.shape[0] for block in self.diag_blocks]
        
        rhs_blocks = []
        offset = 0
        for size in block_sizes:
            rhs_blocks.append(rhs[offset:offset + size])
            offset += size
        
        # Back-substitution: solve from last block to first
        x_blocks = [None] * n_blocks
        
        for i in range(n_blocks - 1, -1, -1):
            # x_i = D_i^{-1} (rhs_i - L_{i+1} x_{i+1})
            if i == n_blocks - 1:
                # Last block: x_i = D_i^{-1} rhs_i
                x_blocks[i] = torch.linalg.solve(self.diag_blocks[i], rhs_blocks[i])
            else:
                # x_i = D_i^{-1} (rhs_i - L_{i+1} x_{i+1})
                rhs_adjusted = rhs_blocks[i] - self.lower_blocks[i + 1] @ x_blocks[i + 1]
                x_blocks[i] = torch.linalg.solve(self.diag_blocks[i], rhs_adjusted)
        
        return torch.cat(x_blocks)
    
    def solve_transpose(self, rhs: torch.Tensor) -> torch.Tensor:
        """
        Solve M^T x = rhs using forward-substitution.
        """
        n_blocks = len(self.diag_blocks)
        block_sizes = [block.shape[0] for block in self.diag_blocks]
        
        rhs_blocks = []
        offset = 0
        for size in block_sizes:
            rhs_blocks.append(rhs[offset:offset + size])
            offset += size
        
        # Forward-substitution
        x_blocks = [None] * n_blocks
        
        for i in range(n_blocks):
            if i == 0:
                x_blocks[i] = torch.linalg.solve(self.diag_blocks[i].T, rhs_blocks[i])
            else:
                rhs_adjusted = rhs_blocks[i] - self.lower_blocks[i].T @ x_blocks[i - 1]
                x_blocks[i] = torch.linalg.solve(self.diag_blocks[i].T, rhs_adjusted)
        
        return torch.cat(x_blocks)


def build_M_matrix(derivatives: List[LayerDerivatives]) -> BlockBidiagonalMatrix:
    """
    Build the M matrix from the paper.
    M is block bi-diagonal with I on diagonal and -∇_z f_{ℓ+1} on lower diagonal.
    """
    L = len(derivatives)
    
    diag_blocks = []
    lower_blocks = [None]  # First element is dummy
    
    for i in range(L):
        # Diagonal is identity
        dim = derivatives[i].grad_z.shape[0]
        diag_blocks.append(torch.eye(dim, device=derivatives[i].grad_z.device))
        
        # Lower diagonal is -∇_z f_{i+1}
        if i < L - 1:
            lower_blocks.append(-derivatives[i + 1].grad_z)
    
    return BlockBidiagonalMatrix(diag_blocks, lower_blocks)


def build_P_matrix(derivatives: List[LayerDerivatives]) -> torch.Tensor:
    """
    Build the downshifting matrix P.
    P shifts activations down by one layer.
    """
    L = len(derivatives)
    dims = [deriv.grad_z.shape[0] for deriv in derivatives]
    total_dim = sum(dims)
    
    P = torch.zeros(total_dim, total_dim, device=derivatives[0].grad_z.device)
    
    offset_out = 0
    offset_in = 0
    for i in range(L):
        if i > 0:
            # Copy from block i to block i-1
            dim_out = dims[i - 1]
            dim_in = dims[i]
            P[offset_out:offset_out + dim_out, offset_in:offset_in + dim_in] = torch.eye(dim_out, dim_in)
            offset_out += dim_out
        else:
            offset_out += dims[0]
        offset_in += dims[i]
    
    return P


def build_block_diagonal_with_kronecker(derivatives: List[LayerDerivatives], 
                                       use_b: bool = True) -> BlockDiagonalMatrix:
    """
    Build D_D or D_M matrices which have I ⊗ b_ℓ on diagonal blocks.
    """
    blocks = []
    for deriv in derivatives:
        b = deriv.b
        # I ⊗ b^T creates a block diagonal with b^T repeated
        # For matrix-vector product, this is equivalent to Kronecker product
        dim = deriv.grad_x.shape[1] if use_b else deriv.grad_z.shape[1]
        block = torch.kron(torch.eye(dim, device=b.device), b.unsqueeze(0))
        blocks.append(block)
    
    return BlockDiagonalMatrix(blocks)


def compute_Q_inverse_blocks(derivatives: List[LayerDerivatives],
                            P: torch.Tensor,
                            D_D: BlockDiagonalMatrix,
                            D_M: BlockDiagonalMatrix) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the four blocks of Q^{-1} using the partitioned inverse formula.
    
    Q = [[Q11, Q12],
         [Q21, Q22]]
    
    where:
    Q11 = P^T D_M D_{zz} P
    Q12 = P^T D_M D_{xz}
    Q21 = D_D D_{zx} P
    Q22 = D_D D_{xx} - I
    
    Returns the four blocks of Q^{-1}.
    """
    # Build block-diagonal D_{zz}, D_{xz}, D_{zx}, D_{xx}
    D_zz = BlockDiagonalMatrix([deriv.hess_zz for deriv in derivatives])
    D_xz = BlockDiagonalMatrix([deriv.hess_xz for deriv in derivatives])
    D_zx = BlockDiagonalMatrix([deriv.hess_zx for deriv in derivatives])
    D_xx = BlockDiagonalMatrix([deriv.hess_xx for deriv in derivatives])
    
    device = derivatives[0].grad_x.device
    
    # Compute Q blocks as dense matrices (for simplicity)
    # Q11 = P^T D_M D_{zz} P
    total_a_dim = sum(deriv.grad_z.shape[0] for deriv in derivatives)
    total_p_dim = sum(deriv.grad_x.shape[1] for deriv in derivatives)
    
    # Build these as dense for now (can be optimized with block structure)
    Q11 = P.T @ dense_from_block_diag(D_M, total_a_dim, total_a_dim) @ \
          dense_from_block_diag(D_zz, total_a_dim * total_a_dim, total_a_dim) @ P
    
    Q12 = P.T @ dense_from_block_diag(D_M, total_a_dim, total_a_dim) @ \
          dense_from_block_diag(D_xz, total_a_dim * total_a_dim, total_p_dim)
    
    Q21 = dense_from_block_diag(D_D, total_p_dim, total_a_dim * total_p_dim) @ \
          dense_from_block_diag(D_zx, total_a_dim * total_p_dim, total_a_dim) @ P
    
    Q22 = dense_from_block_diag(D_D, total_p_dim, total_a_dim * total_p_dim) @ \
          dense_from_block_diag(D_xx, total_a_dim * total_p_dim, total_p_dim) - \
          torch.eye(total_p_dim, device=device)
    
    # Compute Q^{-1} using partitioned inverse
    # Q22_inv
    Q22_inv = torch.linalg.inv(Q22)
    
    # Schur complement S = Q11 - Q12 Q22^{-1} Q21
    S = Q11 - Q12 @ Q22_inv @ Q21
    S_inv = torch.linalg.inv(S)
    
    # Four blocks of Q^{-1}
    Q_inv_11 = S_inv
    Q_inv_12 = -S_inv @ Q12 @ Q22_inv
    Q_inv_21 = -Q22_inv @ Q21 @ S_inv
    Q_inv_22 = Q22_inv + Q22_inv @ Q21 @ S_inv @ Q12 @ Q22_inv
    
    return Q_inv_11, Q_inv_12, Q_inv_21, Q_inv_22


def dense_from_block_diag(block_mat: BlockDiagonalMatrix, out_dim: int, in_dim: int) -> torch.Tensor:
    """Convert a block-diagonal matrix to dense format."""
    device = block_mat.blocks[0].device
    result = torch.zeros(out_dim, in_dim, device=device)
    
    row_offset = 0
    col_offset = 0
    for block in block_mat.blocks:
        rows, cols = block.shape
        result[row_offset:row_offset + rows, col_offset:col_offset + cols] = block
        row_offset += rows
        col_offset += cols
    
    return result


def hessian_inverse_vector_product(model: DeepMLPWithTracking,
                                   loss: torch.Tensor,
                                   gradient: torch.Tensor) -> torch.Tensor:
    """
    Compute H^{-1} g where H is the Hessian and g is the gradient.
    
    Implements the algorithm from Section 5.1 of the paper (lines 670-741).
    
    Args:
        model: the neural network
        loss: the scalar loss
        gradient: gradient vector (flattened parameters)
    
    Returns:
        H^{-1} g (preconditioned gradient)
    """
    # Compute all derivatives
    activations = model.activations
    derivatives = compute_all_layer_derivatives(model, loss, activations)
    
    L = len(derivatives)
    device = gradient.device
    
    # Build D_x (block diagonal with ∇_x f_ℓ on diagonal)
    D_x_blocks = [deriv.grad_x for deriv in derivatives]
    D_x = BlockDiagonalMatrix(D_x_blocks)
    
    # Build M matrix
    M = build_M_matrix(derivatives)
    
    # Build P matrix  
    P = build_P_matrix(derivatives)
    
    # Build D_D and D_M (involving Kronecker products with b_ℓ)
    D_D = build_block_diagonal_with_kronecker(derivatives, use_b=True)
    D_M = build_block_diagonal_with_kronecker(derivatives, use_b=False)
    
    total_a_dim = sum(deriv.grad_z.shape[0] for deriv in derivatives)
    total_p_dim = sum(deriv.grad_x.shape[1] for deriv in derivatives)
    
    # Step 1: Compute g' = [D_x; I] g
    g_prime_top = D_x.apply(gradient)
    g_prime = torch.cat([g_prime_top, gradient])
    
    # Step 2: Compute Q^{-1} and form A
    Q_inv_11, Q_inv_12, Q_inv_21, Q_inv_22 = compute_Q_inverse_blocks(
        derivatives, P, D_D, D_M
    )
    
    # Build Q_inv as block matrix
    Q_inv = torch.cat([
        torch.cat([Q_inv_11, Q_inv_12], dim=1),
        torch.cat([Q_inv_21, Q_inv_22], dim=1)
    ], dim=0)
    
    # Build M_hat (block diagonal with M on top-left and I on bottom-right)
    M_hat_size = total_a_dim + total_p_dim
    M_hat = torch.eye(M_hat_size, device=device)
    
    # Fill in M part (we need M as dense matrix)
    M_dense = torch.zeros(total_a_dim, total_a_dim, device=device)
    offset = 0
    for i, diag_block in enumerate(M.diag_blocks):
        size = diag_block.shape[0]
        M_dense[offset:offset + size, offset:offset + size] = diag_block
        if i < len(M.lower_blocks) - 1 and M.lower_blocks[i + 1] is not None:
            next_size = M.lower_blocks[i + 1].shape[0]
            M_dense[offset + size:offset + size + next_size, offset:offset + size] = M.lower_blocks[i + 1]
        offset += size
    
    M_hat[:total_a_dim, :total_a_dim] = M_dense
    
    # Compute M_hat Q^{-1} M_hat^T
    MQM = M_hat @ Q_inv @ M_hat.T
    
    # Build D_x as dense
    D_x_dense = dense_from_block_diag(D_x, total_a_dim, total_p_dim)
    
    # Build A = M_hat Q^{-1} M_hat^T + [[D_x D_x^T, D_x], [D_x^T, I]]
    A = MQM.clone()
    A[:total_a_dim, :total_a_dim] += D_x_dense @ D_x_dense.T
    A[:total_a_dim, total_a_dim:] += D_x_dense
    A[total_a_dim:, :total_a_dim] += D_x_dense.T
    A[total_a_dim:, total_a_dim:] += torch.eye(total_p_dim, device=device)
    
    # Step 3: Apply A^{-1} to g' using LDL^T decomposition
    # Extract blocks of A
    A_11 = A[:total_a_dim, :total_a_dim]
    A_12 = A[:total_a_dim, total_a_dim:]
    A_22 = A[total_a_dim:, total_a_dim:]
    
    # Compute A^{-1} using block LDL^T
    A_22_inv = torch.linalg.inv(A_22)
    schur = A_11 - A_12 @ A_22_inv @ A_12.T
    schur_inv = torch.linalg.inv(schur)
    
    # Apply A^{-1} to g' efficiently
    # A^{-1} = [[I, -A_12 A_22^{-1}], [0, I]]^T @ 
    #          [[schur^{-1}, 0], [0, A_22^{-1}]] @
    #          [[I, -A_12 A_22^{-1}], [0, I]]
    
    g_prime_top_part = g_prime[:total_a_dim]
    g_prime_bottom_part = g_prime[total_a_dim:]
    
    # First apply [[I, -A_12 A_22^{-1}], [0, I]]
    temp_top = g_prime_top_part - A_12 @ A_22_inv @ g_prime_bottom_part
    temp_bottom = g_prime_bottom_part
    
    # Apply [[schur^{-1}, 0], [0, A_22^{-1}]]
    temp2_top = schur_inv @ temp_top
    temp2_bottom = A_22_inv @ temp_bottom
    
    # Apply [[I, -A_12 A_22^{-1}], [0, I]]^T = [[I, 0], [-A_22^{-1} A_12^T, I]]
    g_double_prime_top = temp2_top
    g_double_prime_bottom = temp2_bottom - A_22_inv @ A_12.T @ temp2_top
    
    g_double_prime = torch.cat([g_double_prime_top, g_double_prime_bottom])
    
    # Step 4: Compute y = g - [D_x; I]^T g''
    result_top = g_double_prime[:total_a_dim]
    result_bottom = g_double_prime[total_a_dim:]
    
    final_result = gradient - (D_x.apply_transpose(result_top) + result_bottom)
    
    return final_result


if __name__ == "__main__":
    # Test the implementation
    print("Testing Hessian-inverse implementation...")
    
    # Create a small model
    model = DeepMLPWithTracking(
        input_dim=10,
        hidden_dim=5,
        num_classes=3,
        num_hidden_layers=3
    )
    
    # Create dummy data
    x = torch.randn(2, 10, requires_grad=True)
    targets = torch.randint(0, 3, (2,))
    
    # Forward pass
    loss = model(x, targets, track_activations=True)
    
    print(f"Loss: {loss.item()}")
    print(f"Number of activations tracked: {len(model.activations)}")
    
    # Compute derivatives for first layer
    print("\nComputing derivatives for first layer...")
    derivs = compute_layer_derivatives(model, 0, loss, model.activations)
    
    print(f"grad_x shape: {derivs.grad_x.shape}")
    print(f"grad_z shape: {derivs.grad_z.shape}")
    print(f"hess_xx shape: {derivs.hess_xx.shape}")
    print(f"hess_zx shape: {derivs.hess_zx.shape}")
    print(f"hess_xz shape: {derivs.hess_xz.shape}")
    print(f"hess_zz shape: {derivs.hess_zz.shape}")
    
    print("\nBasic implementation complete!")

