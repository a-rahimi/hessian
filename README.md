Shows how to apply the inverse of the Hessian of a deep net to a vector.

Pearlmutter showed a clever way to compute the product of the Hessian of a deep
net with a vector.  This repo shows how to comptue the product of the
**inverse** of the Hessian with a vector. If the Hessian-vector product is $H
v$ for some fixed vector $v$, we're interested in solving $H x = v$ for $x$.
Solving this system naively requires a number of operations that scales
cubically with the number of parameters in the deep net, which is impractical
for most modern networks.  The trick is to augment the system of equations
$Hx=v$ with auxiliary variables, pivot the system into a block-tri-diagonal
system, factor that system, and then solve it. This, in effect, ends up looking
like running propagation on a network that is the dual of the original network.

The full idea is described in [hessian.pdf](hessian.pdf). For a quick look at
the algortihm, see the [hessian_inverse_product](src/hessian.py#L269).

# Partitioned Matrix Library

The algorithm relies heavily on operations on structured, partitioned, block
matrices, so the code includes a library for manipuating block-partitioned
matrices in [block_partitioned_matrices.py](src/block_partitioned_matrices.py).
See [the tutorial](src/tutorial_block_partitioned_matrices.ipynb).