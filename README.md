# The Hessian of very deep networks is easy to invert

This package shows how to multiply the inverse of the Hessian of a deep network
with a vector.  If the Hessian-vector
*product is $H v$ for some fixed vector $v$, we're interested in solving $H x =
*v$ for $x$.
The hope is to soon use this as a preconditioner to speed up stochastic gradient
descent.

[Pearlmutter](https://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf) showed a
clever way to compute the Hessian-vector-product.  This repo shows how to
compute the Hessian-inverse-product, the product of the
**inverse** of the Hessian of a deep net with a vector.
Solving this system naively requires a number of operations that scales
cubically with the number of parameters in the deep net, which is impractical
for most modern networks.  The trick is to augment the system of equations
$Hx=v$ with auxiliary variables, pivot the system into a block-tri-diagonal
system, factor that system, and then solve it. These steps, in effect, end up
looking like running propagation on a network that is the dual of the original
network.

The full idea is described in [hessian.pdf](hessian.pdf). For a demo, see
[demo_hessian.ipynb](src/demo_hessian.ipynb). For a quick look at how the
algortihm is implemented, see the
[hessian_inverse_product](src/hessian.py#L269).

# Partitioned Matrix Library

The algorithm relies heavily on operations on structured, hierarchically nested, block
matrices. The code includes a library for manipuating block-partitioned
matrices in [block_partitioned_matrices.py](src/block_partitioned_matrices.py).
See [the tutorial](src/tutorial_block_partitioned_matrices.ipynb).