Shows how to apply the inverse of the Hessian of a deep net against a vector.

Pearlmutter showed a clever way to compute the product of the Hessian of a deep
net against a vector.  This repo shows how to comptue the product of the
**inverse** of the Hessian against a vector. If the Hessian-vector product $H
v$ for some fixed vector $v$, we're interested in solving $H x = v$ for $x$.
The trick is to augment this system of equations with auxiliary variables,
pivoting it into a block-tri-diagonal system, factoring that system, and
solving it. This, in effect, ends up looking like running propagation on a dual
network.

See [hessian.pdf](hessian.pdf) for the full documentation. For a quick look at
the algortihm, see the [hessian_inverse_product](src/hessian.py#L269) function.

# Partitioned Matrix Library

The implementation under [src/](src/) is work in progress.  It relies heavily on
operations on structured, partitioned, block matrices, so the code includes a
library for manipuating block-partitioned matrices in
[block_partitioned_matrices.py](src/block_partitioned_matrices.py).