# A fixed-temperature trust-region sampler

## Target

Sample from the Gibbs distribution

    π(x) ∝ exp(−L(x) / T),

with L the population loss and T > 0 a fixed temperature. The mode of π is the
minimizer of L, and a small T makes π concentrate there, so drawing from π at a small
fixed T spends most steps near the peak. Nothing is annealed; T is a constant of the run,
the way a learning rate is.

## Reference point: SGD is unadjusted Langevin

With an identity preconditioner, the discretized Langevin update is

    x_{t+1} = x_t − ε g_B(x_t) + sqrt(2 ε T) ξ_t,     ξ_t ~ N(0, I),        (SGLD)

where g_B is the minibatch gradient. The drift −ε g_B is one SGD step of size ε, and the
added noise sqrt(2 ε T) ξ is the only thing separating sampling from optimization, because
at T = 0 this is plain SGD. So a fixed temperature is a fixed step size plus a fixed noise
injection: ε sets how far each step moves and T sets how much it jitters. That is the
"fixed step size in SGD" the temperature corresponds to.

## The trust-region sampler

Replace the identity preconditioner with the damped inverse curvature the method already
computes,

    M = (H + λ I)^{−1},

where H is the structured curvature at x and λ ≥ 0 is the trust-region damping. The
preconditioned Langevin update is

    x_{t+1} = x_t + p_t + sqrt(2 T) · M^{1/2} ξ_t,     ξ_t ~ N(0, I),        (★)

with

    p_t = −(H + λ I)^{−1} g_B(x_t).                    ← the trust-region / damped-Newton step

Equation (★) is the whole proposal, and p_t is exactly the trust-region step this codebase
produces, namely the damped solve `hessian_inverse_solve(setup, g, λ)`. So the Newton/TR
update appears as the drift of the sampler, and the sampler in words is "take one
trust-region step, then add curvature-shaped Gaussian noise."

The scaling is why the trust-region step lands in (★) with coefficient one. Preconditioned
Langevin discretizes to `x' = x − (ε / 2T) M g + sqrt(ε) M^{1/2} ξ`, and choosing ε = 2T
makes the drift exactly `−M g = p_t` and the noise scale sqrt(2T). The single free step
scale is folded into T, and the trust radius Δ (equivalently λ) sets ‖p_t‖. So the two
fixed knobs T and Δ play the roles that ε and T play in SGLD, and fixed T is the "fixed
something in TR": it fixes the size of the noise added to a trust-region step whose own
size is fixed by the trust radius.

The preconditioner is the point. `M^{1/2}` shapes the noise so it is large along
low-curvature directions and small along high-curvature ones, which is the correct
Riemannian noise for π, and it is why a curvature-aware sampler can mix faster than
isotropic SGLD when the loss is ill-conditioned.

## The denoised reduction estimator

The accept/reject below needs the change in the population loss, ΔL = L(x') − L(x), and a
single minibatch gives too noisy an estimate, which is what breaks the current acceptance
ratio ρ. Estimate ΔL with two stacked variance reductions:

1. Common-batch pairing. On a fresh batch B, form `Δ̂_B = L_B(x') − L_B(x)` using the same
   B for both terms, so the shared batch noise cancels in the difference and `Δ̂_B` has far
   lower variance than differencing two independent batches.
2. Averaging. Average `Δ̂_B` over k fresh batches, or keep an exponential moving average of
   the paired difference along the trajectory, so the variance falls with k at the cost of
   k forward passes.

Call the result `Δ̂L`. It is a denoised version of the trust-region actual reduction, so it
feeds both the rejector and, if you want, the radius rule.

## The randomized rejector

Accept the proposal x' from (★) with the Metropolis probability for the fixed-T target,

    α = min(1, exp(−Δ̂L / T) · q(x | x') / q(x' | x)),

where q(·|·) are the Gaussian proposal densities of (★). A downhill proposal (Δ̂L < 0) is
accepted with probability near 1, and an uphill proposal is accepted with a probability
that decays with the increase and with 1/T, which is the soft-reject: a noise-sized uphill
move is still taken sometimes, so the chain keeps moving rather than stalling on a hard
threshold. Dropping the accept/reject entirely leaves the unadjusted sampler, which just
iterates (★), is the preconditioned analogue of SGLD, and is the cheapest option.

The proposal ratio q(x|x')/q(x'|x) requires the trust-region step at x' as well, since the
reverse move drifts by `p'(x')`, so a Metropolis correction costs a second curvature solve
at the proposed point. The unadjusted version skips it and accepts the discretization bias.

## Algorithm

```
fixed: temperature T, trust radius Δ (or damping λ), pairing count k
state: x
repeat:
    draw fresh batch B0
    g, H = minibatch gradient and curvature at x on B0
    λ    = damping that puts ||p|| = Δ            # the TR subproblem's damping
    p    = -(H + λ I)^{-1} g                       # <-- the trust-region / Newton step
    ξ    ~ N(0, I)
    x'   = x + p + sqrt(2 T) * (H + λ I)^{-1/2} ξ  # (★): TR-step drift + curvature-shaped noise

    dL   = mean over k fresh batches B of [ L_B(x') - L_B(x) ]   # denoised, common-batch paired

    # randomized rejector (drop this block for the unadjusted sampler)
    logq = log q(x|x') - log q(x'|x)               # Gaussian proposal ratio; needs p'(x')
    if log(uniform()) < (-dL / T) + logq:
        x = x'                                     # accept
    # else keep x
```

## What the fixed knobs mean

- T fixes the exploration. Large T samples a broad region around the optimum, small T
  concentrates on the peak, and T → 0 turns (★) back into the deterministic trust-region
  step, so the optimizer is the zero-temperature limit of this sampler.
- Δ (or λ) fixes the drift scale. It sets ‖p‖ and, through M, the shape and size of the
  noise, so it plays the role the learning rate plays in SGLD. Hold it fixed, or adapt it
  to a target acceptance rate.
- k trades cost for a cleaner acceptance. k = 1 is cheapest and biases the chain most;
  larger k spends forward passes to sharpen `Δ̂L`.

## Honest caveats

- A noisy acceptance biases the stationary distribution, because plugging a noisy `Δ̂L`
  into the Metropolis ratio does not target π exactly. The bias shrinks as k grows but does
  not vanish, and removing it entirely would need pseudo-marginal (unbiased likelihood)
  estimates, which are expensive here.
- Sampling the curvature-shaped noise `M^{1/2} ξ` needs `(H + λ I)^{−1/2}` applied to a
  Gaussian, which the linear-time solver does not directly give. Approximate it with a
  Lanczos or Chebyshev expansion on the solver's matvec, or fall back to isotropic noise
  sqrt(2T) ξ, which is cheaper but samples a slightly different target.
- The state-dependent preconditioner's divergence term is dropped, as is standard in
  practical preconditioned SGLD, so the unadjusted version is approximate and the
  Metropolis version leans on accept/reject to correct only the discretization.
- The drift is still a damped-Newton direction built from a noisy H, so this sampler
  denoises the acceptance and the radius but not the direction, and closing the remaining
  gap to SGD on the stream may still need a larger batch for g and H.
