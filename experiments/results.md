# Newton experiment results

One block per completed experiment, appended in completion order. The Interpreter agent writes each block by parsing `experiments/runs/<id>/stdout.log` and comparing against all earlier blocks in this file.

**Schema per block:** YAML frontmatter (id, run_name, commit_hash, descent_score, ranks among prior runs, hypothesis_supported boolean) followed by a one-paragraph narrative interpretation.

**Ranking metric:** `probe_loss` at the final step (lower = better). Secondary: descent slope = (probe_loss[14] - probe_loss[0]) / 14.

---

```yaml
id: exp-001-very-high-eps
run_name: exp-001-very-high-eps
commit_hash: 49c429ce21de99fd6aa6c737b78ca33619b44e54
probe_loss_initial: 2.7178
probe_loss_final: 2.3839
probe_loss_min: 2.3800
descent_slope: -0.01366
rejection_rate: 0.0
rank_among_prior_runs: 1/1
hypothesis_supported: partial
```

Heavy damping (epsilon=100) did produce overall descent on probe_loss, from 2.7178 down to 2.3839, with a clean negative slope of about -0.0137 over the final five steps and zero LM rejections, so the scaled-SGD-like prediction was directionally right. The descent was not monotonic, though, because probe_loss rose at steps 3 and 8 and the trajectory wobbled by roughly 0.07 to 0.10 between adjacent steps, which is large relative to the total descent. That bumpiness, together with a final value of 2.38 rather than the predicted 2.25, suggests two things. First, the per-batch Hessian and gradient mismatch is meaningful even with strong damping because we evaluate on a held-out probe while stepping on minibatch directions, so single-step noise dominates the signal. Second, with epsilon this large the effective step is essentially gradient descent with lr near 1/100 on the well-conditioned directions, which is too small to clear the random-guess plateau at 2.30 in only 15 steps. The plateau-approach shape in the last five steps, where the slope flattens and |g| drops from about 6 to about 1.2, hints that we are stalling near a flat region rather than still meaningfully descending. The next axis worth probing is whether the bumpiness comes from batch-to-batch disagreement, which is testable by reusing batches, or from the damping level itself being mistuned, which is testable by sweeping epsilon downward toward the natural Hessian scale.

---

```yaml
id: exp-002-batch-reuse-K3
run_name: exp-002-batch-reuse-K3
commit_hash: 34aa5ec725a70513bf0f00a1ffaf1c6c311f581e
probe_loss_initial: 2.7178
probe_loss_final: 2.4906
probe_loss_min: 2.3533
descent_slope: 0.04368
rejection_rate: 0.2
rank_among_prior_runs: 2/2
hypothesis_supported: no
```

The batch-reuse hypothesis did not hold up. The prediction was that probe_loss would drop sharply inside each 3-step reuse window and end near 2.10, but the trajectory instead drifted upward over the last five steps with a positive slope of about +0.044, and the final probe_loss of 2.4906 is meaningfully worse than exp-001's 2.3839 and well above the random-guess 2.30 plateau. The minimum of 2.3533 was reached at step 9 and then lost, so even the best moment of this run only matched exp-001's plateau rather than beating it. Within-window descent is also not visible in the trace, because the loss bounces in both directions across consecutive same-batch steps (for example steps 3-5 cover 2.4000 to 2.8261 to 2.7990), which means converging on a fixed batch's loss surface is not what is happening. The LM rejection rate of 0.2, compared to zero in exp-001, reinforces this: with epsilon dropped from 100 to 1, the Newton step along ill-conditioned directions is now large enough that the per-batch quadratic model is regularly wrong about its own batch, so LM has to back off. Taken together with exp-001, the comparison points away from batch noise as the dominant issue. If batch-to-batch disagreement were the bottleneck, freezing the batch for three steps should at least produce local descent on the probe, but it does not. What does correlate with outcome quality across the two runs is the damping level: epsilon=100 with no rejections gave a clean, slow descent to 2.38, while epsilon=1 with 20 percent rejections gave a noisy non-descent to 2.49. The root cause therefore leans toward damping rather than noise, and specifically toward an epsilon that, at this initialization, needs to be somewhere between 1 and 100 to balance effective step size against the validity of the local quadratic model.

