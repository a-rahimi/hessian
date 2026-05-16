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

---

```yaml
id: exp-003-epsilon-sweep-down
run_name: exp-003-epsilon-sweep-down
commit_hash: e6adc8a7cf4238679d4a114b95aa249f7d70b417
probe_loss_initial: 2.7178
probe_loss_final: 2.3525
probe_loss_min: 2.2846
descent_slope: -0.00161
rejection_rate: 0.1333
rank_among_prior_runs: 1/3
hypothesis_supported: partial
```

Dropping epsilon from 100 to 10 did improve the final probe_loss versus exp-001, from 2.3839 down to 2.3525, and the minimum reached during the run dropped from 2.3800 to 2.2846 at step 12, so the directional prediction that smaller damping yields a larger effective step was correct. The improvement falls short of the predicted 2.10 target, though, because the trajectory got noticeably bumpier rather than just faster: the run produced two LM rejections (steps 6 and 13, rate 0.133) where exp-001 had none, and the spread between adjacent probe values reaches 0.34 around step 6, compared to roughly 0.07 to 0.10 in exp-001. The shapes line up with the damping story across all three runs. At epsilon=100 (exp-001) we get a clean, slow descent that asymptotes to the 2.30 plateau because the effective step is too small to break through. At epsilon=10 (exp-003) the per-step bites are larger and the run actually clears the plateau briefly (four steps below 2.30 between step 8 and step 12) before drifting back up, because once the iterate enters a region where the per-batch quadratic model is less reliable, LM kicks in and the trajectory oscillates around the plateau rather than settling below it. At epsilon=1 (exp-002, complicated further by batch reuse) the step is large enough that the model is wrong often enough to push the run upward overall. So among three runs only exp-003 has touched a probe_loss below 2.30, and even it cannot hold the descent: it visits 2.28 then climbs back to 2.35 over the last three steps. The picture after three runs is that the success criterion of probe_loss < 2.20 in 15 steps is probably not reachable with any single fixed (epsilon, lr) pair on this batch-noise regime, because the regime where steps are large enough to descend is also the regime where single-batch Hessian noise prevents the descent from sticking. Progress over the last five steps is on the order of 0.0016 per step downward at best, which would need roughly 100 more steps to reach 2.20 at this rate, so the budget is the real constraint and either the per-step quality or the batch averaging will have to change to fit inside 15 steps.

---

```yaml
id: exp-004-large-batch-moderate-eps
run_name: exp-004-large-batch-moderate-eps
commit_hash: 147f050ae787a492b8e0a05aeba46e4a55ed545e
probe_loss_initial: null
probe_loss_final: null
probe_loss_min: null
descent_slope: null
rejection_rate: null
rank_among_prior_runs: null  # failed, not ranked
hypothesis_supported: failed-to-run
failure_reason: |
  OOM at step 0 — batch_size=256 caused ~10GB RSS during hessian_inverse_product and SIGKILL.
  batch_size <= 128 is the practical ceiling on this machine.
```

This failure invalidates the planned batch-noise isolation test, because exp-004 was designed to attack batch-to-batch Hessian/gradient mismatch from the large-batch-averaging side as a complement to exp-002's batch-reuse approach, and we now have no data point at batch_size=256 to compare against exp-003. The question "is the step-to-step bumpiness driven by batch noise?" therefore remains open on the large-batch axis, although the batch-reuse direction has already been explored: exp-002 showed that freezing the batch for three consecutive steps did not produce visible within-window descent and instead made the trajectory worse than exp-001, which is evidence against batch noise being the dominant problem. The aggregate picture across exp-001 through exp-003 still points to damping level rather than batch noise as the limiting factor. Future plans should constrain `batch-size <= 128` on this machine, and probably `<= 64` for fast iteration, because batch_size=128 was previously observed to push step time to roughly 22s and batch_size=256 is not runnable at all.

---

```yaml
id: exp-005-small-lr-damped-newton
run_name: exp-005-small-lr-damped-newton
commit_hash: f453954ed2221b25c1d798d90f58001e5a865b62
probe_loss_initial: 2.7178
probe_loss_final: 2.3302
probe_loss_min: 2.3302
descent_slope: -0.06088
rejection_rate: 0.1333
rank_among_prior_runs: 1/4
hypothesis_supported: yes
```

Holding epsilon=1.0 fixed and shrinking lr to 0.1 produced the best final probe_loss of any run so far, 2.3302, which beats exp-003's previous best of 2.3525 and is also the run's own minimum, so unlike exp-001 and exp-003 the trajectory ended on its low point rather than drifting back up. The last-five-step slope of about -0.061 per step is roughly forty times steeper than exp-003's -0.0016 and is the first run where the trajectory is still meaningfully descending at step 14 rather than flattening into the 2.30 plateau, which is exactly what the small-lr-damped-Newton hypothesis predicted. The comparison against exp-002 is the cleanest piece of evidence about what was wrong before: exp-002 used the same epsilon=1.0 with the default lr and ended at 2.4906 with a +0.044 ascending slope, while exp-005 with the same epsilon=1.0 but lr=0.1 ended at 2.3302 with a -0.061 descending slope, so at this damping level the natural Newton step is simply too large and lr=0.1 brings it back inside the trust region of the per-batch quadratic. The comparison against exp-003 says something different but consistent: exp-003 got most of the way there with epsilon=10 and lr=1.0, reaching min 2.2846 before LM rejections at steps 6 and 13 knocked it back to 2.3525, whereas exp-005 reaches a similar final region via a smoother path, with the same 0.1333 rejection rate but with rejections clustered early (steps 5 and 11) rather than at the end. Taken together, exp-002 and exp-005 isolate global step size as a real lever on its own and not just a proxy for damping, because cutting lr by 10x at fixed epsilon moved the run from rank 4/4 to rank 1/4; meanwhile exp-001 and exp-003 already showed that damping level is also a real lever in its own right. The bottleneck is therefore not solely one or the other: both global step size and damping control how often the per-batch quadratic model is trustworthy, and exp-005 shows that even at low damping you can recover good behavior by paying for it with a smaller global step.

