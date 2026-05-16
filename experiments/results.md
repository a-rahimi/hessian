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

---

```yaml
id: exp-006-high-eps-fast-relax
run_name: exp-006-high-eps-fast-relax
commit_hash: 7e875d3b8bdb5324852e2eae506e019fdcc7d652
probe_loss_initial: 2.7178
probe_loss_final: 2.4047
probe_loss_min: 2.3646
descent_slope: 0.000375
rejection_rate: 0.3333
rank_among_prior_runs: 4/5
hypothesis_supported: no
```

Aggressive LM relaxation did not help. Starting at epsilon=100 and halving on every accepted step let the run mimic exp-001 early (the first three accepted steps trace 2.7178, 2.6448, 2.5990 with no rejections, almost identical to exp-001's opening), but once epsilon dropped below about 3 the per-batch quadratic model stopped agreeing with the proposed Newton step and the rejection cascade the agent warned about set in: steps 3, 8, 9, 12, and 14 are all rejections, for a rejection rate of 0.333 that is 2.5x exp-003 and exp-005 and infinitely worse than exp-001's clean zero. The trajectory's minimum of 2.3646 at step 12 sits between exp-005's 2.3302 and exp-001's 2.3800, so the descent phase did briefly find a region as good as the best static-damping runs, but LM could not stabilize there and the final two steps bounced back up to 2.4047, which puts this run at rank 4/5, beating only the broken exp-002 batch-reuse run. The last-five-step slope of essentially zero (+0.0004) is the flattest of any successful run and is the signature of a controller that is oscillating around a fixed point rather than descending toward one. Comparing to exp-001 (frozen eps=100, final 2.3839, zero rejections, slow clean descent) and exp-005 (frozen eps=1, lr=0.1, final 2.3302, still descending at step 14), the takeaway is that the LM schedule itself appears to be the problem: both fixed-damping runs outperform the adaptive-damping run, and exp-005's static eps=1 with small lr beats this run's adaptive descent through the same epsilon range. The data across all five successful runs now says that LM as currently implemented adds little or nothing on this problem, and the practical recipe is either frozen high epsilon with lr=1.0 (exp-001) or frozen low epsilon with small lr (exp-005), with the latter doing better.

---

```yaml
id: exp-007-sgd-baseline-15-steps
run_name: exp-007-sgd-baseline-15-steps
commit_hash: a2f5985e2da0bfaf0210f95a9b8d5d0f0d761a9f
probe_loss_initial: 2.7178
probe_loss_final: 2.3969
probe_loss_min: 2.3631
descent_slope: 0.0034
rejection_rate: null  # N/A for SGD
rank_among_prior_runs: 4/5  # vs four successful Newton runs by probe_loss_final
hypothesis_supported: yes
```

CRITICAL: this SGD control run is the most informative data point in the study so far, because it dissolves the framing under which exp-001 through exp-006 were being interpreted. Plain SGD at the same 15-step budget reaches probe_loss=2.3969 final and min=2.3631, which lands squarely inside the [2.33, 2.49] band that every single Newton variant has been bouncing around in, and at rank 4/5 it beats one Newton run (exp-006, 2.4047) while losing to the other three (exp-005 2.3302, exp-003 2.3525, exp-001 2.3839) by margins of 0.01 to 0.07 that are smaller than the per-step probe wobble within any individual run. The shape of the SGD trajectory is also telling: it spikes to 3.1564 at step 1, recovers to 2.4591 by step 3, then spends the last ten steps oscillating between 2.36 and 2.40 with a last-five-step slope of essentially +0.003, which is the same "asymptote to the plateau and wobble there" pattern we have been attributing to Newton-specific issues (per-batch quadratic model breakdown, LM rejection cascades, damping mistuning). Since SGD has none of those mechanisms and produces the same plateau, the plateau must be a budget-imposed noise floor for this model and probe_batch configuration, not a Newton failure mode. The 2.20 success criterion is almost certainly unreachable at 15 steps for any optimizer on this configuration, so all five Newton hypotheses (epsilon level, batch reuse, lr scaling, LM relaxation, large batch averaging) have been competing for fractional differences inside that noise floor rather than against a real target. The recommendation is therefore explicit and three-pronged: (a) raise num_steps substantially (e.g., to 100 or 200) so that any optimizer has room to actually clear the 2.30 random-guess plateau before being measured, which is the cleanest fix because it preserves comparability with the existing five Newton runs as early-trajectory data; (b) if the 15-step budget is fixed for some external reason, reduce the metric noise floor by increasing probe_batch_size to drive down probe variance and/or by re-scaling the init so probe_loss_initial sits closer to 2.30, which would make the available descent range physically resolvable; (c) failing both of those, accept that within 15 steps Newton and SGD are statistically indistinguishable on this configuration and stop running variants inside this budget, because the next several Newton tweaks will produce more results in the 2.33-2.49 band and tell us nothing we do not already know. The honest one-line conclusion across exp-001 through exp-007 is that the study has been measuring optimizer noise floor rather than optimizer quality, and continuing to vary epsilon, lr, and LM schedules at this budget will not change that.

---

```yaml
id: exp-009-width-sgd-baseline
run_name: exp-009-width-sgd-baseline
commit_hash: ef0cf50e49d38144eff41b606a4b982d2d41e69b
probe_loss_initial: 2.8510
probe_loss_final: 2.2839
probe_loss_min: 2.2474
probe_accuracy_final: 0.121
probe_accuracy_max: 0.152
descent_slope: -0.01230
rejection_rate: null  # N/A for SGD
rank_among_prior_runs: 1/6  # vs successful exp-001..exp-007 by probe_loss_final
hypothesis_supported: yes
```

At hidden_dim=16 (double Phase 1's width) and num_steps=60 (4x Phase 1's budget), plain SGD does clear the 2.30 random-guess plateau, ending at probe_loss=2.2839 with a minimum of 2.2474 at step 50 and a probe_accuracy that climbs from 0.121 at init to a peak of 0.152, so the configuration is no longer pinned to the chance-level noise floor that bracketed exp-001 through exp-007 in [2.33, 2.49]. The descent is also not just a one-shot drop into the plateau and a flat tail, because the last-five logged points slope -0.0123 per logged step (roughly -0.0025 per actual step) while still wobbling by 0.04 step-to-step, which says the trajectory is making real but noisy progress at step 55 rather than asymptoting. This matters for exp-010 in two ways. First, it confirms the Phase 2 framing was right: the previous study had been measuring optimizer noise floor, and the wider-and-longer regime gives Newton a real target to beat instead of a coin-flip band to wobble around in. Second, it sets a concrete and non-trivial Phase 2 success bar: Newton at the same (hidden_dim=16, num_steps=60) configuration must reach probe_loss <= 2.2339 (=2.2839 - 0.05) to beat SGD by the required >=0.05 margin, and probe_loss <= 2.1974 to beat SGD's minimum by that same margin, which is meaningfully below anything any optimizer has shown in this study so far. If exp-010 lands inside the [2.23, 2.29] band, the right read is "Newton matches SGD at this scale" rather than a win, and the case for the Newton machinery being worth its per-step cost would need a larger model or a different metric to be made.

---

```yaml
id: exp-010-width-newton
run_name: exp-010-width-newton
commit_hash: 9c44725a0b8c6ab11782bf16671dd610c6209f59
truncated: true
steps_completed: 45
probe_loss_initial: 2.8510
probe_loss_final: 2.5870
probe_loss_min: 2.3442
probe_accuracy_final: 0.125
probe_accuracy_max: 0.141
descent_slope: 0.0465  # last 4 logged points (steps 30, 35, 40, 45), positive = ascending
rejection_rate: 0.2  # 2 REJ markers across 10 logged step lines
rank_among_prior_runs: 8/8  # worst final probe_loss among all Phase 1 + Phase 2 successful runs
hypothesis_supported: no  # predicted >=0.05 below exp-009 (2.2839); actual is 0.30 ABOVE exp-009
```

At hidden_dim=16 with num_steps=60, Newton run at the best Phase 1 recipe (epsilon_init=1.0, lr=0.1, LM enabled) is substantially worse than SGD at the same scale, ending at probe_loss=2.5870 versus exp-009's 2.2839, a gap of +0.30 in the wrong direction relative to the prediction that Newton should land at least 0.05 below SGD. The trajectory tells a specific failure story rather than a generic noise story. Newton actually reached a low of 2.3442 at step 25, which is within 0.10 of SGD's plateau and would have made a "matches SGD" reading defensible, but it then drifted back up through steps 30 to 45 (2.4141, 2.5345, 2.4808, 2.5870) for a positive last-four-point slope of +0.0465 per logged step, which is the opposite sign from exp-009's -0.0123 over the same regime. The mechanism is visible in the LM trace: epsilon decayed from 1.00 at step 0 to 0.06 at step 45, and once epsilon dropped below about 0.2 the proposed Newton steps blew up in magnitude (|Delta| values of 4.55, 6.84, 9.55 at steps 30, 35, 40, versus 0.12 to 0.85 in the descent phase before step 25), so the optimizer was taking large under-damped steps into regions where the per-batch quadratic model was unreliable. The rejection rate of 0.2 is double exp-001's clean zero and matches exp-002's pattern, so LM caught some of the bad steps but accepted enough of the merely-mediocre ones to push the iterate uphill. The width-axis hypothesis (wider Hessian -> Newton helps more) is rejected at this scale, because Newton is not just failing to beat SGD by 0.05, it is actively finding worse parameters as training progresses while SGD continues to descend.

---

```yaml
id: exp-011-depth-sgd-baseline
run_name: exp-011-depth-sgd-baseline
commit_hash: 03823d792a4ed5c59daac4661a3b31de84f1da17
probe_loss_initial: 2.6529
probe_loss_final: 2.3348
probe_loss_min: 2.2943
probe_accuracy_final: 0.141
probe_accuracy_max: 0.145
descent_slope: 0.01102  # last 4 logged points (steps 40, 45, 50, 55), positive = ascending
rejection_rate: null  # N/A for SGD
rank_among_prior_runs: 3/9  # by probe_loss_final, beating exp-003/001/007/006/002/010, losing to exp-009/005
hypothesis_supported: yes  # predicted "stalls higher 2.20-2.35 due to vanishing gradients"; 2.3348 sits at the high edge of that band, barely supported
```

The depth-axis SGD baseline at depth=16 with tanh activations stalls at probe_loss=2.3348, which sits at the high edge of the predicted 2.20-2.35 band, so the vanishing-gradient hypothesis is technically supported but only barely. The contrast with exp-009 (SGD at width=16) is sharp and exactly what depth-vs-width theory predicts: width=16 SGD reached 2.2839 final and 2.2474 minimum with a clean negative last-five slope of -0.0123 and probe_accuracy climbing to 0.152, while depth=16 SGD at the identical budget reached only 2.3348 final and 2.2943 minimum with a positive last-four slope of +0.011 and probe_accuracy peaking lower at 0.145. The depth run is 0.0509 worse on probe_loss_final and ended ascending rather than descending, which is the signature of a gradient signal that is being attenuated by the cascade of tanh derivatives rather than a budget limit. Two other features of the trace confirm the vanishing-gradient mechanism. First, |g| collapses from 9.05 at step 0 to roughly 0.3-1.2 by step 5 and stays there, an order of magnitude smaller than what exp-009 saw at comparable points, so the optimizer is fundamentally operating on a tiny gradient. Second, the probe trajectory hits its minimum of 2.2943 at step 20 and then oscillates between 2.30 and 2.38 for the next 35 steps without ever recovering that minimum, which is the classic stall pattern for a network whose effective rank of useful gradient directions has collapsed. This is the best opportunity in Phase 2 for Newton to demonstrate value, because Newton's preconditioner is supposed to rescale exactly the small-curvature directions that tanh saturation creates, and exp-010 already failed to beat SGD on the width axis where SGD was healthy. If exp-012 (the paired depth-16 Newton run) can clear probe_loss < 2.2848 (the >=0.05 margin below exp-009's 2.2839, which is the standing Phase 2 success bar set by the best SGD result), it would be the first run in the study to show Newton actually doing what its theory says it should do. Conversely, if exp-012 lands anywhere above 2.28, the Newton machinery has now failed on both axes where it had a sharp predicted advantage, and the case for the method on this class of problems is in serious trouble.

---

```yaml
id: exp-012-depth-newton
run_name: exp-012-depth-newton
commit_hash: 7f0991f88a4189edbcd4ef8ec455d4eee806927d
probe_loss_initial: 2.6529
probe_loss_final: 2.3046
probe_loss_min: 2.2877
probe_accuracy_final: 0.141
probe_accuracy_max: 0.145
descent_slope: -0.00427  # last 4 logged points (steps 40, 45, 50, 55)
rejection_rate: 0.4167  # 5 REJ markers across 12 logged step lines
rank_among_prior_runs: 2/9  # by probe_loss_final, beating exp-005/011/003/001/007/006/002/010, losing only to exp-009
hypothesis_supported: partial  # predicted Newton beats SGD (exp-011) by >=0.05; actual gap is 2.3348 - 2.3046 = 0.0302, directionally right but below threshold
```

This is the first run in the entire study where Newton beats SGD on probe_loss at the same model and horizon, because exp-012 ended at 2.3046 while its paired exp-011 (SGD, depth=16, same 60-step budget) ended at 2.3348, a gap of 0.0302 in Newton's favor. The gap is real but does not clear the >=0.05 success threshold that Phase 2 set, so the Phase 2 hypothesis is supported PARTIALLY rather than fully, because directionally Newton did the right thing on the axis where it was theoretically supposed to (tanh-saturated depth network with small gradients that need preconditioning) but quantitatively it fell about 40% short of the required margin. The comparison against the width axis is the most informative part of the result. On the width axis, exp-010 (Newton width=16) ended at 2.5870 versus exp-009 (SGD width=16) at 2.2839, so Newton was 0.3031 WORSE than SGD. On the depth axis, exp-012 is 0.0302 BETTER than exp-011. That is a swing of roughly 0.33 in Newton's relative performance between the two axes, which is much larger than the 0.05 success threshold itself and points to a clean qualitative finding: Newton's preconditioning helps when the problem actually has the small-curvature directions Newton is supposed to fix (depth-induced saturation), and hurts when the problem is well-conditioned enough that SGD's noisy descent already works (wider but shallow networks). Three other features of the exp-012 trace support this reading. First, |g| does not collapse the way it did in exp-011, staying in the 0.2-3.4 range across the run rather than the 0.3-1.2 stall band, which is consistent with the preconditioner rescaling the saturated directions back up. Second, the trajectory hits its minimum 2.2877 at step 45 and only drifts 0.017 above that by step 55, versus exp-011 which lost 0.04 between its min at step 20 and its final, so Newton holds its descent better on this axis. Third, the rejection rate of 0.4167 (5 of 12 logged lines marked REJ) is the highest of any successful Newton run in the study, yet the run still produced the best Newton final probe_loss outside Phase 1's exp-005; LM is doing real work to keep the run on the rails as epsilon decays from 1.00 to 0.0185, and the rejections cluster precisely at the moments where the Newton step would have walked into bad regions. The recommendation for Phase 3 is therefore explicit. The Planner should queue paired SGD/Newton runs at deeper configurations (num_layers=24 and num_layers=32, hidden_dim=8 held fixed) to test whether the small directional win at depth=16 widens into a >=0.05 win as the vanishing-gradient regime intensifies, because the theory predicts the preconditioning advantage should grow monotonically with depth. The Planner should also queue at least one longer-horizon run at depth=16 (num_steps=120 and num_steps=200) for both optimizers, because exp-012's trajectory was still descending at step 55 (-0.00427 last-four slope) while exp-011 was ascending (+0.011), so a longer budget likely widens the gap purely from the slope differential, not from any new dynamics. Width-axis Newton experiments should be deprioritized until the depth-axis evidence is conclusive, because the existing exp-009/exp-010 pair already gives a clean negative read on that direction.
