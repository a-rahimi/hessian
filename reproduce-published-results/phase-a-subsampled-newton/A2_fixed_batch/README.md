# A2 — Sub-sampled Newton on the fixed-batch memorization diagnostic

The exp-048b protocol: a single 64-sample batch held for many steps. Sub-sampled
Newton with `ε = 1.0, lr = 0.5`. Trajectory in [metrics.csv](run/metrics.csv).

See the top-level [README.md](../README.md) for the comparison against SGD and
linear-inverse Newton.

The step count is reduced from the spec's 1000 to 100 because each step takes
~25s on the contested CPU and the 1000-step budget would not complete in one
agent turn. See the top-level README for the budget rationale.

## How to reproduce

```
python reproduce-published-results/phase-a-subsampled-newton/subsampled_newton.py \
    --num-steps 100 --batch-size 64 --reuse-batch 100 \
    --lr 0.5 --epsilon 1.0 \
    --logdir reproduce-published-results/phase-a-subsampled-newton/A2_fixed_batch \
    --run-name run
```
