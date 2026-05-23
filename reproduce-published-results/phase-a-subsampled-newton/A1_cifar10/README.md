# A1 — Sub-sampled Newton on full CIFAR-10

Phase 5 anchor (`num_layers=8, hidden_dim=24, image_size=16, activation=relu`),
batch size 64, drawn fresh from the CIFAR-10 train loader each step. Sub-sampled
Newton with `ε = 1.0, lr = 0.5`. Trajectory in [metrics.csv](run/metrics.csv).

See the top-level [README.md](../README.md) for the comparison against SGD's
`probe_loss = 1.97` floor and our linear-Newton's 2.22–2.30 plateau.

The step count is reduced from the spec's 1000 to 100 because each step takes
~25s on the contested CPU and the 1000-step budget would not complete in one
agent turn. See the top-level README for the budget rationale.

## How to reproduce

```
python reproduce-published-results/phase-a-subsampled-newton/subsampled_newton.py \
    --num-steps 100 --batch-size 64 \
    --lr 0.5 --epsilon 1.0 \
    --logdir reproduce-published-results/phase-a-subsampled-newton/A1_cifar10 \
    --run-name run
```
