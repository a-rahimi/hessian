#!/usr/bin/env bash
set -euo pipefail
cd /Users/arahimi/hessian
uv run python src/train_newton.py \
  --mode sgd \
  --lr 0.1 \
  --batch-size 64 \
  --num-steps 15 \
  --logdir runs/auto \
  --run-name exp-007-sgd-baseline-15-steps \
  --log-every 1 \
  --hidden-dim 8 \
  --num-layers 8 \
  --image-size 16 \
  --activation tanh
