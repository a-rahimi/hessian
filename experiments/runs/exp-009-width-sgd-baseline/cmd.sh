#!/usr/bin/env bash
set -euo pipefail
cd /Users/arahimi/hessian
uv run python src/train_newton.py \
  --mode sgd \
  --lr 0.1 \
  --batch-size 64 \
  --num-steps 60 \
  --logdir runs/auto \
  --run-name exp-009-width-sgd-baseline \
  --log-every 5 \
  --hidden-dim 16 \
  --num-layers 8 \
  --image-size 16 \
  --activation tanh
