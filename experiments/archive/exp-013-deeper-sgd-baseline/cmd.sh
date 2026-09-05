#!/usr/bin/env bash
set -euo pipefail
uv run python src/train_newton.py \
  --mode sgd \
  --lr 0.1 \
  --batch-size 64 \
  --num-steps 60 \
  --logdir runs/auto \
  --run-name exp-013-deeper-sgd-baseline \
  --log-every 5 \
  --hidden-dim 8 \
  --num-layers 24 \
  --image-size 16 \
  --activation tanh
