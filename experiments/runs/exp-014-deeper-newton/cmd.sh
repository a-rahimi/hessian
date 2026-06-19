#!/usr/bin/env bash
set -euo pipefail
uv run python src/train_newton.py \
  --mode newton \
  --epsilon 1.0 \
  --lr 0.1 \
  --lm-up 1.1 \
  --lm-down 0.9 \
  --batch-size 64 \
  --num-steps 60 \
  --logdir runs/auto \
  --run-name exp-014-deeper-newton \
  --log-every 5 \
  --hidden-dim 8 \
  --num-layers 24 \
  --image-size 16 \
  --activation tanh
