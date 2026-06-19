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
  --run-name exp-012-depth-newton \
  --log-every 5 \
  --hidden-dim 8 \
  --num-layers 16 \
  --image-size 16
