#!/usr/bin/env bash
set -euo pipefail
cd /Users/arahimi/hessian
uv run python src/train_newton.py \
  --mode newton \
  --epsilon 1.0 \
  --lr 0.1 \
  --lm-up 1.1 \
  --lm-down 0.9 \
  --batch-size 64 \
  --num-steps 15 \
  --logdir runs/auto \
  --run-name exp-005-small-lr-damped-newton \
  --log-every 1
