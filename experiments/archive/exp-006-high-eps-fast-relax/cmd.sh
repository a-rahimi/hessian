#!/usr/bin/env bash
set -euo pipefail
cd /Users/arahimi/hessian
uv run python src/train_newton.py \
  --mode newton \
  --epsilon 100.0 \
  --lr 1.0 \
  --lm-up 2.0 \
  --lm-down 0.5 \
  --batch-size 64 \
  --num-steps 15 \
  --logdir runs/auto \
  --run-name exp-006-high-eps-fast-relax \
  --log-every 1
