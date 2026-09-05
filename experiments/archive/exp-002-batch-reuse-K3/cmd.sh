#!/usr/bin/env bash
set -euo pipefail
cd /Users/arahimi/hessian
uv run python src/train_newton.py \
  --mode newton \
  --epsilon 1.0 \
  --lr 0.5 \
  --batch-size 64 \
  --num-steps 15 \
  --logdir runs/auto \
  --run-name exp-002-batch-reuse-K3 \
  --log-every 1 \
  --reuse-batch 3
