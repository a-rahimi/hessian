#!/usr/bin/env bash
set -euo pipefail
cd /Users/arahimi/hessian
uv run python src/train_newton.py \
  --mode newton \
  --epsilon 10.0 \
  --lr 1.0 \
  --lm-up 1.0 \
  --lm-down 1.0 \
  --batch-size 64 \
  --num-steps 15 \
  --logdir runs/auto \
  --run-name exp-003-epsilon-sweep-down \
  --log-every 1
