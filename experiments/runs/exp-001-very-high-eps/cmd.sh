#!/usr/bin/env bash
set -euo pipefail
uv run python src/train_newton.py \
  --mode newton --epsilon 100.0 --lr 1.0 --lm-up 1.0 --lm-down 1.0 \
  --batch-size 64 --num-steps 15 \
  --logdir runs/auto --run-name exp-001-very-high-eps --log-every 1
