#!/usr/bin/env bash
set -euo pipefail
uv run python src/train_newton.py --mode newton --epsilon 0.01 --lr 1.0 --lm-up 1.0 --lm-down 1.0 --batch-size 64 --num-steps 15 --reuse-batch 15 --logdir runs/auto --run-name exp-029-newton-bug-check --log-every 1 --num-layers 8 --hidden-dim 24 --image-size 16 --activation relu
