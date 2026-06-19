#!/usr/bin/env bash
set -euo pipefail
uv run python src/train_newton.py --mode newton --epsilon 1.0 --lr 0.3 --lm-up 1.1 --lm-down 0.9 --batch-size 64 --num-steps 30 --logdir runs/auto --run-name exp-024-newton-anchor-high-lr --log-every 1 --num-layers 8 --hidden-dim 24 --image-size 16 --activation relu
