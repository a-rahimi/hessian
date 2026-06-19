#!/usr/bin/env bash
set -euo pipefail
uv run python src/train_newton.py --mode newton --epsilon 5.0 --lr 0.01 --lm-up 1.0 --lm-down 1.0 --batch-size 64 --num-steps 15 --logdir runs/auto --run-name exp-035-newton-high-eps-low-lr --log-every 1 --num-layers 8 --hidden-dim 24 --image-size 16 --activation relu
