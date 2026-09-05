#!/usr/bin/env bash
set -euo pipefail
uv run python src/train_newton.py --mode newton --epsilon 0.5 --lr 0.05 --lm-up 1.0 --lm-down 1.0 --batch-size 64 --num-steps 15 --logdir runs/auto --run-name exp-038-newton-lr0.05 --log-every 1 --num-layers 8 --hidden-dim 24 --image-size 16 --activation relu
