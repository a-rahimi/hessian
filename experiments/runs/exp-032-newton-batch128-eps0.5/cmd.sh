#!/usr/bin/env bash
set -euo pipefail
uv run python src/train_newton.py --mode newton --epsilon 0.5 --lr 0.1 --lm-up 1.0 --lm-down 1.0 --batch-size 128 --num-steps 15 --logdir runs/auto --run-name exp-032-newton-batch128-eps0.5 --log-every 1 --num-layers 8 --hidden-dim 24 --image-size 16 --activation relu
