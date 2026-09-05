#!/usr/bin/env bash
set -euo pipefail
uv run python src/train_newton.py --mode sgd --lr 0.1 --batch-size 64 --num-steps 30 --logdir runs/auto --run-name exp-021-newton-anchor-sgd-control --log-every 1 --num-layers 8 --hidden-dim 24 --image-size 16 --activation relu
