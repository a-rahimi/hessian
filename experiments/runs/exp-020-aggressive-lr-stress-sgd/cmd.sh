#!/usr/bin/env bash
set -euo pipefail
uv run python src/train_newton.py --mode sgd --lr 0.5 --batch-size 64 --num-steps 300 --logdir runs/auto --run-name exp-020-aggressive-lr-stress-sgd --log-every 10 --hidden-dim 8 --num-layers 16 --image-size 16 --activation tanh
