#!/usr/bin/env bash
set -euo pipefail
uv run python src/train_newton.py --mode sgd --lr 0.1 --batch-size 64 --num-steps 500 --logdir runs/auto --run-name exp-018-extreme-deep-long-horizon-sgd --log-every 10 --hidden-dim 8 --num-layers 32 --image-size 16 --activation tanh
