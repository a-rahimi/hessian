#!/usr/bin/env bash
set -euo pipefail
uv run python src/train_newton.py --mode sgd --lr 0.1 --batch-size 64 --num-steps 1000 --reuse-batch 1000 --logdir runs/auto --run-name exp-048-sgd-reuse-batch-baseline --log-every 25 --num-layers 8 --hidden-dim 24 --image-size 16 --activation relu
