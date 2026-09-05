#!/usr/bin/env bash
set -euo pipefail
uv run python src/train_newton.py --mode newton --epsilon 0.5 --lr 0.1 --lm-up 1.1 --lm-down 0.9 --batch-size 64 --num-steps 60 --lm-check-batch fresh --lr-lm-on-accept 1.05 --lr-lm-on-reject 0.7 --lr-min 0.001 --logdir runs/auto --run-name exp-051-eps-lr-coupled-lm --log-every 1 --num-layers 8 --hidden-dim 24 --image-size 16 --activation relu
