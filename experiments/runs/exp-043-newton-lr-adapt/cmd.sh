#!/usr/bin/env bash
set -euo pipefail
uv run python src/train_newton.py --mode newton --epsilon 0.5 --lr 0.1 --lm-up 1.0 --lm-down 1.0 --batch-size 64 --num-steps 60 --lm-check-batch fresh --adapt-lr-on-plateau --lr-decay-window 10 --lr-decay-factor 0.5 --lr-min 0.001 --logdir runs/auto --run-name exp-043-newton-lr-adapt --log-every 1 --num-layers 8 --hidden-dim 24 --image-size 16 --activation relu
