#!/bin/bash

methods=("deepsurv" "deephit" "discrete")
seeds=(42)

for method in "${methods[@]}"; do
  for seed in "${seeds[@]}"; do
    python3 main.py --debug --config collagen_img --method "$method" --seed "$seed"
  done
done

