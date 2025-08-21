#!/bin/bash

configs=("metabric" "support" "gbsg" "flchain" "nwtco")
methods=("deepsurv" "deephit" "discrete")
seeds=(42)

for config in "${configs[@]}"; do
  for method in "${methods[@]}"; do
    for seed in "${seeds[@]}"; do
      python3 main.py --debug --config "$config" --method "$method" --seed "$seed"
    done
  done
done
