#!/bin/bash

# methods=("deepsurv" "deephit" "discrete")
# seeds=(42)

# for method in "${methods[@]}"; do
#   for seed in "${seeds[@]}"; do
#     python3 main.py --debug --config collagen_img --method "$method" --seed "$seed"
#   done
# done

methods=("deepsurv" "deephit" "discrete")
maggs=("mean" "min" "max")

for method in "${methods[@]}"; do
  for magg in "${maggs[@]}"; do
    python3 main.py --debug --config collagen_img --method "$method" --metric_agg "$magg"
  done
done

methods=("deepsurv" "deephit" "discrete")
aggs=("mean" "min" "max" "att")

for method in "${methods[@]}"; do
    for agg in "${aggs[@]}"; do
      python3 main.py --debug --config collagen_pat --method "$method" --aggregator "$agg"
    done
done