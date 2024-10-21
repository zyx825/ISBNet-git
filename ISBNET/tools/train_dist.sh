#!/usr/bin/env bash
GPUS=$1

OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/plant/isbnet_plant_test.yaml --trainall --exp_name default