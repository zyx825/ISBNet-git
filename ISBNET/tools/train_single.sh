#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python3 train.py configs/plant/isbnet_plant_test.yaml --trainall --exp_name default