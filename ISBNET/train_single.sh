#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python3 train.py configs/plant/isbnet_plant_train.yaml --trainall --exp_name default

python3 tools/train.py configs/stpls3d/isbnet_backbone_stpls3d.yaml --only_backbone  --exp_name default