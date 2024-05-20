#! /bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --cfg configs/spatial/vit_bi_small.yaml
