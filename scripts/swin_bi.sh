#! /bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --cfg configs/spatial/swin_bi_small.yaml
