#! /bin/bash

CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --cfg configs/temporal/FakeSFormer_small.yaml
