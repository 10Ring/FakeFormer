#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --cfg configs/spatial/binary_cls/vits/vit_small_112p8.yaml
