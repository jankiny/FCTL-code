#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
n_gpu=1
exp_type='common'
# choices=['vitb16', 'swin', 'cnnr101']
arch='cnnr101'
dataset='CIFAR10'


for i in 0
do
  split=$(($i + 1))
  python test_fctl_ood.py \
    --checkpoint-path "runs/report/${exp_type}_${arch}_${dataset}+ood_split${split}_2_exp/ckpts/best.ckpt" \
    --random-seed 2023
#  python test_fctl.py \
#    --checkpoint-path "runs/train/${exp_type}_${arch}_${dataset}_split${split}_2_exp/ckpts/best.ckpt" \
#    --random-seed 2023
done

