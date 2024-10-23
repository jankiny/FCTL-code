#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
n_gpu=1
exp_type='common'
arch='vitb16' # choices=['vitb16', 'cnnr101']
dataset='MNIST'
#dataset='SVHN'
#dataset='CIFAR10'
#dataset='CIFAR10+10'
#dataset='CIFAR10+50'
#dataset='TinyImageNet'
seed=2023


for i in 0 1 2 3 4
do
  split=$(($i + 1))
  python test_fctl.py \
    --exp-name ${exp_type}_${arch}_${dataset}_split$((i+1))_1 \
    --checkpoint-path "runs/report/${exp_type}_${arch}_${dataset}_split${split}_2_exp/ckpts/best.ckpt" \
    --random-seed 2023
#  python test_fctl.py \
#    --exp-name ${exp_type}_${arch}_${dataset}_split$((i+1))_1 \
#    --checkpoint-path "runs/train/${exp_type}_${arch}_${dataset}_split${split}_2_exp/ckpts/best.ckpt" \
#    --random-seed 2023
done

