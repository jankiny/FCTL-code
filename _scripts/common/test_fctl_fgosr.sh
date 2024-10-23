#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
n_gpu=1
exp_type='common'
arch='vitb16' # choices=['vitb16', 'cnnr101']
dataset='FGVC' # choices=['CUB', 'FGVC']


for i in 0
do
  split=$(($i + 1))
  python test_fctl.py \
    --checkpoint-path "runs/report/${exp_type}_${arch}_${dataset}_split${split}_2_exp/ckpts/best.ckpt" \
    --dataset CUB \
    --random-seed 2023
#  python test_fctl.py \
#    --checkpoint-path "runs/train/${exp_type}_${arch}_${dataset}_split${split}_2_exp4/ckpts/best.ckpt" \
#    --dataset ${dataset} \
#    --random-seed 2023
done

