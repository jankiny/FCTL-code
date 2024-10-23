#!/bin/bash

exp_type='common'
arch='vitb16' # choices=['vitb16', 'cnnr101']
#dataset='MNIST'
#dataset='SVHN'
#dataset='CIFAR10'
#dataset='CIFAR10+10'
#dataset='CIFAR10+50'
dataset='TinyImageNet'
seed=2023


for i in 0 1 2 3 4
do
  wandb artifact put \
  runs/report/common_${arch}_${dataset}_split$((i+1))_2_exp/ckpts/best.ckpt \
  -n "FCTL/${arch}_${dataset}_split$((i+1))" \
  -d "checkpoint of ${arch}_${dataset}_split$((i+1))_2" \
  --type "model" \
  --skip_cache
done

