#!/bin/bash

#data splits are followed from:
#https://github.com/iCGY96/ARPL/blob/master/split.py#L9
#or you can found in:
#https://github.com/sgvaze/osr_closed_set_all_you_need/blob/main/data/open_set_splits/osr_splits.py#L9
splits=(\
"[5, 3, 7, 2, 8, 6]" \
"[3, 8, 7, 6, 2, 5]" \
"[8, 9, 4, 7, 2, 1]" \
"[3, 8, 2, 5, 0, 6]" \
"[4, 9, 2, 7, 1, 0]" \
)

export CUDA_VISIBLE_DEVICES=3
n_gpu=1
exp_type='common0521_1'
stage1_ckpt='runs/report/common0521_1_cnnr101_SVHN'
#stage1_ckpt='runs/report/common0528_1_vitb16_SVHN'
arch='cnnr101' # choices=['vitb16', 'swin', 'cnnr101', 'vgg32']
dataset='SVHN'
known_classes=6
image_size=128
seed=2023

function getCheckpoint() {
  if [ "$1" -eq 1 ]; then
    if [ "$arch" == "vitb16" ]; then
      echo './pretrained_model/imagenet21k+imagenet2012_ViT-B_16.pth'
    elif [ "$arch" == "swin" ]; then
      echo './pretrained_model/swin_base_patch4_window7_224_22k.pth'
    elif [ "$arch" == "cnnr101" ]; then
      echo './pretrained_model/resnet101-IMAGENET1K_V2.pth'
    else
      echo ''
    fi
  else
    last_stage=$(( $1 - 1 ))
    split=$(($2 + 1))
    if [ -n "$stage1_ckpt" ]; then
      echo "${stage1_ckpt}_split${split}_${last_stage}_exp/ckpts/best.ckpt"
    else
      echo "runs/train/${exp_type}_${arch}_${dataset}_split${split}_${last_stage}_exp/ckpts/best.ckpt"
    fi
  fi
}

for i in 0
do
#  python train_fctl_1.py \
#    --wandb \
#    --exp-name ${exp_type}_${arch}_${dataset}_split$((i+1))_1 \
#    --cfg "configs/${dataset}/${arch}_${exp_type}.yaml" \
#    --random-seed ${seed} \
#    --n-gpu ${n_gpu} \
#    --num-workers-pgpu 10 \
#    --model-arch ${arch} \
#    --image-size ${image_size} \
#    --dataset ${dataset} \
#    --num-classes ${known_classes} \
#    --known-classes "${splits[$i]}" \
#    --checkpoint-path "$(getCheckpoint 1 $i)"

  python train_fctl_2.py \
    --wandb 'offline' \
    --exp-name ${exp_type}_${arch}_${dataset}_split$((i+1))_2 \
    --cfg "configs/${dataset}/${arch}_${exp_type}.yaml" \
    --random-seed ${seed} \
    --n-gpu ${n_gpu} \
    --num-workers-pgpu 10 \
    --model-arch ${arch} \
    --dataset ${dataset} \
    --image-size ${image_size} \
    --num-classes ${known_classes} \
    --known-classes "${splits[$i]}" \
    --checkpoint-path "$(getCheckpoint 2 $i)"
done
