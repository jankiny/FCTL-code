#!/bin/bash

#data splits are followed from:
#https://github.com/iCGY96/ARPL/blob/master/split.py#L23
#or you can found in:
#https://github.com/sgvaze/osr_closed_set_all_you_need/blob/main/data/open_set_splits/osr_splits.py#L25
splits_cifar10=(\
"[4, 7, 9, 1]" \
"[6, 7, 1, 9]" \
"[9, 6, 1, 7]" \
"[6, 4, 9, 1]" \
"[1, 0, 9, 8]" \
)
#data splits are followed from:
#https://github.com/iCGY96/ARPL/blob/master/split.py#L30
#or you can found in:
#https://github.com/sgvaze/osr_closed_set_all_you_need/blob/main/data/open_set_splits/osr_splits.py#L33
splits_cifar100=(\
"[30, 25, 1, 9, 8, 0, 46, 52, 49, 71]" \
"[41, 9, 49, 40, 73, 60, 48, 30, 95, 71]" \
"[8, 9, 49, 40, 73, 60, 48, 95, 30, 71]" \
"[95, 60, 30, 73, 46, 49, 68, 99, 8, 71]" \
"[33, 2, 3, 97, 46, 21, 64, 63, 88, 43]" \
)

export CUDA_VISIBLE_DEVICES=3
n_gpu=1
exp_type='common0522_2'
stage1_ckpt='runs/report/common0522_2_cnnr101_CIFAR10+10'
#stage1_ckpt='runs/report/common0526_3_vitb16_CIFAR10+10'
arch='cnnr101' # choices=['vitb16', 'swin', 'cnnr101', 'vgg32']
dataset='CIFAR10'
known_classes=4
osr_dataset='CIFAR100'
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
#    --exp-name ${exp_type}_${arch}_${dataset}+10_split$((i+1))_1 \
#    --cfg "configs/${dataset}+10/${arch}_${exp_type}.yaml" \
#    --random-seed ${seed} \
#    --n-gpu ${n_gpu} \
#    --num-workers-pgpu 10 \
#    --model-arch ${arch} \
#    --image-size ${image_size} \
#    --dataset ${dataset} \
#    --num-classes ${known_classes} \
#    --known-classes "${splits_cifar10[$i]}" \
#    --osr-dataset ${osr_dataset} \
#    --osr-classes "${splits_cifar100[$i]}" \
#    --checkpoint-path "$(getCheckpoint 1 $i)"


  python train_fctl_2.py \
    --wandb 'offline' \
    --exp-name ${exp_type}_${arch}_${dataset}+10_split$((i+1))_2 \
    --cfg "configs/${dataset}+10/${arch}_${exp_type}.yaml" \
    --random-seed ${seed} \
    --n-gpu ${n_gpu} \
    --num-workers-pgpu 10 \
    --model-arch ${arch} \
    --image-size ${image_size} \
    --dataset ${dataset} \
    --num-classes ${known_classes} \
    --known-classes "${splits_cifar10[$i]}" \
    --osr-dataset ${osr_dataset} \
    --osr-classes "${splits_cifar100[$i]}" \
    --checkpoint-path "$(getCheckpoint 2 $i)"
  done
done
