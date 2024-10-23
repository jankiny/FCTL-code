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
#https://github.com/iCGY96/ARPL/blob/master/split.py#L37
#or you can found in:
#https://github.com/sgvaze/osr_closed_set_all_you_need/blob/main/data/open_set_splits/osr_splits.py#L41
splits_cifar100=(\
"[27, 94, 29, 77, 88, 26, 69, 48, 75, 5, 59, 93, 39, 57, 45, 40, 78, 20, 98, 47, 66, 70, 91, 76, 41, 83, 99, 32, 53, 72, 2, 95, 21, 73, 84, 68, 35, 11, 55, 60, 30, 25, 1, 9, 8, 0, 46, 52, 49, 71]" \
"[65, 97, 86, 24, 45, 67, 2, 3, 91, 98, 79, 29, 62, 82, 33, 76, 0, 35, 5, 16, 54, 11, 99, 52, 85, 1, 25, 66, 28, 84, 23, 56, 75, 46, 21, 72, 55, 68, 8, 69, 41, 9, 49, 40, 73, 60, 48, 30, 95, 71]" \
"[20, 83, 65, 97, 94, 2, 93, 16, 67, 29, 62, 33, 24, 98, 5, 86, 35, 54, 0, 91, 52, 66, 85, 84, 56, 11, 1, 76, 25, 55, 21, 99, 72, 41, 23, 75, 28, 68, 69, 46, 8, 9, 49, 40, 73, 60, 48, 95, 30, 71]" \
"[92, 82, 77, 64, 5, 33, 62, 56, 70, 0, 20, 28, 67, 14, 84, 53, 91, 29, 85, 2, 52, 83, 75, 35, 11, 21, 72, 98, 55, 1, 41, 76, 25, 66, 69, 9, 48, 54, 40, 23, 95, 60, 30, 73, 46, 49, 68, 99, 8, 71]" \
"[47, 6, 19, 0, 62, 93, 59, 65, 54, 70, 34, 55, 23, 38, 72, 76, 53, 31, 78, 96, 77, 27, 92, 18, 82, 50, 98, 32, 1, 75, 83, 4, 51, 35, 80, 11, 74, 66, 36, 42, 33, 2, 3, 97, 46, 21, 64, 63, 88, 43]" \
)

export CUDA_VISIBLE_DEVICES=3
n_gpu=1
exp_type='common0523_2'
stage1_ckpt='runs/report/common0523_2_cnnr101_CIFAR10+50'
#stage1_ckpt='runs/report/common0527_1_vitb16_CIFAR10+50'
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
#    --exp-name ${exp_type}_${arch}_${dataset}+50_split$((i+1))_1 \
#    --cfg "configs/${dataset}+50/${arch}_${exp_type}.yaml" \
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
    --exp-name ${exp_type}_${arch}_${dataset}+50_split$((i+1))_2 \
    --cfg "configs/${dataset}+50/${arch}_${exp_type}.yaml" \
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
