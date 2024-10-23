#!/bin/bash

#data splits are copied from:
#https://github.com/iCGY96/ARPL/blob/master/split.py#L44
#or you can found in:
#https://github.com/sgvaze/osr_closed_set_all_you_need/blob/main/data/open_set_splits/osr_splits.py#L49
splits=(\
"[108, 147, 17, 58, 193, 123, 72, 144, 75, 167, 134, 14, 81, 171, 44, 197, 152, 66, 1, 133]" \
"[198, 161, 91, 59, 57, 134, 61, 184, 90, 35, 29, 23, 199, 38, 133, 19, 186, 18, 85, 67]" \
"[177, 0, 119, 26, 78, 80, 191, 46, 134, 92, 31, 152, 27, 60, 114, 50, 51, 133, 162, 93]" \
"[98, 36, 158, 177, 189, 157, 170, 191, 82, 196, 138, 166, 43, 13, 152, 11, 75, 174, 193, 190]" \
"[95, 6, 145, 153, 0, 143, 31, 23, 189, 81, 20, 21, 89, 26, 36, 170, 102, 177, 108, 169]" \
)

export CUDA_VISIBLE_DEVICES=3
n_gpu=1
exp_type='common0523_1'
stage1_ckpt='runs/report/common0523_1_cnnr101_TinyImageNet'
#stage1_ckpt='runs/report/common0526_1_vitb16_TinyImageNet'
# choices=['vitb16', 'swin', 'cnnr101', 'vgg32']
arch='cnnr101'
dataset='TinyImageNet'
known_classes=20
image_size=224
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
#    --wandb 'offline' \
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

