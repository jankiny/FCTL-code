#!/bin/bash

#splits=(\
#"150 70 34 178 199 131 129 147 134 11 26 93 95 121 123 99 149 167 18 31 69 198 116 158 126 17 5 179 111 163 184 81 174 42 53 89 77 55 23 48 43 44 56 28 193 143 0 176 84 15 38 154 141 190 172 124 189 19 80 157 12 9 79 30 94 67 197 97 168 137 119 76 98 88 40 106 171 87 166 186 27 51 144 135 161 64 177 7 146 61 50 162 133 82 39 74 72 91 196 136" \
#)
splits=(\
"[150, 70, 34, 178, 199, 131, 129, 147, 134, 11, 26, 93, 95, 121, 123, 99, 149, 167, 18, 31, 69, 198, 116, 158, 126, 17, 5, 179, 111, 163, 184, 81, 174, 42, 53, 89, 77, 55, 23, 48, 43, 44, 56, 28, 193, 143, 0, 176, 84, 15, 38, 154, 141, 190, 172, 124, 189, 19, 80, 157, 12, 9, 79, 30, 94, 67, 197, 97, 168, 137, 119, 76, 98, 88, 40, 106, 171, 87, 166, 186, 27, 51, 144, 135, 161, 64, 177, 7, 146, 61, 50, 162, 133, 82, 39, 74, 72, 91, 196, 136]" \
)
export CUDA_VISIBLE_DEVICES=3
n_gpu=1
exp_type='common0605_1'
#stage1_ckpt='runs/report/common0701_1_cnnr101_CUB'
#stage1_ckpt='runs/report/common0605_1_vitb16_CUB'
arch='cnnr101' # choices=['vitb16', 'swin', 'cnnr101', 'vgg32']
dataset='CUB'
known_classes=100
image_size=448
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
    --wandb \
    --exp-name ${exp_type}_${arch}_${dataset}_split$((i+1))_2 \
    --cfg "configs/${dataset}/${arch}_${exp_type}.yaml" \
    --random-seed ${seed} \
    --n-gpu ${n_gpu} \
    --num-workers-pgpu 10 \
    --model-arch ${arch} \
    --image-size ${image_size} \
    --dataset ${dataset} \
    --num-classes ${known_classes} \
    --known-classes "${splits[$i]}" \
    --checkpoint-path "$(getCheckpoint 2 $i)"
done
