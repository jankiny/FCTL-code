#!/bin/bash

splits=(\
"[0, 1, 2, 3, 4, 5, 10, 11, 14, 16, 17, 19, 21, 22, 23, 24, 27, 28, 29, 30, 33, 36, 37, 38, 39, 41, 43, 44, 45, 46, 47, 48, 52, 53, 56, 57, 58, 63, 64, 65, 66, 67, 71, 73, 76, 77, 79, 92, 95, 99]" \
)

export CUDA_VISIBLE_DEVICES=3
n_gpu=1
exp_type='common0810_2'
# choices=['vitb16', 'swin', 'cnnr101', 'vgg32']
arch='vitb16'
dataset='FGVC' # aircraft
known_classes=50
image_size=448
seed=2023

function getCheckpoint() {
  if [ "$1" -eq 1 ]; then
    if [ "$arch" == "vitb16" ]; then
      echo './pretrained_model/imagenet21k+imagenet2012_ViT-B_16.pth'
#      echo './pretrained_model/vit_b_16_IMAGENET1K_SWAG_E2E_V1.pth'
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
    echo "runs/train/${exp_type}_${arch}_${dataset}_split${split}_${last_stage}_exp/ckpts/best.ckpt"
  fi
}

for i in 0
do
#  python train_fctl_1.py \
#    --wandb \
#    --exp-name ${exp_type}_${arch}_${dataset}_split$((i+1))_1 \
#    --cfg "configs/${dataset}/${arch}_${exp_type}.yaml" \
#    --transform "rand-augment" \
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
    --transform "rand-augment" \
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
