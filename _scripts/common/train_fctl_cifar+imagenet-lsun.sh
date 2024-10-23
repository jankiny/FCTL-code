#!/bin/bash
# --known-classes "all_10"，使用‘all’加类别数量n，在utils/fileio.py#140被解析为从0开始的list

export CUDA_VISIBLE_DEVICES=3
n_gpu=1
exp_type='common0610_3'
# choices=['vitb16', 'swin', 'cnnr101']
arch='cnnr101'
dataset='CIFAR10'
known_classes=10
osr_dataset='ImageNetCrop'
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
    echo "runs/train/${exp_type}_${arch}_${dataset}+ood_split${split}_${last_stage}_exp/ckpts/best.ckpt"
  fi
}

for i in 0
do
  python train_fctl_1.py \
    --wandb \
    --exp-name ${exp_type}_${arch}_${dataset}+ood_split$((i+1))_1 \
    --cfg "configs/${dataset}+ood/${arch}_${exp_type}.yaml" \
    --random-seed ${seed} \
    --n-gpu ${n_gpu} \
    --num-workers-pgpu 10 \
    --model-arch ${arch} \
    --image-size ${image_size} \
    --dataset ${dataset} \
    --known-classes "all_10" \
    --num-classes ${known_classes} \
    --osr-dataset ${osr_dataset} \
    --osr-classes "all_200" \
    --checkpoint-path "$(getCheckpoint 1 $i)"


  python train_fctl_2.py \
    --wandb \
    --exp-name ${exp_type}_${arch}_${dataset}+ood_split$((i+1))_2 \
    --cfg "configs/${dataset}+ood/${arch}_${exp_type}.yaml" \
    --random-seed ${seed} \
    --n-gpu ${n_gpu} \
    --num-workers-pgpu 10 \
    --model-arch ${arch} \
    --image-size ${image_size} \
    --dataset ${dataset} \
    --known-classes "all_10" \
    --num-classes ${known_classes} \
    --osr-dataset ${osr_dataset} \
    --osr-classes "all_200" \
    --checkpoint-path "$(getCheckpoint 2 $i)"
  done
done
