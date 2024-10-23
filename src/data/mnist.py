from torchvision import datasets, transforms
import torch
import numpy as np
import os
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import pandas as pd
from torchvision.datasets.folder import default_loader

from src.utils import log
from src.data.augmentations import RandAugment


def getMNISTDataset(data_path='./data', **args):
    """
    image: 28, 28
    """
    log(' ---------------- Get Dataset: MNIST --------------- ')
    ## image size: [3, 128, 128]
    # mean = (0.1307, 0.1307, 0.1307)
    # std = (0.3081, 0.3081, 0.3081)
    ## image size: [3, 128, 128]
    mean = (0.1307, 0.1307, 0.1307)
    std = (0.2891, 0.2891, 0.2891)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    log('{:>35}: {:<30}'.format('transform', str(transform)))
    if transform == 'train' and args['transform'] == 'rand-augment':
        transform.transforms.insert(0, RandAugment(1, 9, args=args))

    split = args['split']
    log('{:>35}: {:<30}'.format('split', str(split)))

    data_split = True if split == 'train' else False
    dataset = datasets.MNIST(data_path, download=True, train=data_split, transform=transform)
    log('{:>35}: {:<30}'.format('dataset length', str(len(dataset))))
    log('{:>35}: {:<30}'.format('image size', str(dataset[0][0].shape)))

    if 'known_classes' in args:
        known_classes = sorted(args['known_classes'])
        known_mapping = {val: idx for idx, val in enumerate(known_classes)}
        unknown_classes = list(set(range(10)) - set(known_classes))
        unknown_classes = sorted(unknown_classes)
        unknown_mapping = {val: idx + len(known_classes) for idx, val in enumerate(unknown_classes)}

        classes = known_classes if split == 'train' or split == 'in_test' else unknown_classes
        mapping = known_mapping if split == 'train' or split == 'in_test' else unknown_mapping

        idx = None
        for i in classes:
            if idx is None:
                idx = (dataset.targets == i)
            else:
                idx |= (dataset.targets == i)

        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]
        for idx, val in enumerate(dataset.targets):
            dataset.targets[idx] = torch.tensor(mapping[val.item()])

        log('{:>35}: {:<30}'.format('include classes', str(classes)))

    log(' -------------------- End ------------------- ')

    return dataset

