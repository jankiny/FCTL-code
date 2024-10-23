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


def getSVHNDataset(data_path='./data', **args):
    """
    image: 32, 32
    """
    log(' ---------------- Get Dataset: SVHN --------------- ')
    ## image size: [3, 32, 32]
    # mean = (0.4376, 0.4436, 0.4727)
    # std = (0.1980, 0.2010, 0.1970)
    ## image size: [3, 128, 128]
    mean = (0.4380, 0.4441, 0.4732)
    std = (0.1958, 0.1988, 0.1954)

    transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    log('{:>35}: {:<30}'.format('transform', str(transform)))
    if transform == 'train' and args['transform'] == 'rand-augment':
        transform.transforms.insert(0, RandAugment(1, 9, args=args))

    split = args['split']
    log('{:>35}: {:<30}'.format('split', str(split)))

    data_split = 'train' if split == 'train' else 'test'
    dataset = datasets.SVHN(os.path.join(data_path, 'svhn'), download=True, split=data_split, transform=transform)
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

        dataset.labels = np.array(dataset.labels)

        idx = None
        for i in classes:
            if idx is None:
                idx = (dataset.labels == i)
            else:
                idx |= (dataset.labels == i)

        dataset.labels = dataset.labels[idx]
        dataset.data = dataset.data[idx]
        for idx, val in enumerate(dataset.labels):
            dataset.labels[idx] = torch.tensor(mapping[val.item()])

        log('{:>35}: {:<30}'.format('include classes', str(classes)))

    log(' -------------------- End ------------------- ')

    return dataset

