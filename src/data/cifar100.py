from torchvision import datasets, transforms
import torch
import numpy as np
import os
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import pandas as pd
from torchvision.datasets.folder import default_loader

from src.utils import log


def getCIFAR100Dataset(data_path='./data', **args):
    log(' ---------------- Get Dataset: CIFAR100 --------------- ')
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    split = args['split']
    log('{:>35}: {:<30}'.format('split', str(split)))

    data_split = True if split == 'train' else False
    dataset = datasets.CIFAR100(os.path.join(data_path, 'cifar-100'), download=True, train=data_split, transform=transform)
    # dataset = datasets.CIFAR100(data_path, download=True, train=data_split, transform=transform)
    log('{:>35}: {:<30}'.format('dataset length', str(len(dataset))))

    if 'known_classes' in args:
        known_classes = sorted(args['known_classes'])
        known_mapping = {val: idx for idx, val in enumerate(known_classes)}
        unknown_classes = list(set(range(100)) - set(known_classes))
        unknown_classes = sorted(unknown_classes)
        unknown_mapping = {val: idx + len(known_classes) for idx, val in enumerate(unknown_classes)}

        classes = known_classes if split == 'train' or split == 'in_test' else unknown_classes
        mapping = known_mapping if split == 'train' or split == 'in_test' else unknown_mapping
    elif 'include_classes' in args:
        classes = sorted(args['include_classes'])
        mapping = {val: idx for idx, val in enumerate(classes)}
    else:
        raise Exception(f"can not find 'known_classes' or 'include_classes' in {args}")

    dataset.targets = np.array(dataset.targets)

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
