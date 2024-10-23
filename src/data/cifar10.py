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


def getCIFAR10Dataset(data_path='./data', **args):
    """
    image: 32, 32
    """
    log(' ---------------- Get Dataset: CIFAR10 --------------- ')
    ## image size: [3, 32, 32]
    # mean = (0.4914, 0.4822, 0.4465)
    # std = (0.2470, 0.2435, 0.2616)
    ## image size: [3, 128, 128]
    mean = (0.4919, 0.4827, 0.4471)
    std = (0.2409, 0.2375, 0.2562)

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

    data_split = True if split=='train' else False
    dataset = datasets.CIFAR10(os.path.join(data_path, 'cifar-10'), download=True, train=data_split, transform=transform)
    log('{:>35}: {:<30}'.format('dataset length', str(len(dataset))))
    log('{:>35}: {:<30}'.format('image size', str(dataset[0][0].shape)))

    if 'known_classes' in args:
        known_classes = sorted(args['known_classes'])
        known_mapping = {val:idx for idx, val in enumerate(known_classes)}
        unknown_classes = list(set(range(10)) - set(known_classes))
        unknown_classes = sorted(unknown_classes)
        unknown_mapping = {val: idx+len(known_classes) for idx, val in enumerate(unknown_classes)}

        classes = known_classes if split=='train' or split=='in_test' else unknown_classes
        mapping = known_mapping if split=='train' or split=='in_test' else unknown_mapping

        dataset.targets = np.array(dataset.targets)
        dataset.classes = classes

        idx = None
        for i in classes:
            if idx is None:
                idx = (dataset.targets==i)
            else:
                idx |= (dataset.targets==i)
        # dataset.targets = dataset.targets[idx][:2048]  # TODO
        # dataset.data = dataset.data[idx][:2048]  # TODO
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]
        for idx, val in enumerate(dataset.targets):
            dataset.targets[idx] = torch.tensor(mapping[val.item()])

        log('{:>35}: {:<30}'.format('include classes', str(classes)))
    else:
        log('{:>35}: {:<30}'.format('include classes', str(range(10))))

    log(' -------------------- End ------------------- ')
    return dataset
