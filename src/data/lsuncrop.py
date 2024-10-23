from torchvision import datasets, transforms
import torch
import numpy as np
import os
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import pandas as pd
from torchvision.datasets.folder import default_loader

from src.utils import log


class LSUN_Crop_Filter(datasets.ImageFolder):
    """LSUN_Crop Dataset.
    """

    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                # new_targets.append(targets[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets
        self.include_classes = known


def getLSUNCropDataset(data_path='./data', **args):
    """
    image: 32, 32
    """
    log(' ---------------- Get Dataset: LSUN-Crop --------------- ')
    ## image size: [3, 64, 64]
    # mean = (0.4802, 0.4481, 0.3975)
    # std = (0.2764, 0.2689, 0.2816)
    ## image size: [3, 128, 128]
    mean = (0.4805, 0.4483, 0.3978)
    std = (0.2662, 0.2582, 0.2718)

    transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ])

    split = args['split']
    log('{:>35}: {:<30}'.format('split', str(split)))

    data_split = 'train' if split == 'train' else 'test'
    dataset = LSUN_Crop_Filter(os.path.join(data_path, 'LSUN_crop', data_split), transform)
    log('{:>35}: {:<30}'.format('dataset length', str(len(dataset))))
    log('{:>35}: {:<30}'.format('image size', str(dataset[0][0].shape)))

    if 'known_classes' in args:
        known_classes = sorted(args['known_classes'])
        known_mapping = {val: idx for idx, val in enumerate(known_classes)}
        unknown_classes = list(set(range(200)) - set(known_classes))
        unknown_classes = sorted(unknown_classes)
        unknown_mapping = {val: idx + len(known_classes) for idx, val in enumerate(unknown_classes)}

        if split == 'in_test' or split == 'train':
            dataset.__Filter__(known=known_classes)
        else:
            dataset.__Filter__(known=unknown_classes)

        log('{:>35}: {:<30}'.format('include classes', str(dataset.include_classes)))

    else:
        dataset.__Filter__(known=[i for i in range(200)])
        log('{:>35}: {:<30}'.format('include classes', str(range(200))))

    log(' -------------------- End ------------------- ')

    return dataset
