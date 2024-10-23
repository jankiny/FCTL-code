from torchvision import datasets, transforms
import torch
import numpy as np
import os
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import pandas as pd
from torchvision.datasets.folder import default_loader

from src.utils import log
from src.data.augmentations import get_transform


class CustomCub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader, download=True):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        # self.data = data.merge(train_test_split, on='img_id')[:2048]  # TODO

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            log('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target  # , self.uq_idxs[idx]


def subsample_dataset_cub(dataset, idxs):
    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes_cub(dataset, include_classes=list(range(160)), target_xform_dict=None):
    # if is_osr:
    #     # include classes作为已知类，剩余作为未知类-1
    #     unknown_classes = list(set(range(200)) - set(include_classes))
    #     for i, k in enumerate(unknown_classes):
    #         target_xform_dict[k] = -1
    #     include_classes.append(-1)  # return include_classes
    # else:
    include_classes_cub = np.array(include_classes) + 1  # CUB classes are indexed 1 --> 200 instead of 0 --> 199
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]
    dataset = subsample_dataset_cub(dataset, cls_idxs)

    if target_xform_dict is not None:
        dataset.target_transform = lambda x: target_xform_dict[x]  # 如果不进行映射，会返回原始的target
    dataset.include_classes = include_classes
    return dataset


def getCUBDataset(data_path='./data', **args):
    log(' ---------------- Get Dataset: CUB --------------- ')
    ## image size: [3, 448, 448]
    mean = (0.4854, 0.4993, 0.4319)
    std = (0.2294, 0.2249, 0.2635)

    train_transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        # transforms.RandomCrop(args['image_size'], padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    split = args['split']
    log('{:>35}: {:<30}'.format('split', str(split)))

    data_split = True if split == 'train' else False
    dataset = CustomCub2011(
        root=os.path.join(data_path, 'CUB'),
        train=data_split,
        transform=train_transform if data_split else test_transform,
    )
    log('{:>35}: {:<30}'.format('dataset length', str(len(dataset))))
    log('{:>35}: {:<30}'.format('image size', str(dataset[0][0].shape)))

    if 'known_classes' in args:
        known_classes = sorted(args['known_classes'])
        known_mapping = {val: idx for idx, val in enumerate(known_classes)}
        # unknown_classes = list(set(range(1, 200+1)) - set(known_classes))
        unknown_classes = list(set(range(200)) - set(known_classes))
        unknown_classes = sorted(unknown_classes)

        target_xform_dict = {}
        for i, k in enumerate(known_classes):  # include classes映射到0-99，用于训练
            target_xform_dict[k] = i
        l = len(known_classes)
        for i, k in enumerate(unknown_classes):  # 剩余的映射到100-199，用于测试
            target_xform_dict[k] = i + l

        if split == 'train' or split == 'in_test':
            dataset = subsample_classes_cub(dataset, include_classes=known_classes, target_xform_dict=target_xform_dict)
        # elif split == 'osr_test':
        #     dataset = subsample_classes_cub(dataset, include_classes=known_classes)
        elif split == 'Easy' or split == 'Medium' or split == 'Hard':
            # src/reproduce/osr_closed_set_all_you_need/data/open_set_splits/read_pkl.ipynb
            pkl_split = {
                'Easy': [20, 159, 173, 148, 1, 57, 113, 165, 52, 109, 14, 4, 180, 6, 182, 68, 33, 108, 46, 35, 75, 188,
                         187, 100, 47, 105, 41, 86, 16, 54, 139, 138],
                'Medium': [152, 195, 132, 83, 22, 192, 153, 175, 191, 155, 49, 194, 73, 66, 170, 151, 169, 96, 103, 37,
                           181, 127, 78, 21, 10, 164, 62, 2, 183, 85, 45, 60, 92, 185],
                'Hard': [29, 110, 3, 8, 13, 58, 142, 25, 145, 63, 59, 65, 24, 140, 120, 32, 114, 107, 160, 130, 118,
                         101, 115, 128, 117, 71, 156, 112, 36, 122, 104, 102, 90, 125]
            }
            dataset = subsample_classes_cub(dataset, include_classes=pkl_split[split],
                                            target_xform_dict=target_xform_dict)
        else:
            dataset = subsample_classes_cub(dataset, include_classes=unknown_classes,
                                            target_xform_dict=target_xform_dict)
        log('{:>35}: \n{:<30}'.format('include classes', str(dataset.include_classes)))

    log(' -------------------- End ------------------- ')
    return dataset
