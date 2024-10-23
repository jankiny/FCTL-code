import torch
from tqdm import tqdm

from src.data.mnist import *
from src.data.svhn import *
from src.data.cifar10 import *
from src.data.cifar100 import *
from src.data.tinyimagenet import *
from src.data.cub import *
# from src.data.aircraft import *
from src.data.fgvc_aircraft import *
# from src.data.fgvc_cub import *
from src.data.imagenetcrop import *
from src.data.imagenetresize import *
from src.data.lsuncrop import *
from src.data.lsunresize import *


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    nb = len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=nb, bar_format='{l_bar}{bar:10}{r_bar}',
                desc='Calc (mean_and_std)')
    for batch_idx, (data) in pbar:
        data = data[0]
        # print(data.shape)
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # import pickle
    #
    # with open("src/aircraft_osr_splits.pkl", 'rb') as f:
    #     splits = pickle.load(f)
    #     print(len(splits['known_classes']))

    # dataset = getMNISTDataset(image_size=28, split='train')  # , known_classes=splits['known_classes'])
    # dataset = getSVHNDataset(image_size=32, split='train')  # , known_classes=splits['known_classes'])
    # dataset = getCIFAR10Dataset(image_size=32, split='train')  # , known_classes=splits['known_classes'])
    # dataset = getTinyImageNetDataset(image_size=64, split='train')  # , known_classes=splits['known_classes'])
    # dataset = getCUBDataset(image_size=448, split='train')  # , known_classes=splits['known_classes'])
    dataset = getFGVCDataset(image_size=448, split='train')  # , known_classes=splits['known_classes'])
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    mean, std = get_mean_and_std(loader)
    print('mean:', mean)
    print('std:', std)
    # data = next(iter(loader))
    # print(data)
    # print(data.shape)
    # print(data[0][2][14][14])