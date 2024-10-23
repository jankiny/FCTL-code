from collections import defaultdict
import numpy as np

import torch
from torch.utils.data.sampler import Sampler


class RandomMetricSampler(Sampler):
    """
    Randomly sample N classes, then for each identity, randomly sample K instances,
    therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instences (int): number of instances per class.
    """

    def __init__(self, data_source, num_instances=4):
        super(RandomMetricSampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_samples * self.num_instances


if __name__ == '__main__':

    from data import getCUBDataset
    import random
    dataset = getCUBDataset(
        image_size=224,
        split='train',
        data_path='./data',
        known_classes=random.sample(range(0, 200), 100)
    )
    sampler = RandomMetricSampler(dataset, num_instances=4)
    a = sampler.__iter__()
