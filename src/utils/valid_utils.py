import torch
import numpy as np
import pytorch_lightning as pl
import wandb
from datetime import timedelta


def normalize_scores(scores, reverse=False):
    # Normalize the scores
    normalized_scores = (scores - np.min(scores, axis=0)) / (np.max(scores, axis=0) - np.min(scores, axis=0))
    # Reverse the normalized scores if reverse is True
    if reverse:
        normalized_scores = 1 - normalized_scores
    return normalized_scores


def euclidean_distance(x, anchor_feature):
    '''
    Compute euclidean distance between two tensors
    Args:
    - x: torch.tensor of shape (N, D)
    - support_mean: torch.tensor of shape (M, D)

    Returns:
    - dist: torch.tensor of shape (N, M) containing the euclidean distances
    '''
    # Ensure input shapes are correct
    assert x.size(1) == anchor_feature.size(1), "The dimensions of input tensors do not match"

    n = x.size(0)
    m = anchor_feature.size(0)
    d = x.size(1)

    # Compute euclidean distance
    x_expand = x.unsqueeze(1).expand(n, m, d)
    support_mean_expand = anchor_feature.unsqueeze(0).expand(n, m, d)
    dist = torch.sum(torch.pow(x_expand - support_mean_expand, 2), dim=2)

    return dist

