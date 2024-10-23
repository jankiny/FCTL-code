import json
import math
from pathlib import Path
from datetime import datetime
import os
import sys
import logging
import copy
import threading
from multiprocessing.pool import ThreadPool

import torch
import matplotlib
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import importlib
import pandas as pd
import numpy as np


class MetricTracker:
    def __init__(self, *keys):
        # self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self._custom_data = {}
        self.reset()

    def bind_custom_metric(self, tag, data):
        self._custom_data[tag] = data

    def get_custom_metric(self, tag):
        return self._custom_data[tag]

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0
        for key in self._custom_data:
            self._custom_data[key].reset()

    def update(self, key, value, n=1):
        if key not in self._custom_data:
            # if self.writer is not None:
            #     self.writer.add_scalar(key, value)
            self._data.total[key] += value * n
            self._data.counts[key] += n
            self._data.average[key] = self._data.total[key] / self._data.counts[key]
        else:
            pass

    def avg(self, key):
        return self._data.average[key]

    def result(self, custom_data=True):
        result = dict(self._data.average)
        if custom_data:
            result.update({key: self._custom_data[key].result() for key in self._custom_data})
        return result


class OSCR:
    def __init__(self, writer=None):
        self.writer = writer
        self.oscr = None

    def reset(self):
        self.oscr = None

    def update(self, known_score, unknown_score, outputs, targets):
        self.oscr = compute_oscr(known_score, unknown_score, outputs, targets)

    def result(self):
        # return self.roc_auc * 100
        return f'{self.oscr}'

class AUROC:
    def __init__(self, is_plot=False, writer=None):
        self.writer = writer
        self.is_plot = is_plot
        self.fpr = None
        self.tpr = None
        self.roc_auc = None

    def reset(self):
        self.fpr = None
        self.tpr = None
        self.roc_auc = None

    def update(self, outputs, targets):
        self.fpr, self.tpr, thresholds = roc_curve(targets, outputs)
        self.roc_auc = auc(self.fpr, self.tpr)

    def plot(self):
        if not self.is_plot:
            return
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        ax.plot(self.fpr, self.tpr, color='darkorange', lw=2, label=f'ROC curve (area = {self.roc_auc * 100:.2f}%)')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver operating characteristic')
        ax.legend(loc="lower right")
        self.writer.add_figure('roc_auc', fig)

    def result(self):
        # return self.roc_auc * 100
        return f'Plot - {self.is_plot}'


class FeatureEmbedding:
    def __init__(self, is_plot=False, writer=None, save_path=None):
        self.writer = writer
        self.save_path = save_path
        self.is_plot = is_plot
        self.embeddings = None
        self.targets = None

    def reset(self):
        self.embeddings = None
        self.targets = None

    def update(self, embeddings, targets):
        self.embeddings = embeddings
        self.targets = targets

    # delete: plot in jupyter
    # def plot(self):
    #     if not self.is_plot:
    #         return
    #
    #     X_2d = TSNE(n_components=2,
    #                 learning_rate='auto',
    #                 early_exaggeration=12,
    #                 init='pca',
    #                 perplexity=20,
    #                 n_iter=10000
    #                 ).fit_transform(self.embeddings.detach().numpy())
    #
    #     fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    #     # 在Axes上绘制散点图
    #     sc = ax.scatter(x=X_2d[:, 0], y=X_2d[:, 1], c=self.targets, s=5)
    #     fig.colorbar(sc, ax=ax)
    #     ax.set_title('Feature Embedding')
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     # if self.writer is None:
    #     #     fig.savefig(Path(save_dir) / 'feature_embedding.png', dpi=250)
    #     #     plt.close(fig)
    #     # else:
    #     self.writer.add_figure('feature_embedding', fig)

    def save(self, epoch):
        torch.save(self.embeddings, os.path.join(self.save_path, f'epoch{epoch}_embeddings.pt'))
        torch.save(self.targets, os.path.join(self.save_path, f'epoch{epoch}_targets.pt'))

    def result(self):
        return f'Plot - {self.is_plot}'


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, is_plot=False, writer=None, tag=''):
        self.is_plot = is_plot
        self.writer = writer
        self.tag = tag
        self.confusion_matrix = np.zeros((nc, nc))
        self.nc = nc  # number of classes
        # self.conf = conf
        # self.iou_thres = iou_thres

    def reset(self):
        self.confusion_matrix = np.zeros((self.nc, self.nc))

    def add(self, batch_confusion_matrix):
        self.confusion_matrix += batch_confusion_matrix

    @staticmethod
    def calculate(output, target):
        if output.ndimension() == 2:
            output = output.argmax(1)
        return confusion_matrix(target.cpu(), output.cpu())

    def tp_fp(self):
        tp = self.confusion_matrix.diagonal()  # true positives
        fp = self.confusion_matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    def plot(self, normalize=True, save_dir='', names=()):
        if not self.is_plot:
            return

        import seaborn as sn

        array = self.confusion_matrix / (
            (self.confusion_matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = names if labels else 'auto'
        sn.heatmap(array,
                   ax=ax,
                   annot=nc < 30,
                   annot_kws={
                       'size': 8},
                   cmap='Blues',
                   fmt='.2f',
                   square=True,
                   vmin=0.0,
                   xticklabels=ticklabels,
                   yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Confusion Matrix')
        if self.writer is None:
            fig.savefig(Path(save_dir) / f'{self.tag} confusion_matrix.png', dpi=250)
            plt.close(fig)
        else:
            self.writer.add_figure(f'{self.tag} confusion_matrix', fig)

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.confusion_matrix[i])))

    def result(self):
        np.set_printoptions(suppress=True)
        return f'Plot - {self.is_plot}\n' + str(self.confusion_matrix)


def recall_k(actual, predicted, k):
    """Computes the Recall@k for the specified values of k"""
    if len(actual) != len(predicted):
        raise ValueError("actual 和 predicted 列表长度不一致")

    sorted_predictions = sorted(zip(actual, predicted), key=lambda x: x[1], reverse=True)
    top_k_actual = [x[0] for x in sorted_predictions[:k]]

    recall = sum(top_k_actual) / sum(actual)

    return recall


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
#
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
#
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k / batch_size * 100.0)
    return res


def compute_oscr(x1, x2, pred, labels):

    """
    Modified from https://github.com/sgvaze/osr_closed_set_all_you_need/blob/main/test/utils.py#L125
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """
    if pred.dim() == 2:
        pred = pred.argmax(axis=1)

    correct = (pred == labels)
    m_x1 = torch.zeros(len(x1), device=pred.device)  # 确保在相同的设备
    m_x1[pred == labels] = 1
    k_target = torch.cat((m_x1, torch.zeros(len(x2), device=pred.device)), dim=0)
    u_target = torch.cat((torch.zeros(len(x1), device=pred.device), torch.ones(len(x2), device=pred.device)), dim=0)
    predict = torch.cat((x1, x2), dim=0)
    n = len(predict)

    CCR = torch.zeros(n + 2, device=pred.device)
    FPR = torch.zeros(n + 2, device=pred.device)

    idx = predict.argsort()
    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    # 如果len_x2为零，则避免除以零
    len_x2 = float(len(x2))
    if len_x2 == 0:
        len_x2 = 1

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / len_x2

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = torch.stack((FPR, CCR), dim=1)
    ROC, _ = ROC.sort(dim=0, descending=True)

    OSCR = 0.0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j, 0] - ROC[j + 1, 0]
        w = (ROC[j, 1] + ROC[j + 1, 1]) / 2.0

        OSCR += h * w

    return OSCR.item()  # 返回Python数值
