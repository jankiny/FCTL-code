import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """
    Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.Tensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist = dist.addmm(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        # dist_ap, dist_an
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss


class BCEContrastiveLoss(nn.Module):
    def __init__(self):
        super(BCEContrastiveLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # inputs: [batch_size] - scores
        # targets: [batch_size]

        # Ensure have the same size
        if inputs.dim() > 1:
            inputs = inputs.squeeze()

        # Check if targets are float, and convert if not
        if not torch.is_floating_point(targets):
            targets = targets.float()

        # Calculate Binary Cross-Entropy Loss
        if inputs.size() != targets.size():
            inputs = inputs.view(-1)
            targets = targets.view(-1)

        loss = self.bce_loss(inputs, targets)

        return loss

if __name__ == '__main__':
    t = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,]
    t = torch.Tensor(t)
    i = torch.ones(32, 2048)
    a = TripletLoss()
    print(a(i, t))
