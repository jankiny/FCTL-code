import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from src.model.vit import VisionTransformer, LinearGeneral, MlpBlock, SelfAttention

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Attention(nn.Module):
    """
    Modify from https://github.com/lucidrains/vit-pytorch &
    https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_io.py#L94
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout_rate=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == query_dim)

        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.dropout = nn.Dropout(dropout_rate)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        # self.to_out = nn.Linear(inner_dim, query_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout_rate)
        ) if project_out else nn.Identity()

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) -> (b h) d', h=h), (q, k, v))
        # sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # out = einsum('b i j, b j d -> b i d', attn, v)
        out = torch.matmul(attn, v)
        out = rearrange(out, '(b h) d -> b (h d)', h=h)
        return self.to_out(out)


class FusionBlock(nn.Module):
    def __init__(self, in_dim, context_dim, mlp_dim, num_heads=12, dropout_rate=0.1):
        super(FusionBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = Attention(in_dim, context_dim=context_dim, heads=num_heads, dropout_rate=dropout_rate)  # CrossAttention

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, query, context):
        residual = query
        out = self.norm1(query)
        out = self.attn(out, context)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out


class FusionLayer(nn.Module):
    def __init__(self, emb_dim, mlp_dim, num_heads=12, dropout_rate=0.1, sim_block='attention', num_sim_block=1):
        super(FusionLayer, self).__init__()

        in_dim = emb_dim
        if sim_block == 'attention':
            context_dim = 2 * emb_dim
        else: # sim_block == 'mlp'
            context_dim = 2 * emb_dim

        self.fusion_layers = nn.ModuleList()
        for i in range(num_sim_block):
            layer = FusionBlock(in_dim, context_dim, mlp_dim, num_heads, dropout_rate)
            self.fusion_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, query, context_list):
        out = query
        for layer, context in zip(self.fusion_layers, context_list):
            out = layer(out, context)

        out = self.norm(out)
        return out


class SimilarityMlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim, num_head=12, dropout_rate=0.1):
        super(SimilarityMlpBlock, self).__init__()

        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.norm1 = nn.LayerNorm(mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.norm2(out)
        if self.dropout2:
            out = self.dropout2(out)

        return out


class SimilarityAttentionBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim, num_heads=12, dropout_rate=0.1):
        super(SimilarityAttentionBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = Attention(in_dim, heads=num_heads, dropout_rate=dropout_rate)  # SelfAttention

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, out_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out


class SimilarityLayer(nn.Module):
    def __init__(self, emb_dim, mlp_dim, sim_block='attention', sim_task='cls', num_sim_block=2, num_heads=12, dropout_rate=0.1, attn_dropout_rate=0.0):
        super(SimilarityLayer, self).__init__()

        if sim_block == 'attention':
            SimilarityBlock = SimilarityAttentionBlock
        elif sim_block == 'mlp':
            SimilarityBlock = SimilarityMlpBlock
        else:
            SimilarityBlock = SimilarityMlpBlock
        # encoder blocks
        in_dim = emb_dim
        self.similarity_layers = nn.ModuleList()
        for i in range(num_sim_block):
            layer = SimilarityBlock(in_dim, mlp_dim, in_dim, num_heads, dropout_rate=dropout_rate)
            self.similarity_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

        out_features = 2 if sim_task == 'cls' else 1
        self.sim_classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_features)
        )

    def forward(self, x, feat_cls=False):

        diff_feat_list = []
        out = x
        for layer in self.similarity_layers:
            out = layer(out)
            diff_feat_list.append(out)

        out = self.norm(out)
        if feat_cls:
            return diff_feat_list, self.sim_classifier(out)
        else:
            return self.sim_classifier(out)


# class SimilarityBlock(nn.Module):
#     def __init__(self, in_dim, mlp_dim, num_heads=12, dropout_rate=0.1, attn_dropout_rate=0.1):
#         super(SimilarityBlock, self).__init__()
#         self.fc1 = nn.Sequential(
#             nn.Linear(in_dim * 2, in_dim * 2),
#             nn.BatchNorm1d(in_dim * 2),
#             nn.ReLU(inplace=True))
#
#         self.fc2 = nn.Sequential(
#             nn.Linear(in_dim * 2, in_dim),
#             nn.BatchNorm1d(in_dim),
#             nn.ReLU(inplace=True))
#
#         self.fc3 = nn.Sequential(
#             nn.Linear(in_dim, in_dim),
#             nn.BatchNorm1d(in_dim),
#             nn.ReLU(inplace=True))
#
#         self.fc4 = nn.Linear(in_dim, 2)
#
#         self.diff_dim = in_dim
#
#     def forward(self, x, feat_cls=False):
#         x = self.fc1(x)
#         x = self.fc2(x)
#
#         diff_feat = x
#
#         x = self.fc3(x)
#         x = self.fc4(x)
#
#         if feat_cls:
#             return diff_feat, x
#         else:
#             return x


class OSRClassifier(nn.Module):
    def __init__(self, in_dim=2048, num_classes=1000):
        super(OSRClassifier, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = in_dim
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.num_classes)
        )

    def forward(self, features):
        logits = self.fc(features)
        return logits

    def get_prob(self, logits, temperature=1):
        probs = torch.softmax(logits / temperature, dim=1)
        return probs

    def get_loss(self, logits, targets, temperature=1):
        log_probs = torch.log_softmax(logits / temperature, dim=1)
        loss = - torch.sum(log_probs * targets)
        return loss

    def estimate_threshold(self, probs, labels, percentile=5):  # TODO
        self.classwise_thresholds = []
        classwise_probs = []
        for i in range(self.num_classes):
            classwise_probs.append([])

        for i, val in enumerate(probs):
            if self.classid_list.count(labels[i]) > 0:
                id_index = self.classid_list.index(labels[i])
                maxProb = np.max(probs[i])
                if probs[i, id_index] == maxProb:
                    classwise_probs[id_index].append(probs[i, id_index])

        for val in classwise_probs:
            if len(val) == 0:
                self.classwise_thresholds.append(0)
            else:
                threshold = np.percentile(val, percentile)
                self.classwise_thresholds.append(threshold)

        return self.classwise_thresholds

    def estimate_threshold_logits(self, feature_encoder, validation_loader, percentile=5):
        self.eval()
        self.classwise_thresholds = []
        classwise_logits = []
        for i in range(self.num_classes):
            classwise_logits.append([])

        for i, (images, labels) in enumerate(validation_loader):
            images = images.to(self.device)
            with torch.no_grad():
                features = feature_encoder.get_feature(images)
                logits = self.forward(features)
                maxLogit, maxIndexes = torch.max(logits, 1)

            for j, label in enumerate(labels):
                id_index = self.classid_list.index(label)
                if maxIndexes[j] == id_index:
                    classwise_logits[id_index].append(logits[j, id_index].item())

        for val in classwise_logits:
            if len(val) == 0:
                self.classwise_thresholds.append(0)
            else:
                threshold = np.percentile(val, percentile)
                self.classwise_thresholds.append(threshold)
        return self.classwise_thresholds

    def predict_logits(self, features):
        logits = self.forward(features)

        maxLogits, maxIndexes = torch.max(logits, 1)
        prediction = torch.zeros([maxIndexes.shape[0]], requires_grad=False).to(self.device)

        for i in range(maxIndexes.shape[0]):
            prediction[i] = self.classid_list[maxIndexes[i]]
            if maxLogits[i] <= self.classwise_thresholds[maxIndexes[i]]:
                prediction[i] = -1

        return prediction.long(), logits.detach().cpu().numpy()

    def predict_closed(self, features):
        outs = self.forward(features)
        probs = torch.sigmoid(outs)

        maxProb, maxIndexes = torch.max(probs, 1)
        prediction = torch.zeros([maxIndexes.shape[0]], requires_grad=False).to(self.device)

        for i in range(maxIndexes.shape[0]):
            prediction[i] = self.classid_list[maxIndexes[i]]

        return prediction.long()


def make_similarities(t1, t2):
    pair = pair_enumeration(t1.unsqueeze(1), t2.unsqueeze(1))
    # print(pair[:200])
    return (pair[:, 0] == pair[:, 1]).long()


def pair_enumeration(x1, x2):
    """
    input: x1 [A, ..., D]
           x2 [B, ..., D]
    return: [A * B, ..., 2D]

    input x1:
    [[a],
     [b]]

    input x2:
    [[x],
     [y],
     [z]]

    return:
    [[a, x],
     [b, x],
     [a, y],
     [b, y],
     [a, z],
     [b, z]]
    """
    # assert x1.ndimension() == 2 and x2.ndimension() == 2, 'Input dimensions must be 2'
    A, D = x1.size(0), x1.size(-1)
    B = x2.size(0)

    # Repeat x1 and x2 for pair-wise concatenation
    x1_rep = x1.repeat(B, 1)
    # x1_rep = x1.repeat_interleave(B, dim=0)
    x2_rep = x2.repeat(1, A).view(-1, D)
    # x2_rep = x2.repeat_interleave(A, dim=-1).view(-1, D)

    # Concatenate x1_rep and x2_rep element-wise
    return torch.cat((x1_rep, x2_rep), dim=-1)


def pair_reshape(x, num_classes=None):
    '''
    training == True
        input:  [B*B,D]
        return: [B,B,D]

        input  [[a,a],
                [b,a],
                [a,b],
                [b,b]]
        return [[[a,a],
                 [b,a]],
                [[a,b],
                 [b,b]]]
    training == False
        input:  [B*N,D]  # N is num of known class
        return: [B,N,D]

        input  [[a,a],
                [b,a],
                [a,b],
                [b,b]]
        return [[[a,a],
                 [b,a]],
                [[a,b],
                 [b,b]]]
    '''
    if num_classes is None:
        B = int(x.size(0) ** 0.5)
        N = B
    else:
        B = int(x.size(0) / num_classes)
        N = num_classes
    return x.view(B, N, -1)


# def mixup_data(img, target, alpha=1.0):
#     beta = np.random.beta(alpha, alpha)
#
#     index = torch.randperm(img.size(0))
#     mixed_img = beta * img + (1 - beta) * img[index]
#     mixed_target = beta * target + (1 - beta) * target[index]
#     # target: mixed_target if mixed_target == target else -1
#     return mixed_img, torch.where(mixed_target == target, mixed_target, -1).to(torch.int)
def mixup_data(x, beta, index):
    # beta = np.random.beta(alpha, alpha)
    # index = torch.randperm(img.size(0))
    # mixed_img = beta * img + (1 - beta) * img[index]
    mixed_x = beta * x + (1 - beta) * x[index]
    # target: mixed_target if mixed_target == target else -1
    # return mixed_img, torch.where(mixed_target == target, mixed_target, -1).to(torch.int)
    return mixed_x

