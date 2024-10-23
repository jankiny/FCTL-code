import argparse
import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_curve

from src.data import *
from src.config import print_config
from src.utils import parse_known_classes
from train_fctl_2 import FCTLStage2
from src.model.simnet_2s_mixup_ad import (make_similarities, pair_enumeration, pair_reshape)
from src.utils import (compute_oscr, process_config)

class FCTLTester(FCTLStage2):
    def test_dataloader(self, split_type='out_test'):
        valid_dataset_in = eval("get{}Dataset".format(self.dataset))(
            image_size=self.image_size,
            split='in_test',
            data_path=self.data_dir,
            known_classes=self.known_classes
        )
        valid_dataloader_in = DataLoader(
            valid_dataset_in,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        if self.osr_dataset is not None:
            valid_dataset_out = eval("get{}Dataset".format(self.osr_dataset))(
                image_size=self.image_size,
                split=split_type,
                data_path=self.data_dir,
                include_classes=self.osr_classes
            )
            valid_dataloader_out = DataLoader(
                valid_dataset_out,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
        else:
            valid_dataset_out = eval("get{}Dataset".format(self.dataset))(
                image_size=self.image_size,
                split=split_type,
                data_path=self.data_dir,
                known_classes=self.known_classes
            )
            valid_dataloader_out = DataLoader(
                valid_dataset_out,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )

        return [valid_dataloader_in, valid_dataloader_out]

    def on_all_test_start(self):
        self._calc_anchor()


    # def unknown_score(self, feat, cls_pred, sim_pred, anchor_feature):
    #     score_reduce = lambda x: x.view(x.size(0), -1).mean(dim=1)
    #     feat_detach = feat.detach()
    #
    #     # Maximum Logit Score (not Maximum Softmax Probability (MSP))
    #     pred = cls_pred.argmax(dim=1)  # 最大值的下标，预测值
    #     max_prob = cls_pred.softmax(dim=1).max(dim=1).values  # 使用softmax获取最大概率值
    #     cls_scores = torch.gather(cls_pred, 1, pred.unsqueeze(1)).squeeze(1)
    #     rep_scores = torch.abs(feat_detach).mean(dim=1)
    #     R = [cls_scores, rep_scores, max_prob]
    #
    #     # distance
    #     if feat.size(1) != anchor_feature.size(1):
    #         def pad_features_to_match(x, anchor_feature):
    #             max_dim = max(x.size(1), anchor_feature.size(1))
    #
    #             padded_x = torch.zeros((x.size(0), max_dim)).to(x.device)
    #             padded_anchor_feature = torch.zeros((anchor_feature.size(0), max_dim)).to(anchor_feature.device)
    #
    #             padded_x[:, :x.size(1)] = x
    #             padded_anchor_feature[:, :anchor_feature.size(1)] = anchor_feature
    #
    #             return padded_x, padded_anchor_feature
    #         feat, anchor_feature = pad_features_to_match(feat, anchor_feature)
    #     dists = euclidean_distance(feat, anchor_feature)
    #     dis_scores = torch.gather(dists, 1, pred.unsqueeze(1)).squeeze(1)
    #
    #     # similarity
    #     if self.model_config.sim_task == 'cls':
    #         # print('batch_sim_pred size', sim_pred.size()) # size: [1600, 2]
    #         reshaped_sim_pred = pair_reshape(sim_pred, num_classes=self.num_classes) # size: [16, 100, 2]
    #         reshaped_sim_pred = reshaped_sim_pred[:, :, 1] # 取1作为相似度值，
    #         similarity_scores = torch.gather(reshaped_sim_pred, 1, pred.unsqueeze(1)).squeeze(1)
    #     else: # self.model_config.sim_task == 'reg'
    #         similarity_scores = torch.gather(sim_pred, 0, pred.unsqueeze(1)).squeeze(1)
    #
    #     scores = torch.stack([
    #         score_reduce(R[0]),
    #         score_reduce(-0.01 * dis_scores),
    #         score_reduce(-self.sim_lambda * similarity_scores),
    #         # score_reduce(-0.01 * similarity_scores),
    #     ], dim=1)
    #     return pred, scores

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        batch_data, batch_target = batch
        device = batch_data.device
        if self.anchor_class_idx.device != device:
            # For lightning sanity checking
            self.anchor_class_idx = self.anchor_class_idx.to(device)
            self.anchor_feats = self.anchor_feats.to(device)
        batch_sim_target = make_similarities(self.anchor_class_idx, batch_target)
        batch_emd, batch_sim_pred, batch_cls_pred = self(
            batch_data,
            mean_feat=self.anchor_feats,
            feat_cls=True
        )

        # update loss
        if dataloader_idx == 0:
            # calculate sim_loss in out-data leads 'nll_loss' error, which labels should in [0, n_classes - 1]
            sim_loss = self.sim_alpha * self.sim_criterion(batch_sim_pred, batch_sim_target)
            cls_loss = self.criterion(batch_cls_pred, batch_target)
            loss = sim_loss + cls_loss
            # logging loss
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_sim_loss', sim_loss, on_step=False, on_epoch=True)
            self.log('val_cls_loss', cls_loss, on_step=False, on_epoch=True)

            # calculate metrics
            acc1 = self.accuracy(torch.argmax(batch_cls_pred, dim=1), batch_target)
            self.log('val_acc1', acc1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # 返回需要在validation_epoch_end中使用的数据
        _, unknown_score = self.unknown_score(batch_emd, batch_cls_pred, batch_sim_pred, self.anchor_feats)
        # print('unknown_score', unknown_score[:4])
        # print('unknown_score[:, 0]', unknown_score[:, 0][:4])
        if dataloader_idx == 0:  # known sample (Closed-Set)
            binary_target = torch.full_like(batch_target, 1)
            # self.val_auroc.update(unknown_score[:, 2], binary_target) # CS
            # self.val_auroc.update(unknown_score[:, 0], binary_target) # MLS
            self.val_auroc.update(unknown_score[:, 0] + unknown_score[:, 2], binary_target) # CS + MLS
            batch_target = batch_target  # ground_truth of known classes are their labels
        else:  # unknown sample (Open-Set/Out-Set)
            binary_target = torch.full_like(batch_target, 0)
            # self.val_auroc.update(unknown_score[:, 2], binary_target) # CS
            # self.val_auroc.update(unknown_score[:, 0], binary_target) # MLS
            self.val_auroc.update(unknown_score[:, 0] + unknown_score[:, 2], binary_target) # CS + MLS
            batch_target = torch.full_like(batch_target, -1)  # but of unknown classes are -1
        self.validation_step_outputs.append({
            "unknown_score": unknown_score,
            "closeset_unknown_score": unknown_score if dataloader_idx == 0 else None,
            "binary_target": binary_target,
            "batch_cls_pred": batch_cls_pred,
            "batch_target": batch_target,
            "batch_emd": batch_emd
        })

    def on_test_end(self):
        # confusion_matrix = self.sim_confusion_matrix.compute()
        # self.sim_confusion_matrix.reset()
        auroc_score = self.val_auroc.compute()
        self.val_auroc.reset()

        unknown_score_list = []
        binary_target_list = []
        pred_list = []
        target_list = []
        for out in self.validation_step_outputs:
            unknown_score_list.append(out["unknown_score"])
            binary_target_list.append(out["binary_target"])
            pred_list.append(out["batch_cls_pred"])
            target_list.append(out["batch_target"])
        # epoch_scores = torch.cat([(scores[:, 2]).unsqueeze(1).mean(dim=1) for scores in unknown_score_list]) # CS
        # epoch_scores = torch.cat([(scores[:, 0]).unsqueeze(1).mean(dim=1) for scores in unknown_score_list]) # MLS
        epoch_scores = torch.cat([(scores[:, 0] + scores[:, 2]).unsqueeze(1).mean(dim=1) for scores in unknown_score_list]) # CS + MLS
        epoch_binary_targets = torch.cat(binary_target_list)
        epoch_pred = torch.cat(pred_list, dim=0)
        epoch_targets = torch.cat(target_list, dim=0)

        # Open Set Classification Rate (OSCR)
        kl = torch.sum(torch.ne(epoch_targets, -1)).item()  # length of known samples
        oscr = compute_oscr(epoch_scores[:kl], epoch_scores[kl:], epoch_pred[:kl], epoch_targets[:kl])
        # if isinstance(self.logger, WandbLogger):
        #     self.logger.experiment.log(
        #         {'unknown_score': wandb.Table(
        #             data=torch.cat(unknown_score_list[:10] + unknown_score_list[-10:]).cpu().numpy(), # 取前10个和后10个
        #             columns=["known", "unknown"]
        #         )})

        # 使用 AUROC 计算最佳阈值
        fpr, tpr, thresholds = roc_curve(epoch_binary_targets.cpu().numpy(), epoch_scores.cpu().numpy())
        youdens_j = tpr - fpr # 通过 Youden's J statistic 来选择最佳阈值，得到一个能够平衡 TPR 和 FPR 的阈值
        best_threshold_index = np.argmax(youdens_j)
        threshold = thresholds[best_threshold_index]

        # Macro F1-Score
        # print('epoch_scores[:100]', epoch_scores[:100])
        # print('epoch_scores[kl:kl+100]', epoch_scores[kl:kl+100])
        # closed_set_scores = epoch_scores[:kl].sort().values
        # print("closed_set_scores", closed_set_scores)
        # threshold = epoch_scores[int(0.05 * len(closed_set_scores)) - 1]
        log(f'threshold: {threshold}')
        test_results = torch.argmax(epoch_pred, dim=1)
        test_results[torch.max(epoch_pred, dim=1)[0] <= threshold] = self.num_classes  # 将低于阈值的样本标记为未知类别
        true_results = epoch_targets.clone()
        true_results[true_results == -1] = self.num_classes  # 将未知样本的标签设置为未知类别
        macro_f1_score = f1_score(true_results.cpu(), test_results.cpu(), average="macro")


        self.validation_step_outputs.clear()

        acc1 = self.trainer.logged_metrics.get('val_acc1/dataloader_idx_0', torch.tensor(0.0))
        results = {
            "val_acc1": acc1,
            "val_auroc": auroc_score,
            "val_macro_f1": macro_f1_score,
            "val_oscr": oscr
        }
        for key, value in results.items():
            log('    {:15s}: {}'.format(key, value))

def get_test_config():
    parser = argparse.ArgumentParser("FCTL Test")
    parser.add_argument("--exp-name", type=str, default="Test", help="experiment name")
    parser.add_argument("--checkpoint-path", type=str,
                        default=None,
                        help="model checkpoint to load weights")
    # parser.add_argument("--known-classes", type=parse_known_classes,
    #                     default=None,
    #                     help="list of known classes, e.g. '[2, 3, 4, 5, 6, 7]'")
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=2023,
                        help="random seed for choosing the training classes")
    parser.add_argument("--dataset", type=str, default=None,
                            help="dataset for fine-tunning/evaluation")
    parser.add_argument('--eval', default=True, action='store_true', help='evaluate on dataset')

    config = parser.parse_args()

    # model config
    # config = eval("get_{}_config".format(config.model_arch))(config)
    # config.model_arch = 'vitb16'
    # config.model_config = eval("{}Config".format(config.model_arch))(config)
    # process_config(config)
    print_config(config)
    return config


if __name__ == '__main__':
    config = get_test_config()
    pl.seed_everything(config.random_seed)
    model = FCTLTester.load_from_checkpoint(config.checkpoint_path)
    # model.n_gpu = config.n_gpu
    if model.dataset == 'TinyImageNet':
        model.sim_lambda = 0.1
    model.training_ended = True # To output result
    # read hyperparameters
    config.dataset = model.dataset
    config.batch_size = model.batch_size
    config.model_arch = model.arch
    config.image_size = model.image_size
    config.lr = model.lr
    config.wd = model.weight_decay
    config.num_classes = model.num_classes
    process_config(config, exp_type='eval')


    trainer = pl.Trainer(
        devices=1,
        accelerator='gpu',
    )

    if config.dataset == 'CUB' or config.dataset == 'FGVC':
        difficulties = ('out_test', 'Easy', 'Medium', 'Hard')
    else:
        difficulties = ('out_test')

    if config.dataset == 'CUB' or config.dataset == 'FGVC':
        model.on_all_test_start()
        for difficulty in difficulties:
            print(f'----------------- {difficulty} -------------------')
            trainer.test(model=model, dataloaders=model.test_dataloader(split_type=difficulty))
            print(f'-------------- {difficulty} - END ----------------')
    else:
        # trainer.test(model=model) # for ood
        trainer.validate(model=model)

