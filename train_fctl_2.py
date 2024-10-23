# Stage2
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torchmetrics import Accuracy, AUROC, F1Score

from src.config import get_train_config
from src.data import *
from src.losses import *
from src.model.fctl import FCTL
from src.model.simnet_2s_mixup_ad import (make_similarities, pair_enumeration, pair_reshape)
from src.utils import (compute_oscr, euclidean_distance, write_result)


class FCTLStage2(FCTL):
    def __init__(
            self,
            # train/eval
            is_train: bool = True,
            # data
            image_size: int = 256,
            data_dir: str = None,
            dataset: str = 'CUB',
            known_classes: list = None,
            num_classes: int = 1000,
            osr_dataset: str = None,
            osr_classes: list = None,
            num_workers: int = 0,
            transform: str = None,
            # model
            arch: str = 'vitb16',
            model_config: object = None,
            # train
            checkpoint_path: str = None,
            batch_size: int = 32,
            opt: str = 'adam',
            lr: float = 1e-2,
            lr_strategy: str = 'cosine',
            num_restarts: int = 2,
            label_smoothing: float = 0.1,
            weight_decay: float = 1e-4,
            train_steps: int = None,
            warmup_steps: int = None,
            max_epochs: int = None,
            warmup_epochs: int = None,
            random_seed: int = 0,
            # stage2
            cls_block: str = 'attention',
            sim_alpha: float = 0.0,
            sim_lambda: float = 0.0,
            frozen_encoder: bool = False,
            class_anchor: str = 'mean',
            anchor_freq: int = 30,
            **kwargs: object
    ):
        super().__init__(image_size, data_dir, dataset, known_classes, num_classes, osr_dataset,
                         osr_classes, num_workers, transform, arch, model_config, checkpoint_path, batch_size,
                         opt, lr, lr_strategy, num_restarts, label_smoothing, weight_decay, train_steps,
                         warmup_steps, max_epochs, warmup_epochs, random_seed, kwargs=kwargs)
        self.checkpoint_path = checkpoint_path
        self.cls_block = cls_block
        self.sim_alpha = sim_alpha
        self.frozen_encoder = frozen_encoder
        self.sim_lambda = sim_lambda if sim_lambda is not None else 0.01 # Some ckpts do not have this parameter
        self.class_anchor = class_anchor
        self.anchor_freq = anchor_freq
        self.save_hyperparameters()

        self.encoder, self.similarity_layer, self.fusion_layer, self.classifier \
            = self.init_model(self.arch, self.cls_block)

        self.anchor_feats = torch.ones((self.num_classes, self.encoder.num_features))
        self.anchor_class_idx = torch.ones(self.num_classes, dtype=torch.long)
        self.validation_step_outputs = []

        # self.confusion_matrix = ConfusionMatrix(num_classes=2, task='binary')
        self.accuracy = Accuracy(top_k=1, task='multiclass', num_classes=self.num_classes)
        self.val_auroc = AUROC(task='binary', num_classes=2)  # unknown(0)/ known(1)
        # self.val_macro_f1 = F1Score(average='macro', task='binary', num_classes=2)  # unknown(0)/ known(1), for ood test
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        if self.model_config.sim_task == 'cls':
            self.sim_criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:  # reg
            self.sim_criterion = BCEContrastiveLoss()

        if self.frozen_encoder is True:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _calc_anchor(self):
        print(f'Calculate anchor_feature in epoch {self.current_epoch}')
        class_embedding = {}
        data_loader = self.train_dataloader()
        with torch.no_grad():
            nb = len(data_loader)
            anchor_pbar = tqdm(enumerate(data_loader), total=nb, bar_format='{l_bar}{bar:10}{r_bar}',
                               desc='Feat ( In Data)')
            for batch_idx, (batch_data, batch_target) in anchor_pbar:
                batch_data = batch_data.to(self.device, non_blocking=True)
                batch_emd = self.encoder(batch_data)

                # Collect class-specific features
                for i in range(len(batch_target)):
                    class_id = batch_target[i].item()
                    if class_id not in class_embedding:
                        class_embedding[class_id] = []
                    class_embedding[class_id].append(batch_emd[i])
        # Calculate the mean embedding for each class
        if self.class_anchor == 'mean':
            self.anchor_class_idx = []
            mean_class_features = []
            for class_id, feats in class_embedding.items():
                self.anchor_class_idx.append(class_id)
                mean_class_features.append(torch.stack(feats).mean(dim=0))
            self.anchor_feats = torch.stack(mean_class_features, dim=0)
            # moving self.anchor_class_idx to gpu
            self.anchor_class_idx = torch.tensor(self.anchor_class_idx, device=self.device)

    def forward(self, x, mean_feat, feat_cls=True):
        feat = self.encoder(x)
        residual = feat # [bs, 768]

        feat_pairs = pair_enumeration(mean_feat, feat) # feat_pairs: [bs * nc, 768 * 2]
        diff_feat_list, sim_pred = self.similarity_layer(feat_pairs, feat_cls=True) # diff_feat: [bs * nc, 768 * 2]

        if self.cls_block == 'concat':
            cls_feat = pair_reshape(diff_feat_list[-1], num_classes=self.num_classes) # [bs, nc, 768 * 2]
            cls_feat = torch.mean(cls_feat, dim=1)  # [bs, 768 * 2]
            cls_feat = torch.cat((cls_feat, feat), dim=1)
        elif self.cls_block == 'cross-attention':
            fusion_feat = self.fusion_layer(feat, diff_feat_list)  # query, content
            cls_feat = fusion_feat
        else:
            cls_feat = feat

        if feat_cls:
            return cls_feat, sim_pred, self.classifier(cls_feat)
        else:
            return sim_pred, self.classifier(cls_feat)

    def on_fit_start(self):
        if self.training:
            self.load_checkpoint(self.checkpoint_path)

    def on_train_epoch_start(self):
        if self.current_epoch % self.anchor_freq != 0:
            return
        # calculating anchor_feature before train start
        self._calc_anchor()

    def training_step(self, batch, batch_idx):
        batch_data, batch_target = batch
        batch_sim_target = make_similarities(self.anchor_class_idx, batch_target)

        batch_emd, batch_sim_pred, batch_cls_pred = self(
            batch_data,
            mean_feat=self.anchor_feats,
            feat_cls=True
        )
        # update loss
        sim_loss = self.sim_alpha * self.sim_criterion(batch_sim_pred, batch_sim_target)
        cls_loss = self.criterion(batch_cls_pred, batch_target)
        loss = sim_loss + cls_loss
        # log loss
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_sim_loss', sim_loss, on_step=False, on_epoch=True)
        self.log('train_cls_loss', cls_loss, on_step=False, on_epoch=True)


        # calculate metrics
        acc1 = self.accuracy(torch.argmax(batch_cls_pred, dim=1), batch_target)
        # f1_score = self.f1(torch.argmax(batch_cls_pred, dim=1), batch_target)
        # self.sim_confusion_matrix.update(torch.argmax(batch_sim_pred, dim=1), batch_sim_target) # not used in training

        # log
        self.log('train_acc1', acc1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_validation_start(self):
        if self.training_ended:
            self._calc_anchor()

    def unknown_score(self, feat, cls_pred, sim_pred, anchor_feature):
        score_reduce = lambda x: x.view(x.size(0), -1).mean(dim=1)
        feat_detach = feat.detach()

        # Maximum Logit Score (not Maximum Softmax Probability (MSP))
        pred = cls_pred.argmax(dim=1)  # 最大值的下标，预测值
        max_prob = cls_pred.softmax(dim=1).max(dim=1).values  # 使用softmax获取最大概率值
        cls_scores = torch.gather(cls_pred, 1, pred.unsqueeze(1)).squeeze(1)
        rep_scores = torch.abs(feat_detach).mean(dim=1)
        R = [cls_scores, rep_scores, max_prob]

        # distance
        if feat.size(1) != anchor_feature.size(1):
            def pad_features_to_match(x, anchor_feature):
                max_dim = max(x.size(1), anchor_feature.size(1))

                padded_x = torch.zeros((x.size(0), max_dim)).to(x.device)
                padded_anchor_feature = torch.zeros((anchor_feature.size(0), max_dim)).to(anchor_feature.device)

                padded_x[:, :x.size(1)] = x
                padded_anchor_feature[:, :anchor_feature.size(1)] = anchor_feature

                return padded_x, padded_anchor_feature
            feat, anchor_feature = pad_features_to_match(feat, anchor_feature)
        dists = euclidean_distance(feat, anchor_feature)
        dis_scores = torch.gather(dists, 1, pred.unsqueeze(1)).squeeze(1)

        # similarity
        if self.model_config.sim_task == 'cls':
            # print('batch_sim_pred size', sim_pred.size()) # size: [1600, 2]
            reshaped_sim_pred = pair_reshape(sim_pred, num_classes=self.num_classes) # size: [16, 100, 2]
            reshaped_sim_pred = reshaped_sim_pred[:, :, 1]
            similarity_scores = torch.gather(reshaped_sim_pred, 1, pred.unsqueeze(1)).squeeze(1)
        else: # self.model_config.sim_task == 'reg'
            similarity_scores = torch.gather(sim_pred, 0, pred.unsqueeze(1)).squeeze(1)

        scores = torch.stack([
            score_reduce(R[0]),
            score_reduce(-0.01 * dis_scores),
            score_reduce(-self.sim_lambda * similarity_scores),
            # score_reduce(-0.01 * similarity_scores),
        ], dim=1)
        return pred, scores

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
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
            # self.val_auroc.update(unknown_score[:, 0], binary_target)
            self.val_auroc.update(unknown_score[:, 0] + unknown_score[:, 2], binary_target)
            batch_target = batch_target # ground_truth of known classes are their labels
        else:  # unknown sample (Open-Set/Out-Set)
            binary_target = torch.full_like(batch_target, 0)
            # self.val_auroc.update(unknown_score[:, 0] , binary_target)
            self.val_auroc.update(unknown_score[:, 0] + unknown_score[:, 2], binary_target)
            batch_target = torch.full_like(batch_target, -1) # but of unknown classes are -1
        self.validation_step_outputs.append({
            "unknown_score": unknown_score,
            "binary_target": binary_target,
            "batch_cls_pred": batch_cls_pred,
            "batch_target": batch_target,
            "batch_emd": batch_emd
        })

    def on_validation_epoch_end(self):
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
        epoch_scores = torch.cat([scores[:, 0].unsqueeze(1).mean(dim=1) for scores in unknown_score_list])
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

        self.validation_step_outputs.clear()

        # logging
        self.log('val_auroc', auroc_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_oscr', oscr, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # average val loss
        loss_0 = self.trainer.logged_metrics.get('val_loss/dataloader_idx_0', torch.tensor(0.0))
        self.log('val_loss', loss_0, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # val acc1 = val dataset_in_acc1
        acc1 = self.trainer.logged_metrics.get('val_acc1/dataloader_idx_0', torch.tensor(0.0))
        self.log('val_acc1', acc1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_train_epoch_end(self):
        # Lightning sequence: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks

        # logging to console
        keys = ['train_loss', 'train_acc1', 'val_loss', 'val_acc1', 'val_auroc', 'val_oscr']
        log('\n    {:15s}: {}'.format(str('epoch'), self.current_epoch))
        for key in keys:
            log('    {:15s}: {}'.format(str(key), self.trainer.callback_metrics.get(key, 'N/A')))
        # log('    {:15s}: \n{}'.format(str('train_confusion_matrix'), confusion_matrix))

        # save to result file
        result_log = {}
        for key in keys:
            result_log[key] = float(self.trainer.callback_metrics.get(key, '0.0'))
        write_result(result_log.keys(), result_log.values(), self.current_epoch, config.result_dir)

    def on_validation_end(self):
        if self.training_ended and self.training_ended is True:
            # logging to console
            keys = ['val_loss', 'val_acc1', 'val_auroc', 'val_oscr']
            log('\n    {:15s}: {}'.format(str('epoch'), self.current_epoch))
            for key in keys:
                log('    {:15s}: {}'.format(str(key), self.trainer.callback_metrics.get(key, 'N/A')))
            # log('    {:15s}: \n{}'.format(str('train_confusion_matrix'), confusion_matrix))

            # save to result file. Commented as config.result_dir should not be used in training model.
            # result_log = {}
            # for key in keys:
            #     result_log[key] = float(self.trainer.callback_metrics.get(key, '0.0'))
            # write_result(result_log.keys(), result_log.values(), self.current_epoch, config.result_dir)



if __name__ == '__main__':
    config, lightning_loggers = get_train_config(stage=2)
    config.num_workers = config.num_workers_pgpu * config.n_gpu

    pl.seed_everything(config.random_seed)

    # Lightning Module
    model = FCTLStage2(
        # data
        image_size=config.image_size,
        data_dir=config.data_dir,
        dataset=config.dataset,
        known_classes=config.known_classes,
        num_classes=config.num_classes,
        osr_dataset=config.osr_dataset,
        osr_classes=config.osr_classes,
        num_workers=config.num_workers,
        transform=config.transform,
        # model
        arch=config.model_arch,
        model_config=config.model_config,
        # train
        checkpoint_path=config.checkpoint_path,
        batch_size=config.batch_size,
        opt=config.opt,
        lr=config.lr,
        lr_strategy=config.lr_strategy,
        num_restarts=config.num_restarts,
        label_smoothing=config.label_smoothing,
        weight_decay=config.wd,
        train_steps=config.train_steps,
        warmup_steps=config.warmup_steps,
        max_epochs=config.max_epochs,
        warmup_epochs=config.warmup_epochs,
        random_seed=config.random_seed,
        # stage2
        cls_block=config.fusion,
        sim_alpha=config.sim_alpha,
        sim_lambda=config.sim_lambda,
        frozen_encoder=config.frozen_encoder,
        class_anchor=config.class_anchor,
        anchor_freq=config.anchor_freq
    )
    # watch model
    for logger in lightning_loggers:
        if isinstance(logger, WandbLogger):
            logger.watch(model, log_graph=True)
    # print(model)
    # # summary(model, input_size=(16, 3, 448, 448), mean_feat = torch.rand((100, 768)))
    # exit(0)


    callbacks = [
        ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            # filename='{epoch}-{val_loss:.2f}',
            filename='best',
            save_last=True, # TODO: setting False for less storage
            save_top_k=1,
            monitor="val_auroc",
            mode='max'
        ),
        LearningRateMonitor(logging_interval='step' if config.lr_strategy == 'step' else 'epoch'),
        # EarlyStopping(monitor='val_loss', patience=10)
    ]
    # if config.wandb:
    #     patience = (config.max_epochs if config.max_epochs is not None else config.train_steps) // 5
    #     callbacks.append(WandbAlertCallback(
    #         monitor='val_auroc',
    #         mode='max',
    #         patience=patience
    #     ))

    trainer = pl.Trainer(
        max_epochs=config.max_epochs if config.max_epochs is not None else -1,
        max_steps=config.train_steps if config.train_steps is not None else -1,
        devices=config.n_gpu,
        accelerator='gpu',
        sync_batchnorm=config.n_gpu > 1,
        # precision=32,
        precision='16-mixed',
        logger=lightning_loggers,
        callbacks=callbacks
    )

    # if config.eval:
    #     trainer.validate(model, ckpt_path=config.checkpoint_path)
    # else:
    trainer.fit(model)
    # validate best model
    best_model_path = os.path.join(config.checkpoint_dir, 'best.ckpt')
    trainer.validate(model, ckpt_path=best_model_path)


