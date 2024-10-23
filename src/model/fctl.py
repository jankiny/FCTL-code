import random
from typing import Union

import math
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data import *
from src.config import *
from src.model.cnn import ResNet50, ResNet101
from src.model.simnet_2s_mixup_ad import (SimilarityLayer, FusionLayer, OSRClassifier)
from src.model.vit import VisionTransformer, ViTB16
from src.utils import (load_checkpoint, resize_vit_pos_embedding)


class FCTL(pl.LightningModule):
    def __init__(
            self,
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
            model_config: Union[vitb16Config, swinConfig] = None,
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

            **kwargs: object
    ):

        super().__init__()
        # data
        self.image_size = image_size
        self.data_dir = data_dir
        self.dataset = dataset
        self.known_classes = known_classes
        self.num_classes = num_classes
        self.osr_dataset = osr_dataset
        self.osr_classes = osr_classes
        self.num_workers = num_workers
        self.transform = transform
        # model
        self.arch = arch
        self.model_config = model_config
        # train
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.opt = opt
        self.lr = lr
        self.lr_strategy = lr_strategy
        self.num_restarts = num_restarts
        self.label_smoothing = label_smoothing
        self.weight_decay = weight_decay
        self.train_steps = train_steps
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.random_seed = random_seed

        # set self.total_classes
        random.seed(random_seed)
        if dataset == "MNIST" or dataset == "SVHN" or dataset == "CIFAR10":
            self.total_classes = 10
        elif dataset == "TinyImageNet" or dataset == "CUB":
            self.total_classes = 200
        elif dataset == "FGVC":
            self.total_classes = 100
        else:
            log(f'Unknown dataset, please set total_classes in{__file__}')
            exit(0)

        # set self.known_classes: use custom data split
        if len(known_classes) == num_classes and max(known_classes) < self.total_classes:
            self.known_classes = known_classes
        else:
            self.known_classes = random.sample(range(0, self.total_classes), num_classes)

        self.training_ended = False


        if self.dataset == 'FGVC':
            if isinstance(self.transform, tuple):
                train_transform, test_transform = self.transform
            else:
                train_transform, test_transform = (
                    get_transform(transform_type=self.transform,
                                  image_size=self.image_size,
                                  args=None))
            unknown_classes = list(set(range(100)) - set(self.known_classes))
            self.fgvc_datasets = get_aircraft_datasets(
                train_transform, test_transform,
                train_classes=self.known_classes,
                open_set_classes=unknown_classes,
                balance_open_set_eval=True,
                split_train_val=False,
                seed=2023
            )
            # if self.transform == 'rand-augment':
            #     if args.rand_aug_m is not None:
            #         if args.rand_aug_n is not None:
            #             datasets['train'].transform.transforms[0].m = args.rand_aug_m
            #             datasets['train'].transform.transforms[0].n = args.rand_aug_n
            train_labels = []
            for ii in range(len(self.fgvc_datasets['train'])):
                train_labels.append(self.fgvc_datasets['train'][ii][1])
            print(len(train_labels))  # 2997
            # print(train_labels)
            self.train_labels = np.array(train_labels)

    def on_train_end(self):
        self.training_ended = True

    def init_model(self, arch, cls_block=None):
        if 'vit' in arch:
            encoder = VisionTransformer(
                image_size=(self.image_size, self.image_size),
                patch_size=(self.model_config.patch_size, self.model_config.patch_size),
                emb_dim=self.model_config.emb_dim,
                mlp_dim=self.model_config.mlp_dim,
                num_heads=self.model_config.num_heads,
                num_layers=self.model_config.num_layers,
                num_classes=self.num_classes,
                attn_dropout_rate=self.model_config.attn_dropout_rate,
                dropout_rate=self.model_config.dropout_rate,
            )
            # encoder = ViTB16(
            #     image_size=self.image_size,
            #     # patch_size=self.model_config.patch_size,
            #     # num_layers=self.model_config.num_layers,
            #     # num_heads=self.model_config.num_heads,
            #     # hidden_dim=self.model_config.emb_dim,
            #     # mlp_dim=self.model_config.mlp_dim,
            #     attention_dropout=self.model_config.attn_dropout_rate,
            #     dropout=self.model_config.dropout_rate,
            #     # num_classes=self.num_classes,
            #     )
        elif arch == 'cnnr50':
            encoder = ResNet50(
                feature_dim=self.model_config.emb_dim,
            )
        elif arch == 'cnnr101':
            encoder = ResNet101(
                feature_dim=self.model_config.emb_dim,
            )
        else:
            raise ValueError(f'encoder({arch}) is not supported')

        similarity_layer = SimilarityLayer(
            emb_dim=2 * encoder.num_features,
            mlp_dim=self.model_config.mlp_dim,
            sim_block=self.model_config.sim_block,
            sim_task=self.model_config.sim_task,
            num_sim_block=self.model_config.num_sim_block,
            num_heads=self.model_config.sim_num_heads,
            dropout_rate=self.model_config.dropout_rate,
        )
        if cls_block is not None:
            fusion_layer = FusionLayer(
                emb_dim=encoder.num_features,
                mlp_dim=self.model_config.mlp_dim,
                num_heads=self.model_config.sim_num_heads,
                dropout_rate=self.model_config.dropout_rate,
                sim_block=self.model_config.sim_block,
                num_sim_block=self.model_config.num_sim_block,
            )

            if cls_block == 'none':
                classifier = OSRClassifier(2 * encoder.num_features, self.num_classes)
            elif cls_block == 'concat':
                classifier = OSRClassifier(3 * encoder.num_features, self.num_classes)
            elif cls_block == 'cross-attention_concat':
                classifier = OSRClassifier(2 * encoder.num_features, self.num_classes)
            elif cls_block == 'cross-attention':
                classifier = OSRClassifier(encoder.num_features, self.num_classes)
            elif cls_block == 'residual':
                classifier = OSRClassifier(encoder.num_features, self.num_classes)
            else:
                raise ValueError(f'classifier({cls_block}) is not supported')

            return encoder, similarity_layer, fusion_layer, classifier

        return encoder, similarity_layer

    def load_checkpoint(self, checkpoint_path):
        # Warning: please make sure the pretrained weights are stored in the 'pretrained_model' directory
        if checkpoint_path == '':
            log(colorstr('green', 'bold', "Train from scratch."))
            return
        elif 'pretrained_model' in checkpoint_path:
            if 'swim' in checkpoint_path:
                net_dict = self.encoder.state_dict()
                pretrained_dict = torch.load(checkpoint_path)['model']
                pretrained_dict = {('swinB.' + k): v for k, v in pretrained_dict.items() if
                                   (('swinB.' + k) in net_dict) and ('classifier' not in k) and (
                                           k not in ['layers.0.blocks.1.attn_mask',
                                                     'layers.1.blocks.1.attn_mask',
                                                     'layers.2.blocks.1.attn_mask',
                                                     'layers.2.blocks.3.attn_mask',
                                                     'layers.2.blocks.5.attn_mask',
                                                     'layers.2.blocks.7.attn_mask',
                                                     'layers.2.blocks.9.attn_mask',
                                                     'layers.2.blocks.11.attn_mask',
                                                     'layers.2.blocks.13.attn_mask',
                                                     'layers.2.blocks.15.attn_mask',
                                                     'layers.2.blocks.17.attn_mask'])}
                net_dict.update(pretrained_dict)
                log(colorstr('green', 'bold', "Loading encoder weights from {}".format(checkpoint_path)))
                self.encoder.load_state_dict(net_dict)

            elif 'ViT' in checkpoint_path:
                state_dict = load_checkpoint(checkpoint_path, new_img=self.image_size,
                                             emb_dim=self.model_config.emb_dim, layers=self.model_config.num_layers,
                                             patch=self.model_config.patch_size)
                # state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))['state_dict']
                log("Loading pretrained weights from {}".format(checkpoint_path))
                if ('classifier.weight' in state_dict
                        and self.num_classes != state_dict['classifier.weight'].size(0)):  # vit16b pretrain
                    del state_dict['classifier.weight']
                    del state_dict['classifier.bias']
                    log("re-initialize fc layer")
                    missing_keys = self.encoder.load_state_dict(state_dict, strict=False)
                else:
                    # missing_keys = model.encoder.load_state_dict(state_dict['encoder_state_dict'], strict=False)
                    # model.similarity_head.load_state_dict(state_dict['simnethead_state_dict'], strict=False)
                    missing_keys = self.load_state_dict(state_dict, strict=False)

                log(colorstr('green', 'bold', "Loading encoder weights from {}".format(checkpoint_path)))
                log(f"Missing keys from checkpoint {missing_keys.missing_keys}")
                log(f"Unexpected keys in network : {missing_keys.unexpected_keys}")

            elif 'vit' in checkpoint_path:
                net_dict = self.encoder.state_dict()
                state_dict = torch.load(checkpoint_path)

                def remap_state_dict_keys(state_dict):
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        new_key = key
                        # 将预训练模型中的键名映射到当前模型的键名
                        new_key = new_key.replace("linear_1.weight", "0.weight")
                        new_key = new_key.replace("linear_1.bias", "0.bias")
                        new_key = new_key.replace("linear_2.weight", "3.weight")
                        new_key = new_key.replace("linear_2.bias", "3.bias")
                        new_state_dict[new_key] = value
                    return new_state_dict

                state_dict = remap_state_dict_keys(state_dict)
                state_dict = {('vit.' + k): v for k, v in state_dict.items() if
                              (('vit.' + k) in net_dict) and ('head' not in k)}
                state_dict = resize_vit_pos_embedding(state_dict, new_img_size=self.image_size,
                                                      patch_size=self.model_config.patch_size,
                                                      emb_dim=self.model_config.emb_dim)

                missing_keys = self.encoder.load_state_dict(state_dict, strict=False)
                log(colorstr('green', 'bold', "Loading encoder weights from {}".format(checkpoint_path)))
                log(f"Missing keys from checkpoint {missing_keys.missing_keys}")
                log(f"Unexpected keys in network : {missing_keys.unexpected_keys}")

            elif 'resnet' in checkpoint_path:
                net_dict = self.encoder.state_dict()
                state_dict = torch.load(checkpoint_path)
                state_dict = {('resnet.' + k): v for k, v in state_dict.items() if
                              (('resnet.' + k) in net_dict) and ('head' not in k)}

                missing_keys = self.encoder.load_state_dict(state_dict, strict=False)
                # if ('resnet.fc.weight' in state_dict and self.num_classes != state_dict['resnet.fc.weight'].size(0)):  # vit16b pretrain
                #     del state_dict['resnet.fc.weight']
                #     del state_dict['resnet.fc.bias']
                #     log("re-initialize fc layer")
                # else:
                #     # missing_keys = model.encoder.load_state_dict(state_dict['encoder_state_dict'], strict=False)
                #     # model.similarity_head.load_state_dict(state_dict['simnethead_state_dict'], strict=False)
                #     missing_keys = self.load_state_dict(state_dict, strict=False)

                log(colorstr('green', 'bold', "Loading encoder weights from {}".format(checkpoint_path)))
                log(f"Missing keys from checkpoint {missing_keys.missing_keys}")
                log(f"Unexpected keys in network : {missing_keys.unexpected_keys}")
        else:
            checkpoint = torch.load(checkpoint_path)

            # log("Loading stage1 weights from {}".format(checkpoint_path))
            log(colorstr('green', 'bold', "Loading stage1 weights from {}".format(checkpoint_path)))
            # 使用strict=False来允许部分加载权重，忽略不匹配的键
            self.load_state_dict(checkpoint['state_dict'], strict=False)

    def train_dataloader(self):
        # if self.dataset == 'FGVC' or  self.dataset == 'CUB':
        if self.dataset == 'FGVC':
            train_loader = DataLoader(
                self.fgvc_datasets['train'],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )
            return train_loader

        train_dataset = eval("get{}Dataset".format(self.dataset))(
            image_size=self.image_size,
            split='train',
            data_path=self.data_dir,
            known_classes=self.known_classes,
            transform=self.transform
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        # images, labels = next(iter(train_loader))
        # imshow_cls(images, labels, tf_writer=writer, f=os.path.join(config.result_dir, 'train_images.jpg'))

        return train_loader

    def val_dataloader(self):
        if not hasattr(self, 'cls_block'):  # stage = 1
            # Do not set False in stage 1, as the labels for samples within a single batch must vary.
            # If identical, this results in paired labels being uniformly similar (1),
            # hindering the effectiveness of contrastive
            shuffle = True
        else:   # stage = 2
            shuffle = False
        if self.dataset == 'FGVC':
            valid_dataloader_in = DataLoader(
                self.aircraft_datasets['test_known'],
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers
            )
            valid_dataloader_out = DataLoader(
                self.aircraft_datasets['test_unknown'],
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers
            )
            return [valid_dataloader_in, valid_dataloader_out]
        # print('val_dataloader: shuffle is {}'.format(shuffle))
        # Loading known set (close-set)
        valid_dataset_in = eval("get{}Dataset".format(self.dataset))(
            image_size=self.image_size,
            split='in_test',
            data_path=self.data_dir,
            known_classes=self.known_classes
        )
        valid_dataloader_in = DataLoader(
            valid_dataset_in,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers
        )

        # Loading unknowns
        if self.osr_dataset is not None:
            valid_dataset_out = eval("get{}Dataset".format(self.osr_dataset))(
                image_size=self.image_size,
                split='out_test',
                data_path=self.data_dir,
                include_classes=self.osr_classes
            )
            valid_dataloader_out = DataLoader(
                valid_dataset_out,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers
            )
        else:
            valid_dataset_out = eval("get{}Dataset".format(self.dataset))(
                image_size=self.image_size,
                split='out_test',
                data_path=self.data_dir,
                known_classes=self.known_classes
            )
            valid_dataloader_out = DataLoader(
                valid_dataset_out,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers
            )

        return [valid_dataloader_in, valid_dataloader_out]

    def configure_optimizers(self):
        # set optimizer
        if self.opt == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)

        # set lr scheduler
        if self.lr_strategy == 'one_cycle':
            if self.train_steps is None or self.warmup_steps is None:
                raise ValueError('when using "one_cycle" lr_strategy, train_steps and warmup_steps cannot be None')
            if self.train_steps is not None and self.warmup_steps is not None:
                lr_scheduler = {
                    'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                        optimizer=optimizer,
                        max_lr=self.lr,
                        total_steps=self.train_steps,
                        pct_start=self.warmup_steps / self.train_steps
                    ),
                    'interval': 'step',
                    'frequency': 1
                }
                # lr_scheduler = {
                #     'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                #         optimizer=optimizer,
                #         max_lr=self.lr,
                #         epochs=self.train_steps,
                #         steps_per_epoch=len(self.train_dataloader()),
                #         pct_start=self.warmup_steps / self.train_steps
                #     ),
                #     'interval': 'step',
                #     'frequency': 1
                # }
            elif self.max_epochs is not None and self.warmup_epochs is not None:
                lr_scheduler = {
                    'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                        optimizer=optimizer,
                        max_lr=self.lr,
                        epochs=self.max_epochs,
                        steps_per_epoch=len(self.train_dataloader()),
                        pct_start=self.warmup_epochs / self.max_epochs
                    ),
                    'interval': 'step',
                    'frequency': 1
                }
            else:
                raise ValueError(f"Unexpect Value in train_steps{self.train_steps}, "
                                 f"warmup_steps{self.warmup_steps}, max_epochs{self.max_epochs}, "
                                 f"warmup_epochs{self.warmup_epochs}")

        else:
            if self.max_epochs is None or self.warmup_epochs is None:
                raise ValueError('when using "cosine" lr_strategy, max_epochs and warmup_epochs cannot be None')
            lr_scheduler = {
                'scheduler': CosineAnnealingWarmupRestarts(
                    warmup_epochs=self.warmup_epochs,
                    optimizer=optimizer,
                    T_0=int(self.max_epochs / (self.num_restarts + 1)),
                    eta_min=self.lr * 1e-3
                ),
                'interval': 'epoch',
                'frequency': 1
            }

        return [optimizer], [lr_scheduler]


class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    """
    copied from https://github.com/sgvaze/osr_closed_set_all_you_need/blob/main/utils/schedulers.py#L86
    """
    def __init__(self, warmup_epochs, *args, **kwargs):

        super(CosineAnnealingWarmupRestarts, self).__init__(*args, **kwargs)

        # Init optimizer with low learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min

        self.warmup_epochs = warmup_epochs

        # Get target LR after warmup is complete
        target_lr = self.eta_min + (self.base_lrs[0] - self.eta_min) * (1 + math.cos(math.pi * warmup_epochs / self.T_i)) / 2

        # Linearly interpolate between minimum lr and target_lr
        linear_step = (target_lr - self.eta_min) / self.warmup_epochs
        self.warmup_lrs = [self.eta_min + linear_step * (n + 1) for n in range(warmup_epochs)]

    def step(self, epoch=None):

        # Called on super class init
        if epoch is None:
            super(CosineAnnealingWarmupRestarts, self).step(epoch=epoch)

        else:
            if epoch < self.warmup_epochs:
                lr = self.warmup_lrs[epoch]
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # Fulfill misc super() funcs
                self.last_epoch = math.floor(epoch)
                self.T_cur = epoch
                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

            else:

                super(CosineAnnealingWarmupRestarts, self).step(epoch=epoch)