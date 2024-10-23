import argparse
import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader

from src.data import *
from src.config import print_config
from src.utils import parse_known_classes
from train_fctl_2 import FCTLStage2
from test_fctl import FCTLTester
from src.model.simnet_2s_mixup_ad import (make_similarities, pair_enumeration, pair_reshape)
from src.utils import (compute_oscr)

class FCTLOODTester(FCTLTester):
    def test_dataloader(self, ood_dataset='ImageNetCrop'):
        # Loading the first validation dataset
        valid_dataset_in = eval("get{}Dataset".format(self.dataset))(
            image_size=self.image_size,
            split='in_test',
            data_path=self.data_dir,
            known_classes=self.known_classes
        )
        valid_dataloader_in = DataLoader(
            valid_dataset_in,
            batch_size=self.batch_size // 2,
            shuffle=False,
            num_workers=self.num_workers
        )

        # Loading the second validation dataset（OOD dataset）
        valid_dataset_out = eval("get{}Dataset".format(ood_dataset))(
            image_size=self.image_size,
            split='in_test',
            data_path=self.data_dir,
            known_classes=list(range(200)) if 'ImageNet' in ood_dataset else list(range(10))
        )
        valid_dataloader_out = DataLoader(
            valid_dataset_out,
            batch_size=self.batch_size // 2,
            shuffle=False,
            num_workers=self.num_workers
        )

        return [valid_dataloader_in, valid_dataloader_out]

def get_test_config():
    parser = argparse.ArgumentParser("FCTL OOD Test")
    parser.add_argument("--checkpoint-path", type=str,
                        default=None,
                        help="model checkpoint to load weights")
    parser.add_argument("--random-seed", type=int, default=2023,
                        help="random seed for choosing the training classes")
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
    model = FCTLOODTester.load_from_checkpoint(config.checkpoint_path)

    trainer = pl.Trainer(
        devices=1,
        accelerator='gpu',
    )

    ood_datasets = ['ImageNetCrop', 'ImageNetResize', 'LSUNCrop', 'LSUNResize']
    model.on_all_test_start()
    for ood_dataset in ood_datasets:
        print(f'----------------- {ood_dataset} -------------------')
        trainer.test(model=model, dataloaders=model.test_dataloader(ood_dataset=ood_dataset))
        print(f'-------------- {ood_dataset} - END ----------------')



