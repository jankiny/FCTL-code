import argparse
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from src.utils import process_config, parse_known_classes, load_yaml, colorstr


def get_train_config(stage = 1):
    parser = argparse.ArgumentParser("FCTL Train/Fine-tune")

    # log
    parser.add_argument("--tensorboard", nargs='?', const=True, default=True,
                        help='flag of logging to tensorboard')
    parser.add_argument("--wandb", nargs='?', const=True, default=False,
                        choices=[None, 'offline'],
                        help='flag of logging to wandb; use "offline" for offline mode')

    # base args
    parser.add_argument("--exp-name", type=str, default="ft", help="experiment name")
    parser.add_argument("--cfg", type=str, help="experiment name")
    parser.add_argument("--random-seed", type=int, default=2023,
                        help="random seed for choosing the training classes")
    parser.add_argument("--n-gpu", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--model-arch", type=str, required=True,
                        help='model setting to use',
                        choices=['vitc16', 'vitb16', 'swin', 'vgg32', 'cnnr101'])
                        # choices=['vitb16', 'swin', 'cnnvar', 'cnnr50', 'cnnr101'])
    parser.add_argument("--image-size", type=int, default=384,
                        help="input image size",
                        choices=[128, 160, 224, 384, 448])
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="model checkpoint to load weights")
    # data
    parser.add_argument("--data-dir", type=str, default='./data', help='data folder')
    parser.add_argument("--dataset", type=str, required=True,
                        help="dataset for fine-tunning/evaluation")
    parser.add_argument("--num-classes", type=int, default=None, help="number of classes in dataset")
    # parser.add_argument("--known-classes", nargs='+', type=int, default=None,
    #                     help="list of known classes, e.g. 2 3 4 5 6 7")
    parser.add_argument("--known-classes", type=parse_known_classes, default=None,
                        help="list of known classes, e.g. '[2, 3, 4, 5, 6, 7]'")
    parser.add_argument("--osr-dataset", type=str, default=None,
                        help="dataset for osr fine-tunning/evaluation")
    parser.add_argument("--osr-classes", type=parse_known_classes, default=None,
                        help="list of known classes, e.g. '[2, 3, 4, 5, 6, 7]'")
    # train
    parser.add_argument("--batch-size", type=int, default=None, help="batch size")
    parser.add_argument("--num-workers-pgpu", type=int, default=12,
                        help="number of workers for per gpu, i.e. total_num_workers = pgpu_num_workers * n-gpu")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="number of training/fine-tunning max epochs")
    parser.add_argument("--warmup-epochs", type=int, default=None, help='learning rate warm up epochs')
    parser.add_argument("--train-steps", type=int, default=None,
                        help="number of training/fine-tunning max steps")
    parser.add_argument("--warmup-steps", type=int, default=None, help='learning rate warm up steps')
    parser.add_argument("--lr", type=float, default=None, help="learning rate")
    parser.add_argument("--lr-strategy", type=str, default=None, help="learning rate strategy",
                        choices=['cosine', 'one_cycle'])
    parser.add_argument("--num-restarts", type=int, default=None, help="number of restarts in cosine lr strategy")
    parser.add_argument("--wd", type=float, default=None, help='weight decay')
    parser.add_argument('--opt', default=None, type=str, choices=('AdamW', 'SGD'))
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument("--label_smoothing", type=float, default=None, help='label smoothing')
    parser.add_argument('--transform', type=str, default='default', help='using transform type, e.g., rand-augment, default')
    parser.add_argument("--resume", nargs='?', default=False, help='Resume training from checkpoint')
    parser.add_argument('--eval', action='store_true', help='evaluate on dataset')

    # similarity learning
    parser.add_argument('--sim-block', type=str, default=None,
                        help="similarity block type",
                        choices=['attention', 'mlp'])
    parser.add_argument('--num-sim-block', type=int, default=None, help='number of blocks in similarity layer')
    parser.add_argument('--sim-task', type=str, default=None,
                        help="task type of similarity learning",
                        choices=['cls', 'reg'])

    # visualization
    parser.add_argument('--plot-model', default=False, action='store_true',
                        help='plot model in tensorboard')
    parser.add_argument('--plot-confusion-matrix', default=False, action='store_true',
                        help='plot confusion_matrix in tensorboard')
    parser.add_argument('--plot-feature-embedding', default=False, action='store_true',
                        help='plot Feature Embedding in tensorboard')
    parser.add_argument('--plot-roc', default=False, action='store_true',
                        help='plot confusion_matrix in tensorboard')

    def add_stage1_args():
        # metric learning loss
        # parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
        # parser.add_argument('--num-instances', type=int, default=8,
        #                     help="number of instances per identity")
        # parser.add_argument('--htri-only', action='store_true', default=False,
        #                     help="if this is True, only htri loss is used in training")

        # * Mixup params
        parser.add_argument('--metric-monitor', type=str, default=None,
                            help="metric monitor for saving best model checkpoint",
                            choices=['train_loss', 'val_loss', 'val_sim_f1'])
        parser.add_argument('--mixup', type=float, default=None,
                            help='mixup alpha, mixup enabled if > 0. (default: 1.0)')

    def add_stage2_args():
        # similarity learning
        parser.add_argument('--sim-alpha', type=float, default=None, help="rate for similarity loss")
        parser.add_argument('--sim-lambda', type=float, default=None, help="rate for similarity score")
        parser.add_argument('--frozen-encoder', default=False, action='store_true', help="frozen feature encoder")
        parser.add_argument('--class-anchor', type=str, default=None,
                            help="class anchor type",
                            choices=['none', 'mean'])
        # fusion
        parser.add_argument('--anchor-freq', type=int, default=None, help='calculate anchor feature frequency')
        parser.add_argument('--fusion', type=str, default=None,
                            help="fusion type",
                            choices=['concat', 'residual', 'cross-attention', 'cross-attention_concat', ])

    if stage == 1:
        add_stage1_args()
    elif stage == 2:
        add_stage2_args()

    config = parser.parse_args()

    # read training config
    try:
        yaml_params = load_yaml(config.cfg)
    except FileNotFoundError:
        raise FileNotFoundError(f'YAML file {config.cfg} not exist')
    def flatten_params(params):
        flat_params = {}
        for key, subdict in params.items():
            if isinstance(subdict, dict):
                flat_params.update(subdict)
            else:
                flat_params[key] = subdict
        return flat_params

    from_args = []
    stage_params = flatten_params(yaml_params.get(f'stage{stage}', {}))
    for key, value in stage_params.items():
        if getattr(config, key) is None:
            setattr(config, key, value)
        else:
            from_args.append(key)

    # get model config
    # model_yaml = f'src/model/{config.model_arch}.yaml'
    # try:
    #     model_params = load_yaml(model_yaml)
    # except FileNotFoundError:
    #     raise FileNotFoundError(f'YAML file {model_yaml} not exist')
    config.model_config = eval("{}Config".format(config.model_arch))(config)

    # lightning logger
    # Two tf log folders can be generated,
    # which are necessary for the experiment
    # and can be specified by --tensorboard for real-time monitoring
    # lightning_loggers = [
    #     TensorBoardLogger(
    #         save_dir=config.tensorboard_dir,
    #         # save_dir="tb_logs",
    #         name=config.exp_name
    #     )
    # ]
    lightning_loggers = []
    if config.wandb:
        import wandb
        wandb.require("core")
        # Wandb Logger
        wandb_logger = WandbLogger(
            # project=f"FCTL-stage{stage}-{config.model_arch}",
            project=f"FCTL-stage{stage}",
            # mode="offline",
            offline=(config.wandb == 'offline'),
            name=config.exp_name,
            # save_dir=config.result_dir,
            save_dir=".",
            log_model=False,
            settings=wandb.Settings(code_dir=".")
        )
        # wandb_logger.watch(model, log_graph=True)
        if stage == 1:
            wandb_logger.experiment.config.update({
                'metric_monitor': config.metric_monitor
            })
        lightning_loggers.append(wandb_logger)

    process_config(config, loggers=lightning_loggers)
    print_config(config, from_args)
    return config, lightning_loggers


def print_config(config, from_args=None):
    message = ''
    message += ' ----------------- Config ---------------\n'
    for k, v in sorted(vars(config).items()):
        comment = 'from args' if from_args is not None and k in from_args else ''
        prefix = '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += colorstr('green', 'bold', prefix) if from_args is not None and k in from_args else prefix
    message += ' ----------------- End -------------------'
    print(message)


class ModelConfig(object):
    def __str__(self):
        return f'{self.__dict__}'

class vitc16Config(ModelConfig):
    def __init__(self, config):
        """ ViT-Custom/16 configuration """
        self.sim_block = config.sim_block
        self.num_sim_block = config.num_sim_block
        self.sim_task = config.sim_task
        self.patch_size = 16
        self.emb_dim = 768
        # using in attention block
        self.mlp_dim = 3072
        self.num_heads = 12
        self.sim_num_heads = 12
        self.num_layers = 4
        self.attn_dropout_rate = 0.0
        self.dropout_rate = 0.1

class vitb16Config(ModelConfig):
    def __init__(self, config):
        """ ViT-B/16 configuration """
        self.sim_block = config.sim_block
        self.num_sim_block = config.num_sim_block
        self.sim_task = config.sim_task
        self.patch_size = 16
        self.emb_dim = 768
        # using in attention block
        self.mlp_dim = 3072
        self.num_heads = 12
        self.sim_num_heads = 12
        self.num_layers = 12
        self.attn_dropout_rate = 0.0
        self.dropout_rate = 0.1


class swinConfig(ModelConfig):
    def __init__(self, config):
        self.sim_block = config.sim_block
        self.num_sim_block = config.num_sim_block
        self.sim_task = config.sim_task
        self.patch_size = 4
        self.in_chans = 3
        self.emb_dim = 128
        self.mlp_dim = 1024
        self.depths = [2, 2, 18, 2]
        self.num_heads = [4, 8, 16, 32]
        self.sim_num_heads = 12
        self.window_size = 7
        self.mlp_ratio = 4.
        self.qkv_bias = True
        self.qk_scale = None
        self.dropout_rate = 0.
        self.attn_dropout_rate = 0.
        self.dropout_path_rate = 0.1
        self.ape = False
        self.patch_norm = True
        self.use_checkpoint = False

class vgg32Config(ModelConfig):
    def __init__(self, config):
        self.sim_block = config.sim_block
        self.num_sim_block = config.num_sim_block
        self.sim_task = config.sim_task
        self.emb_dim = 128
        # using in attention block
        self.mlp_dim = 1024
        self.sim_num_heads = 12
        self.num_layers = 12
        self.attn_dropout_rate = 0.0
        self.dropout_rate = 0.1


class cnnr50Config(ModelConfig):
    def __init__(self, config):
        """ CNN ResNet50 configuration """
        self.sim_block = config.sim_block
        self.num_sim_block = config.num_sim_block
        self.sim_task = config.sim_task
        self.emb_dim = 768
        # using in attention block
        self.mlp_dim = 3072
        self.sim_num_heads = 12
        self.num_heads = 12
        self.num_layers = 12
        self.attn_dropout_rate = 0.0
        self.dropout_rate = 0.1


class cnnr101Config(ModelConfig):
    def __init__(self, config):
        """ CNN ResNet101 configuration """
        self.sim_block = config.sim_block
        self.num_sim_block = config.num_sim_block
        self.sim_task = config.sim_task
        self.emb_dim = 768
        # using in attention block
        self.mlp_dim = 3072
        self.sim_num_heads = 12
        self.num_heads = 12
        self.num_layers = 12
        self.attn_dropout_rate = 0.0
        self.dropout_rate = 0.1
