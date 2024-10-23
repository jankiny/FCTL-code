import json
from pathlib import Path
from datetime import datetime
import os
import sys
import logging
import argparse
import yaml
import shutil

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger



def log(msg=''):
    if len(logging.getLogger().handlers) == 0:
        print(str(msg))
    else:
        logging.getLogger().info(str(msg))


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def clean_dir(dirname):
    if os.path.exists(dirname) and os.listdir(dirname):
        # 清空文件夹
        for filename in os.listdir(dirname):
            file_path = os.path.join(dirname, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除目录
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


def make_symlink(src, dst, target_is_directory=True):
    # Convert to absolute path
    src_abs = os.path.abspath(src)
    dst_abs = os.path.abspath(dst)


    if os.path.islink(src_abs) or os.path.exists(src_abs):
        os.remove(src_abs)  # if exists, deleting old symlink or folder
    os.symlink(dst_abs, src_abs, target_is_directory=target_is_directory)
    print(f"Created symlink: {src_abs} -> {dst_abs}")


def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


def write_json(content, fname):
    def default_serializer(obj):
        # 尝试返回对象的 __dict__
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            print(obj)
            # 如果对象没有 __dict__ 属性，抛出TypeError
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False, default=default_serializer)


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def process_config(config, exp_type='train', loggers=None):
    # train or eval
    # exp_type = 'train' if not config.eval else 'eval'

    # experiments dir Path(opt.project) / opt.name
    save_dir = os.path.join('runs', exp_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_dir = str(increment_path(Path(save_dir) / f'{config.exp_name}_exp'))

    # add datetime postfix
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    exp_name = config.exp_name + '_{}_{}_bs{}_lr{}_wd{}_nc{}_rs{}'.format(config.dataset, config.model_arch, config.batch_size, config.lr, config.wd, config.num_classes, config.random_seed)
    exp_name += ('_' + timestamp)

    # create some important directories to be used for that experiments
    config.checkpoint_dir = os.path.join(save_dir, 'ckpts/')
    config.embedding_dir = os.path.join(save_dir, 'embeddings/')
    config.lightning_dir = os.path.join(save_dir, 'lightning_logs/')
    config.result_dir = save_dir
    for dir in [config.checkpoint_dir, config.embedding_dir, config.lightning_dir, config.result_dir]:
        ensure_dir(dir)

    # tensorboard relates to save_dir
    if config.eval is not True:
        if config.tensorboard:
            loggers.append(
            TensorBoardLogger(
                save_dir=config.tensorboard if isinstance(config.tensorboard, str) else 'tb_logs',
                name='FCTL',
                version=os.path.basename(config.result_dir)
            ))

        # create soft links to the log folders for that experiments
        config.wandb = os.path.join(save_dir, 'wandb')
        config.tensorboard_dir = os.path.join(save_dir, 'tb')
        for logger in loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.config.update({
                    'result_dir': config.result_dir,
                })
                wandb_log_dir = os.path.join(logger.experiment.dir, os.pardir)
                make_symlink(os.path.abspath(config.wandb), wandb_log_dir)
            elif isinstance(logger, TensorBoardLogger):
                tensorboard_log_dir = logger.log_dir
                clean_dir(tensorboard_log_dir)
                make_symlink(config.tensorboard_dir, tensorboard_log_dir)

    # init logger
    log_file_name = f'{exp_name}.log'
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    handler = logging.FileHandler(os.path.join(save_dir, log_file_name))
    handler.setFormatter(logging.Formatter('%(asctime)s: |%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # save config
    write_json(vars(config), os.path.join(save_dir, 'config.json'))

    # show
    log(' *************************************** ')
    log(' The experiment name is {} '.format(config.exp_name))
    log(' *************************************** ')

    return config


def write_result(keys, vals, epoch, save_dir):
    keys = list(keys)
    vals = list(vals)

    x = dict(zip(keys, vals))
    file = Path(save_dir) / 'result.csv'
    n = len(x) + 1  # number of cols
    s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # add header
    with open(file, 'a') as f:
        f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')


def parse_known_classes(s):
    # 使用json.loads解析输入字符串为Python列表
    try:
        if 'all' in s:
            num = int(s.split('_')[-1])
            return list(range(num))
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Could not parse '--known-classes' argument: {s}. Error: {e}")

