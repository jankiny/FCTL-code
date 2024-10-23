import json
import math
from pathlib import Path
from datetime import datetime
import os
import sys
import logging

import torch
import matplotlib
from sklearn import metrics
import matplotlib.pyplot as plt
import importlib
import pandas as pd
import numpy as np


def setup_device(n_gpu_use):
    print(' ----------------- GPUs Setup --------------- ')
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu

    if n_gpu > 0:
        import pynvml
        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()

        gpu_memory_list = []
        for i in reversed(range(num_gpus)):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_list.append((memory_info.free, i))

        # pynvml.nvmlShutdown()
        gpu_memory_list.sort(key=lambda x: x[0], reverse=True)
        list_ids = [gpu_memory_list[i][1] for i in range(n_gpu_use)]
    else:
        list_ids = list(range(n_gpu_use))

    # device = torch.device(f'cuda:{list_ids[0]}' if n_gpu_use > 0 else 'cpu')

    print('{:>35}: {:<30}'.format('n_gpu_use', str(n_gpu_use)))
    print('{:>35}: {:<30}'.format('list_ids', str(list_ids)))
    # print('{:>35}: {:<30}'.format('device', str(device)))
    print(' -------------------- End ------------------- ')
    # return device, list_ids
    return list_ids

class TensorboardWriter():
    def __init__(self, log_dir, enabled):
        self.writer = None
        self.selected_module = ""
        self.lock = False

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                print(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_histogram', 'add_histogram_raw', 'add_image', 'add_images',
            'add_figure', 'add_video', 'add_audio', 'add_text', 'add_pr_curve', 'add_pr_curve_raw', 'add_mesh'
        }
        self.tb_writer_other = {
            'add_hparams', 'add_onnx_graph', 'add_graph', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def get_step(self):
        return self.step, self.mode

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    if name == 'add_embedding':
                        add_data(tag=tag, mat=data, global_step=self.step, *args, **kwargs)
                    elif name == 'add_graph':
                        add_data(model=tag, *args, **kwargs)
                    else:
                        add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        elif name in self.tb_writer_other:
            add_data = getattr(self.writer, name, None)

            def wrapper(*args, **kwargs):
                if add_data is not None:
                    add_data(*args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


def resize_vit_pos_embedding(state_dict, new_img_size=384, patch_size=16, emb_dim=768):
    posemb_key = 'vit.encoder.pos_embedding'
    if posemb_key in state_dict:
        posemb = state_dict[posemb_key]
        posemb_tok, posemb_grid = posemb[:, :1], posemb[:, 1:]

        model_grid_seq = new_img_size // patch_size
        ckpt_grid_seq = int(np.sqrt(posemb_grid.shape[1]))

        if model_grid_seq != ckpt_grid_seq:
            posemb_grid = posemb_grid.reshape(1, ckpt_grid_seq, ckpt_grid_seq, emb_dim).permute(0, 3, 1, 2)
            posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=(model_grid_seq, model_grid_seq), mode='bicubic', align_corners=False)
            posemb_grid = posemb_grid.permute(0, 2, 3, 1).flatten(1, 2)
            posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
            state_dict[posemb_key] = posemb
            print(f'Resized positional embedding from ({ckpt_grid_seq},{ckpt_grid_seq}) to ({model_grid_seq},{model_grid_seq})')
        else:
            print('No resizing needed for positional embeddings.')
    return state_dict


def load_checkpoint(path, new_img=384, patch=16, emb_dim=768,layers=12):
    """ Load weights from a given checkpoint path in npz/pth """
    if path.endswith('npz'):
        keys, values = load_jax(path)
        state_dict = convert_jax_pytorch(keys, values)
    elif path.endswith('pth'):
        if 'deit' in os.path.basename(path):
            state_dict = torch.load(path, map_location=torch.device("cpu"))['model']
        elif 'jx' in path or 'vit' in os.path.basename(path):
            state_dict = torch.load(path, map_location=torch.device("cpu"))
        else:
            state_dict = torch.load(path, map_location=torch.device("cpu"))['state_dict']
    else:
        raise ValueError("checkpoint format {} not supported yet!".format(path.split('.')[-1]))
    if 'encoder' in state_dict:
        return state_dict

    if 'jx' in path or any(x in  os.path.basename(path) for x in ['vit','deit']): # for converting rightmann weight
            old_img = (24,24) #TODO: check if not needed
            # num_layers_model = layers  #
            # num_layers_state_dict = int((len(state_dict) - 8) / 12)
            # if num_layers_model != num_layers_state_dict:
            #     raise ValueError(
            #         f'Pretrained model has different number of layers: {num_layers_state_dict} than defined models layers: {num_layers_model}')
            #state_dict['class_token'] = state_dict.pop('cls_token')
            if 'distilled' in path:
                state_dict['distilled_token'] = state_dict.pop('dist_token')
            state_dict['transformer.pos_embedding.pos_embedding'] = state_dict.pop('pos_embed')
            state_dict['embedding.weight'] = state_dict.pop('patch_embed.proj.weight')
            state_dict['embedding.bias'] = state_dict.pop('patch_embed.proj.bias')
            if os.path.basename(path) == 'vit_small_p16_224-15ec54c9.pth' : # hack for vit small
                state_dict['embedding.weight'] = state_dict['embedding.weight'].reshape(768,3, 16,16)
            state_dict['classifier.weight'] = state_dict.pop('head.weight')
            state_dict['classifier.bias'] = state_dict.pop('head.bias')
            state_dict['transformer.norm.weight'] = state_dict.pop('norm.weight')
            state_dict['transformer.norm.bias'] = state_dict.pop('norm.bias')
            posemb = state_dict['transformer.pos_embedding.pos_embedding']
            for i, block_name in enumerate(list(state_dict.keys()).copy()):
                if 'blocks' in block_name:
                    new_block = "transformer.encoder_layers."+block_name.split('.',1)[1]
                    state_dict[new_block]=state_dict.pop(block_name)

    else:
        # resize positional embedding in case of diff image or grid size
        posemb = state_dict['transformer.pos_embedding.pos_embedding']
    # Deal with class token
    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
    model_grid_seq = new_img//patch
    ckpt_grid_seq = int(np.sqrt(posemb_grid.shape[0]))

    if model_grid_seq!=ckpt_grid_seq:
        # Get old and new grid sizes
        posemb_grid = posemb_grid.reshape(ckpt_grid_seq, ckpt_grid_seq, -1)

        posemb_grid = torch.unsqueeze(posemb_grid.permute(2, 0, 1), dim=0)
        posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=(model_grid_seq, model_grid_seq), mode='bicubic', align_corners=False)
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).flatten(1, 2)

        # Deal with class token and return
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        # if 'jx' in path:
        #     state_dict['pos_embed'] = posemb
        # else:
        state_dict['transformer.pos_embedding.pos_embedding'] = posemb
        print('Resized positional embedding from (%d,%d) to (%d,%d)'%(ckpt_grid_seq,ckpt_grid_seq,model_grid_seq,model_grid_seq))
    return state_dict

def load_jax(path):
    """ Loads params from a npz checkpoint previously stored with `save()` in jax implemetation """
    ckpt_dict = np.load(path, allow_pickle=False)
    keys, values = zip(*list(ckpt_dict.items()))
    # with gfile.GFile(path, 'rb') as f:
    #     ckpt_dict = np.load(f, allow_pickle=False)
    #     keys, values = zip(*list(ckpt_dict.items()))
    return keys, values

def convert_jax_pytorch(keys, values):
    """ Convert jax model parameters with pytorch model parameters """
    state_dict = {}
    for key, value in zip(keys, values):

        # convert name to torch names
        names = key.split('/')
        torch_names = replace_names(names)
        torch_key = '.'.join(w for w in torch_names)

        # convert values to tensor and check shapes
        tensor_value = torch.tensor(value, dtype=torch.float)
        # check shape
        num_dim = len(tensor_value.shape)

        if num_dim == 1:
            tensor_value = tensor_value.squeeze()
        elif num_dim == 2 and torch_names[-1] == 'weight':
            # for normal weight, transpose it
            tensor_value = tensor_value.T
        elif num_dim == 3 and torch_names[-1] == 'weight' and torch_names[-2] in ['query', 'key', 'value']:
            feat_dim, num_heads, head_dim = tensor_value.shape
            # for multi head attention q/k/v weight
            tensor_value = tensor_value
        elif num_dim == 2 and torch_names[-1] == 'bias' and torch_names[-2] in ['query', 'key', 'value']:
            # for multi head attention q/k/v bias
            tensor_value = tensor_value
        elif num_dim == 3 and torch_names[-1] == 'weight' and torch_names[-2] == 'out':
            # for multi head attention out weight
            tensor_value = tensor_value
        elif num_dim == 4 and torch_names[-1] == 'weight':
            tensor_value = tensor_value.permute(3, 2, 0, 1)

        # print("{}: {}".format(torch_key, tensor_value.shape))
        state_dict[torch_key] = tensor_value
    return state_dict


def replace_names(names):
    """ Replace jax model names with pytorch model names """
    new_names = []
    for name in names:
        if name == 'Transformer':
            new_names.append('transformer')
        elif name == 'encoder_norm':
            new_names.append('norm')
        elif 'encoderblock' in name:
            num = name.split('_')[-1]
            new_names.append('encoder_layers')
            new_names.append(num)
        elif 'LayerNorm' in name:
            num = name.split('_')[-1]
            if num == '0':
                new_names.append('norm{}'.format(1))
            elif num == '2':
                new_names.append('norm{}'.format(2))
        elif 'MlpBlock' in name:
            new_names.append('mlp')
        elif 'Dense' in name:
            num = name.split('_')[-1]
            new_names.append('fc{}'.format(int(num) + 1))
        elif 'MultiHeadDotProductAttention' in name:
            new_names.append('attn')
        elif name == 'kernel' or name == 'scale':
            new_names.append('weight')
        elif name == 'bias':
            new_names.append(name)
        elif name == 'posembed_input':
            new_names.append('pos_embedding')
        elif name == 'pos_embedding':
            new_names.append('pos_embedding')
        elif name == 'embedding':
            new_names.append('embedding')
        elif name == 'head':
            new_names.append('classifier')
        elif name == 'cls':
            new_names.append('cls_token')
        else:
            new_names.append(name)
    return new_names
