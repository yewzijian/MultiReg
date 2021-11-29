"""PyTorch related utility functions
"""

import logging
import os
import pdb
import shutil
import sys
import time
import traceback
from typing import Union, List

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
import torch_geometric


def all_to_device(data, device):
    """Sends everything into a certain device """
    if isinstance(data, dict):
        for k in data:
            data[k] = all_to_device(data[k], device)
        return data
    elif isinstance(data, list):
        data = [all_to_device(d, device) for d in data]
        return data
    elif isinstance(data, torch_geometric.data.batch.Batch) or isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data  # Cannot be converted


def to_numpy(tensor: Union[np.ndarray, torch.Tensor, List]) -> Union[np.ndarray, List]:
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        return [to_numpy(l) for l in tensor]
    elif isinstance(tensor, str):
        return tensor
    elif tensor is None:
        return None
    else:
        raise NotImplementedError


def seed_numpy_fn(x):
    """Numpy random seeding function to pass into Pytorch's dataloader.

    This is required since numpy RNG is incompatible with fork
    https://pytorch.org/docs/stable/notes/faq.html#my-data-loader-workers-return-identical-random-numbers

    Example usage:
        DataLoader(..., worker_init_fn=seed_numpy_fn)
    """
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)


class CheckPointManager(object):
    """Manager for saving/managing pytorch checkpoints.

    Provides functionality similar to tf.Saver such as
    max_to_keep and keep_checkpoint_every_n_hours
    """
    def __init__(self, save_path: str = None, max_to_keep=5, keep_checkpoint_every_n_hours=10000.0):

        if max_to_keep <= 0:
            raise ValueError('max_to_keep must be at least 1')

        self._max_to_keep = max_to_keep
        self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours

        self._logger = logging.getLogger(self.__class__.__name__)
        self._checkpoints_permanent = []  # Will not be deleted
        self._checkpoints_buffer = []  # Those which might still be deleted
        self._next_save_time = time.time()
        self._best_score = None
        self._best_step = None

        if save_path is not None:
            self._ckpt_dir = os.path.dirname(save_path)
            self._save_path = save_path + '-{}.pth'
            self._checkpoints_fname = os.path.join(self._ckpt_dir, 'checkpoints.txt')
            os.makedirs(self._ckpt_dir, exist_ok=True)
            self._update_checkpoints_file()
        else:
            self._ckpt_dir = None
            self._save_path = None
            self._checkpoints_fname = None

    def _save_checkpoint(self, step, model, optimizer, score):
        save_name = self._save_path.format(step)

        model_state_dict = {k: v for (k, v) in model.state_dict().items() if not v.is_sparse}
        state = {'state_dict': model_state_dict,
                 'optimizer': optimizer.state_dict(),
                 'step': step}
        torch.save(state, save_name)
        self._logger.info('Saved checkpoint: {}'.format(save_name))

        self._checkpoints_buffer.append((save_name, time.time()))

        if self._best_score is None or np.all(np.array(score) >= np.array(self._best_score)):
            best_save_name = self._save_path.format('best')
            shutil.copyfile(save_name, best_save_name)
            self._best_score = score
            self._best_step = step
            self._logger.info('Checkpoint is current best, score={}'.format(
                np.array_str(np.array(self._best_score), precision=3)))

    def _remove_old_checkpoints(self):
        while len(self._checkpoints_buffer) > self._max_to_keep:
            to_remove = self._checkpoints_buffer.pop(0)

            if to_remove[1] > self._next_save_time:
                self._checkpoints_permanent.append(to_remove)
                self._next_save_time = to_remove[1] + self._keep_checkpoint_every_n_hours * 3600
            else:
                os.remove(to_remove[0])

    def _update_checkpoints_file(self):
        checkpoints = [os.path.basename(c[0]) for c in self._checkpoints_permanent + self._checkpoints_buffer]
        with open(self._checkpoints_fname, 'w') as fid:
            fid.write('\n'.join(checkpoints))
            fid.write('\nBest step: {}'.format(self._best_step))

    def save(self, model: torch.nn.Module, optimizer: Optimizer, step: int, score: float = 0.0):
        """Save model checkpoint to file

        Args:
            model: Torch model
            optimizer: Torch optimizer
            step (int): Step, model will be saved as model-[step].pth
            score (float, optional): To determine which model is the best
        """
        if self._save_path is None:
            raise AssertionError('Checkpoint manager must be initialized with save path for save().')

        self._save_checkpoint(step, model, optimizer, score)
        self._remove_old_checkpoints()
        self._update_checkpoints_file()

    def load(self, save_path, model: torch.nn.Module = None, optimizer: Optimizer = None):
        """Loads saved model from file

        Args:
            save_path: Path to saved model (.pth). If a directory is provided instead, model-best.pth is used
            model: Torch model to restore weights to
            optimizer: Optimizer
        """
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, 'model-best.pth')

        state = torch.load(save_path)

        step = 0
        if 'step' in state:
            step = state['step']

        if 'state_dict' in state and model is not None:
            # TODO: Remove hack
            if save_path == 'pretrained/fcgf32_voxel25.pth':
                state['state_dict'] = {'backbone.' + k: state['state_dict'][k]
                                       for k in state['state_dict']}

            retval = model.load_state_dict(state['state_dict'], strict=False)
            if len(retval.unexpected_keys) > 0:
                self._logger.warning('Unexpected keys in checkpoint: {}'.format(
                    retval.unexpected_keys))
            if len(retval.missing_keys) > 0:
                self._logger.warning('Missing keys in checkpoint: {}'.format(
                    retval.missing_keys))

        if 'optimizer' in state and optimizer is not None:
            optimizer.load_state_dict(state['optimizer'])

        self._logger.info('Loaded models from {}'.format(save_path))
        return step


class TorchDebugger(torch.autograd.detect_anomaly):
    """Enters debugger when anomaly detected"""
    def __enter__(self) -> None:
        super().__enter__()

    def __exit__(self, type, value, trace):
        super().__exit__()
        if isinstance(value, RuntimeError):
            traceback.print_tb(trace)
            print(value)
            if sys.gettrace() is None:
                pdb.set_trace()
