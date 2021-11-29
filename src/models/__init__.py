from typing import Type

import torch

import models.PoseGraphNet


def get_model(name: str = 'dummy') -> Type[torch.nn.Module]:
    if name == 'PoseGraphNet':
        return models.PoseGraphNet.PoseGraphNet
    raise NotImplementedError

