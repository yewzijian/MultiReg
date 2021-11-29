import logging
from typing import Dict

import torchvision

import data_loader.graphs as graphs
from data_loader.transforms import CorruptEdges, Perturb

_logger = logging.getLogger()


def get_dataset(config: Dict, phase):
    """Gets the dataset

    Args:
        config: Program arguments
        phase: One of ['train', 'val', 'test']

    Returns:
        torch.utils.data.Dataset instance

    """
    assert phase in ['train', 'val', 'test']

    transforms = []
    if phase == 'train':
        transforms = [CorruptEdges(p=0.2),
                      Perturb(),]
    else:
        pass  # No augmentation

    transform = torchvision.transforms.Compose(transforms)

    # Datasets
    if config['dataset'] == 'neurora':
        dataset = graphs.NeuroraSynthetic(config, phase=phase, transform=transform)
    elif config['dataset'] == '3dmatch':
        if phase == 'train':
            dataset = graphs.ThreeDMatch(config, phase=phase, transform=transform)
        else:
            # Following LMPR, we train on 3DMatch and evaluate on ScanNet
            dataset = graphs.Scannet(config, phase=phase, transform=transform)
    elif config['dataset'] == 'scannet':
        dataset = graphs.Scannet(config, phase=phase, transform=transform)
    else:
        raise ValueError('Invalid dataset: {}'.format(config['dataset']))

    _logger.info('Transform type: {}'.format(dataset.transform_type))
    return dataset
