"""Training script for Transformation Synchronization network"""

import torch
from torch_geometric.data import DataLoader

from arguments_graph import get_config
from common.misc import prepare_logger
from common.torch_helpers import seed_numpy_fn
from data_loader import get_dataset
from models import get_model
from trainer import Trainer

# Set up arguments and logging.
config = get_config(phase='train')
logger = prepare_logger(config)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def main():

    train_set = get_dataset(config, phase='train')
    val_set = get_dataset(config, phase='val')
    Net = get_model(config['model'])

    # Model
    model = Net(train_set.transform_type, config)
    model.to(device)
    # Scheduler
    config['scheduler'] = None

    logger.info('Using dataset {} with {} training and {} validation instances'.format(
        config.dataset, len(train_set), len(val_set)))
    logger.info('Loss weights: {}'.format(
        {k[2:]: config[k] for k in config if k.startswith('w_')}))

    train_loader = DataLoader(train_set, batch_size=config.train_batch_size,
                              shuffle=True, num_workers=config.num_workers,
                              worker_init_fn=seed_numpy_fn)
    val_loader = DataLoader(val_set, batch_size=config.val_batch_size,
                            shuffle=False, num_workers=config.num_workers)

    trainer = Trainer(config=config, gradient_clip_val=config['grad_clip'])
    trainer.train(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
