"""Inference code"""
import torch
from torch_geometric.data import DataLoader

from arguments_graph import get_config
from common.misc import prepare_logger
from common.torch_helpers import seed_numpy_fn
from data_loader import get_dataset
from models import get_model
from trainer import Trainer

# Set up arguments and logging.
config = get_config(phase='test')
logger = prepare_logger(config)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def main():

    dataset = get_dataset(config, phase='test')
    logger.info('Using dataset {} with {} instances'.format(
        config.dataset, len(dataset)))

    # Model
    if config.resume is None:
        logger.error('Weights not supplied during inference. Exiting...')
        exit(-1)

    Net = get_model(config.model)
    model = Net(transform_type=dataset.transform_type, config=config)
    model.to(device)

    test_loader = DataLoader(dataset, batch_size=config.val_batch_size,
                             shuffle=False, num_workers=config.num_workers,
                             worker_init_fn=seed_numpy_fn)

    trainer = Trainer(config=config)
    trainer.test(model, test_loader)


if __name__ == '__main__':
    main()
