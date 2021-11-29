import argparse
from datetime import datetime
import os
import shutil
import sys

from easydict import EasyDict


def generic_arguments():
    """Arguments used for both training and testing"""
    parser = argparse.ArgumentParser(add_help=False)

    # Logging
    parser.add_argument('--logdir', default='../logs', type=str,
                        help='Directory to store logs, summaries, checkpoints.')
    parser.add_argument('--dev', action='store_true', help='If true, will ignore logdir and log to ../logdev instead')
    parser.add_argument('--name', type=str, help='Prefix to add to logging directory')
    parser.add_argument('--debug', action='store_true', help='If set, will enable autograd anomaly detection')
    parser.add_argument('--nodate', action='store_true', help='If set, logging directory will not contain date part')
    # # Training parameters
    parser.add_argument('--train_batch_size', default=16, type=int, metavar='N',
                        help='training mini-batch size (default 16)')
    parser.add_argument('--val_batch_size', default=32, type=int, metavar='N',
                        help='mini-batch size during validation or testing (default: 32)')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers for data_loader loader (default: 4).')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='Pretrained network to load from. Optional for train, required for inference.')
    # Model parameters
    parser.add_argument('--w_abs', default=0.0, type=float,
                        help='Weight for absolute poses')
    parser.add_argument('--w_rel', default=0.2, type=float,
                        help='Weight for relative poses between all pairs')
    parser.add_argument('--w_inliers', default=1.0, type=float,
                        help='Weight for prediction of in/outliers')

    # Dataset configurations, as well as augmentations
    parser.add_argument('--dataset_path', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default='scannet',
                        choices=['neurora', 'rotreal', '3dmatch', 'scannet', 'scannet_l2s'])
    # Network settings
    parser.add_argument('--model', type=str, default='PoseGraphNet')
    parser.add_argument('--bidirectional', type=int, default=0, help='Whether to use bidirectional edges',
                        choices=[0, 1])
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of hidden layer')
    parser.add_argument('--aggr', type=str, default='weighted_sum', help='Node model aggregation scheme',
                        choices=['max', 'mean', 'weighted_sum'])
    parser.add_argument('--num_iter', type=int, default=10,
                        help='Number of iterations for learned optimization.')
    return parser


def train_arguments():
    """Used only for training"""
    parser = argparse.ArgumentParser(parents=[generic_arguments()])

    parser.description = 'Train the network'
    # General parameters
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate during training')
    parser.add_argument('--grad_clip', default=0.05, type=float, help='Gradient clip value (set to 0 to disable)')
    parser.add_argument('--max_steps', default=200000, type=int, metavar='N',
                        help='Number of training steps to run')
    parser.add_argument('--summary_every', default=500, type=int, metavar='N',
                        help='Frequency of saving summary (number of steps if positive, number of epochs if negative)')
    parser.add_argument('--validate_every', default=1000, type=int, metavar='N',
                        help='Frequency of evaluation (number of steps if positive, number of epochs if negative).'
                             'Also saves checkpoints at the same interval')
    return parser


def eval_arguments():
    """Used during evaluation and visualization"""
    parser = argparse.ArgumentParser(parents=[generic_arguments()])

    parser.description = 'Eval the network'
    parser.add_argument('--method', default='PoseGraphNet', help='Method to use.',
                        choices=['PoseGraphNet', 'georeg', 'transsync'],)
    return parser


def get_config(phase='none'):

    if phase == 'train':
        parser = train_arguments()
    elif phase == 'test':
        parser = eval_arguments()
    else:
        raise ValueError('phase should be train or test')

    args = EasyDict(vars(parser.parse_args()))
    args.phase = phase

    # Sets num_workers to 0 if running in a debugger as Pycharm's debugger does
    # not work well with multi-threaded data loaders
    if sys.gettrace() is not None:
        print('Setting args.num_workers to 0 since running in debugger')
        args.num_workers = 0

    # Configure log directory
    if args['dev']:
        log_path = '../logdev'
        shutil.rmtree(log_path, ignore_errors=True)
    elif args['nodate']:
        assert args['name'] is not None, 'When using nodate, --name must be set'
        log_path = os.path.join(args['logdir'], args['dataset'], args['name'])
    else:
        datetime_str = datetime.now().strftime('%y%m%d-%H%M%S')
        if args['name'] is not None:
            log_path = os.path.join(args['logdir'], args['dataset'], datetime_str +
                                    '(' + args['name'] + ')')
        else:
            log_path = os.path.join(args['logdir'], args['dataset'], datetime_str)

    args.log_path = log_path
    return args
