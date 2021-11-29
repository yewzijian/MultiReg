"""
Misc utilities
"""

import argparse
from datetime import datetime
import logging
import os
import shutil
import subprocess
import sys

import coloredlogs
import git
import numpy as np


_logger = logging.getLogger()


def print_info(opt, log_dir=None):
    """ Logs source code configuration
    """
    _logger.info('Command: {}'.format(' '.join(sys.argv)))

    # Print commit ID
    try:
        repo = git.Repo(search_parent_directories=True)
        git_sha = repo.head.object.hexsha
        git_date = datetime.fromtimestamp(repo.head.object.committed_date).strftime('%Y-%m-%d')
        git_message = repo.head.object.message
        _logger.info('Source is from Commit {} ({}): {}'.format(git_sha[:8], git_date, git_message.strip()))

        # Also create diff file in the log directory
        if log_dir is not None:
            with open(os.path.join(log_dir, 'compareHead.diff'), 'w') as fid:
                subprocess.run(['git', 'diff'], stdout=fid)

    except git.exc.InvalidGitRepositoryError:
        pass

    # Arguments
    arg_str = ['{}: {}'.format(key, value) for key, value in vars(opt).items()]
    arg_str = ', '.join(arg_str)
    _logger.info('Arguments: {}'.format(arg_str))


def prepare_logger(config: dict, output_to_file=True):
    """Creates logging directory, and installs colorlogs

    Args:
        config (dict): Program configuration, should include 'log_path' field.
        output_to_file (bool): Whether to write log to file also

    Returns:
        logger (logging.Logger)
    """

    fmt = '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
    datefmt = '%m/%d %H:%M:%S'

    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger, fmt=fmt, datefmt=datefmt)
    if output_to_file:
        log_path = config['log_path']
        os.makedirs(log_path, exist_ok=True)
        log_formatter = logging.Formatter(fmt, datefmt=datefmt)
        file_handler = logging.FileHandler('{}/log.txt'.format(log_path))
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
        logger.info('Output and logs will be saved to {}'.format(log_path))
        print_info(config, log_path)
    else:
        print_info(config)

    return logger


class ReservoirSampler(object):
    def __init__(self, max_size):
        self._size = max_size
        self._all_data = None
        self._n = 0

    def update(self, *args):

        items = args
        if self._all_data is None:
            self._all_data = [[] for _ in range(len(items))]
        elif len(args) != len(self._all_data):
            raise AssertionError('Number of items must be consistent with previous calls')

        k = list(items[0].keys())[0]
        batch_size = items[0][k].shape[0]

        # Save images from a random data batch using reservoir sampling
        for b in range(batch_size):
            self._n += 1
            if self._n <= self._size:
                for i in range(len(items)):
                    self._all_data[i].append({k: items[i][k][b] for k in items[i]})
            else:
                r = np.random.randint(self._n)
                if r < self._size:
                    for i in range(len(items)):
                        self._all_data[i][r] = {k: items[i][k][b] for k in items[i]}

    def get_samples(self):

        samples = []
        for data in self._all_data:
            samples.append({k: [d[k] for d in data] for k in data[0].keys()})

        return samples