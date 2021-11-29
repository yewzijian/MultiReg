import logging
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common.torch_helpers import TorchDebugger, CheckPointManager


class Trainer(object):
    """Training helper

    Args:
        config (Dict): Python dictionary containing configuration
    """
    def __init__(self, config, gradient_clip_val=0.0):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.grad_clip = gradient_clip_val if gradient_clip_val > 0.0 else None

        if not torch.cuda.is_available():
            self.logger.warning('Using CPU for training. This can be slow...')

    def train(self, model, train_loader, val_loader=None):
        """Starts training loop"""
        config = self.config

        optimizer = model.configure_optimizers()

        # Summary writer and Checkpoint manager
        saver = CheckPointManager(os.path.join(config.log_path, 'ckpt', 'model'),
                                  max_to_keep=9999,
                                  keep_checkpoint_every_n_hours=0.5)
        train_writer = SummaryWriter(os.path.join(config.log_path, 'train'),
                                     flush_secs=10)
        if val_loader is not None:
            val_writer = SummaryWriter(os.path.join(config.log_path, 'val'),
                                       flush_secs=10)

        if config.resume is not None:
            global_step = init_step = saver.load(config.resume, model, optimizer)
        else:
            global_step = init_step = 0
        torch.autograd.set_detect_anomaly(config.debug)  # for debugging

        steps_per_epoch = len(train_loader)
        if config.summary_every < 0:
            config.summary_every = abs(config.summary_every) * steps_per_epoch
        if config.validate_every < 0:
            config.validate_every = abs(config.validate_every) * steps_per_epoch

        model.train()
        training_complete = False
        epoch = 0
        while not training_complete:
            self.logger.info('Begin epoch {} (steps {} - {})'.format(
                epoch, global_step, global_step + len(train_loader)))
            tbar = tqdm(total=len(train_loader), ncols=80, leave=False)
            total_loss_epoch, num_steps_epoch = 0.0, 0
            t_start = time.time()
            for train_data in train_loader:
                global_step += 1

                # Forward through neural network
                optimizer.zero_grad()
                pred, endpoints = model.training_step(train_data, global_step)

                # Compute loss, and optimize
                train_losses = model.compute_loss(train_data, pred, endpoints)
                if config.debug:
                    with TorchDebugger():
                        train_losses['total'].backward()
                else:
                    train_losses['total'].backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   self.grad_clip)
                if config.validate_every != 0:
                    optimizer.step()

                total_loss_epoch += train_losses['total']
                num_steps_epoch += 1
                tbar.set_description('Loss:{:.3g}'.format(total_loss_epoch / num_steps_epoch))
                tbar.update(1)

                if global_step % config.summary_every == 0:  # Save tensorboard logs
                    model.save_summaries(train_writer, global_step,
                                         model=model,
                                         optimizer=optimizer,
                                         data=train_data, predicted=pred, endpoints=endpoints,
                                         losses=train_losses)

                # Validation loop
                if config.validate_every == 0 or global_step % config.validate_every == 0:
                    if val_loader is not None:
                        # Run validation
                        train_iter = tbar.n
                        t_start_val = time.time()
                        tbar.close()
                        val_score = self._validate(model, val_loader, val_writer, step=global_step)

                        # Save checkpoints and restore progress bar
                        saver.save(model, optimizer, step=global_step, score=val_score)
                        tbar = tqdm(total=len(train_loader), ncols=80, initial=train_iter)
                        t_start += time.time() - t_start_val
                    else:
                        # Just save checkpoint without score
                        saver.save(model, optimizer, step=global_step)

                if config.validate_every == 0:
                    return

                if global_step - init_step > config.max_steps:
                    training_complete = True
                    break

            epoch += 1

            tbar.close()
            self.logger.info('Time taken: {:.2f}, Average train loss: {:.3g}.'.format(
                time.time() - t_start, total_loss_epoch / num_steps_epoch))

        self.logger.info('Ending training. Number of steps = {}.'.format(global_step))

    def _validate(self, model, val_loader, val_writer, step):

        self.logger.info('Starting validation...')
        model.eval()

        with torch.no_grad():
            for val_data in tqdm(val_loader, ncols=80, leave=False):
                model.validation_step(val_data, step)

        val_score = model.validation_epoch_end(val_writer, step)
        model.train()
        return val_score

    def test(self, model, test_loader):
        """Inference over entire dataset"""
        config = self.config

        saver = CheckPointManager()
        total_inference_time = 0.0
        if config.resume is not None:
            global_step = saver.load(config.resume, model)
            self.logger.info('Global step = {}'.format(global_step))

        model.eval()
        with torch.no_grad():
            for data in tqdm(test_loader, ncols=80):
                # Forward through neural network
                t_start = time.time()
                model.test_step(data, global_step)
                total_inference_time += time.time() - t_start

            model.test_epoch_end(global_step)

        self.logger.info('Total inference time: {:.2f}s'.format(total_inference_time))
