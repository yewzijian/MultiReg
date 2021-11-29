"""Generic implementation for transformation synchronization network,
which implements the evaluation code
"""
import logging
import os
from abc import ABC
from typing import Type, Dict

import h5py
import numpy as np
import torch.nn
from scipy.io import savemat
from torch.utils.tensorboard import SummaryWriter

from common.lie.torch import LieGroupBase, SO3
from common.torch_helpers import to_numpy
from metrics import print_loss_metrics, Accumulator,  compute_metrics, get_aggregated_metrics


def eval_is_relative(dataset, phase):
    """Returns True if evaluation should be done using relative poses
       (NeuRoRA evaluation and 1DSfM datasets).
       False otherwise (ScanNet and NeuRoRA during training)
    """
    if dataset in ['scannet', '3dmatch']:
        return True
    elif dataset == 'neurora' and phase == 'train':
        return True
    elif (dataset == 'neurora' and phase == 'test') or dataset == 'rotreal':
        return False
    else:
        raise AssertionError('Invalid values')


def eval_only_good(dataset, phase):
    """LMPR evaluation uses only"""
    return dataset == 'scannet' and phase == 'test'


class PoseGraphNet_Generic(torch.nn.Module, ABC):

    def __init__(self, transform_type: Type[LieGroupBase], config: Dict):
        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info('Creating model {}'.format(self.__class__.__name__))
        self.Transform = transform_type
        self.config = config

        self.num_iter = config['num_iter']  # Number of message passing iterations
        self._device = None

        # For storing values for validation/testing
        self.loss_accumulator = Accumulator()
        self.metrics_accumulator = Accumulator()
        self.all_endpoints = {}
        self.lr_scheduler = None
        # For testing
        self.num_graphs = 0
        self.num_edges = 0
        self.hf = None

    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def configure_optimizers(self):
        # optimizer
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.config.lr)
        self._logger.info('Using {} optimizer with learning rate {}'.format(type(optimizer), self.config.lr))
        if 'scheduler' in self.config and self.config['scheduler'] is not None:
            self.lr_scheduler = self.config['scheduler'](optimizer, **self.config['scheduler_opt'])
            self._logger.info(
                'Using learning rate scheduler {} with options: {}'.format(
                    type(self.lr_scheduler), self.config['scheduler_opt']))
        return optimizer

    def forward(self, data):
        """Forward pass through the network.

        Args:
            data: Provided view-graph

        Returns:
            T_cw_pred: predicted absolute pose which transforms world points to camera points
            endpoints: Other outputs of the network
        """
        raise NotImplementedError

    def training_step(self, train_data, step: int):

        pred, endpoints = self.forward(train_data)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return pred, endpoints

    """
    Validation code
    """
    def validation_step(self, val_data, step: int):
        """Perform a single validation run, and saves results into tensorboard summaries"""

        pred, endpoints = self.forward(val_data)

        val_losses = self.compute_loss(val_data, pred, endpoints)
        metrics = compute_metrics(val_data, pred,
                                  is_rel=eval_is_relative(self.config.dataset, self.config.phase),
                                  only_good=eval_only_good(self.config.dataset, self.config.phase))
        self.loss_accumulator.update(val_losses)
        self.metrics_accumulator.update(metrics)

        # Stores endpoints for tensorboard summaries
        endpoints_to_store = ['weight_logits', 'theta_update']
        for e in endpoints_to_store:
            if e not in endpoints:
                continue
            if e in self.all_endpoints:
                for i in range(len(endpoints[e])):
                    self.all_endpoints[e][i].append(endpoints[e][i])
            else:
                self.all_endpoints[e] = [[endpoints[e][i]] for i in range(self.num_iter)]

    def validation_epoch_end(self, summary_writer: SummaryWriter, step: int):
        all_losses = self.loss_accumulator.get_aggregated()
        all_metrics = self.metrics_accumulator.get_aggregated()

        # Average the losses from all the steps
        all_losses = {k: torch.mean(all_losses[k]) for k in all_losses}

        # Compute aggregated metrics
        all_metrics, input_stats = get_aggregated_metrics(all_metrics,
                                                          eval_is_relative(self.config.dataset, self.config.phase))

        if 'weight_logits' in self.all_endpoints:
            self.all_endpoints['weight_logits'] = [torch.cat(weight_logits, dim=0)
                                              for weight_logits in self.all_endpoints['weight_logits']]
        if 'theta_update' in self.all_endpoints:
            self.all_endpoints['theta_update'] = [torch.cat(theta_update, dim=0)
                                             for theta_update in self.all_endpoints['theta_update']]

        print_loss_metrics(self._logger, step, all_losses, all_metrics, input_stats=input_stats)
        self.save_summaries(summary_writer, step, endpoints=self.all_endpoints,
                            losses=all_losses, metrics=all_metrics, input_stats=input_stats)

        # Use median error as the metric to select best checkpoint, except for 1DSfM datasets, which we
        # use the mean error
        if 'trans_error_med' in all_metrics:
            main_metric = -(0.1 * all_metrics['rot_error_med'] + all_metrics['trans_error_med']).item()
        else:
            if self.config.dataset == 'rotreal':
                main_metric = -all_metrics['rot_error_mean'].item()
            else:
                main_metric = -all_metrics['rot_error_med'].item()

        # Reset accumulated statistics
        self.loss_accumulator.reset()
        self.metrics_accumulator.reset()
        self.all_endpoints = {}

        return main_metric

    """
    Test code
    """
    def test_step(self, test_data, step: int):

        pred, endpoints = self.forward(test_data)

        test_losses = self.compute_loss(test_data, pred, endpoints)
        test_metrics = compute_metrics(test_data, pred,
                                       is_rel=eval_is_relative(self.config.dataset, self.config.phase),
                                       only_good=eval_only_good(self.config.dataset, self.config.phase))
        batch_size = test_data.num_graphs
        self.loss_accumulator.update(test_losses)
        self.metrics_accumulator.update(test_metrics)

        save_folder = os.path.join(self.config['log_path'], 'computed_transforms')
        os.makedirs(save_folder, exist_ok=True)
        data_list, pred_list, endpoints_list = self._split_data_batch(test_data, pred, endpoints)
        self.num_edges += test_data.tij_hat.shape[0]

        # Save out predictions for further analysis or evaluation.
        for b in range(batch_size):
            if self.config['dataset'] in ['scannet', 'scannet_l2s']:
                # Frame order might be randomized, so we re-sort them back for evaluation
                num_steps = len(pred_list[b])
                num_nodes = data_list[b]['t_star'].shape[0]
                frame_idx = np.array([source[1] for source in data_list[b]['source_clouds']])
                idxsort = np.argsort(frame_idx)

                t_star = to_numpy(data_list[b]['t_star'].as_matrix())[idxsort]
                t_pred = to_numpy(pred_list[b][-1])[idxsort]
                t_pred_iter = np.array(
                    [to_numpy(pred_list[b][i][idxsort]) for i in range(num_steps)])

                # Save out predicted inliers also if available
                edge_weights_2d = None
                if endpoints_list[b]['weight_logits'] is not None:
                    edge_weights = to_numpy(
                        torch.sigmoid(endpoints_list[b]['weight_logits']))  # (num_steps, 2, num_edges)
                    edge_weights_2d = np.full([num_nodes, num_nodes], np.nan)
                    np.fill_diagonal(edge_weights_2d, 1.0)  # diagonals are all inliers by definition
                    edge_weights_2d = np.tile(edge_weights_2d[None, ...], (num_steps, 1, 1))
                    edge_index = frame_idx[
                        to_numpy(data_list[b]['edge_index'])]  # edge_index with the correct frame index

                    for s in range(num_steps):
                        edge_weights_2d[s, edge_index[0], edge_index[1]] = edge_weights[s, 0, :]
                        edge_weights_2d[s, edge_index[1], edge_index[0]] = edge_weights[s, 1, :]
                features_in_mag_2d = None
                if endpoints_list[b]['features_in_mag'] is not None:
                    features_in_mag = to_numpy(
                        torch.sigmoid(endpoints_list[b]['features_in_mag']))  # (num_steps, 2, num_edges)
                    features_in_mag_2d = np.zeros([num_nodes, num_nodes])
                    features_in_mag_2d = np.tile(features_in_mag_2d[None, ...], (num_steps, 1, 1))
                    edge_index = frame_idx[
                        to_numpy(data_list[b]['edge_index'])]  # edge_index with the correct frame index

                    for s in range(num_steps):
                        features_in_mag_2d[s, edge_index[0], edge_index[1]] = features_in_mag[s, 0, :]
                        features_in_mag_2d[s, edge_index[1], edge_index[0]] = features_in_mag[s, 1, :]

                scene = data_list[b]['source_clouds'][0][0][0]
                if edge_weights_2d is None:
                    edge_weights_2d = np.array([])
                if features_in_mag_2d is None:
                    features_in_mag_2d = np.array([])

                np.savez_compressed(os.path.join(save_folder, scene + '.npz'),
                                    t_star=t_star,
                                    t_pred=t_pred,
                                    t_pred_iter=t_pred_iter,
                                    edge_weights_pred=edge_weights_2d,
                                    features_in_mag=features_in_mag_2d)

            elif self.config['dataset'] == 'neurora':
                """Save neurora h5 outputs"""
                if self.hf is None:
                    self.hf = h5py.File(os.path.join(save_folder, 'output_transforms.h5'), 'w')

                pred_vec = SO3.from_matrix(pred_list[b][-1]).as_quaternion()
                gt_vec = data_list[b].t_star.as_quaternion()
                init_vec = torch.tensor([[1.0, 0., 0., 0.]]).repeat(pred_vec.shape[0], 1).to(
                    pred_vec.device)
                to_write = torch.cat([init_vec, pred_vec, pred_vec, gt_vec], dim=1).data.cpu().numpy()
                self.hf.create_dataset('/data/' + str(self.num_graphs + 1 + b), data=to_write)
            elif self.config['dataset'] == 'rotreal':
                """Save as MATLAB .mat file so it can be easily read using matlab"""
                pred_mat = SO3.from_matrix(pred_list[b][-1]).as_matrix()
                pred_mat = np.transpose(to_numpy(pred_mat.data), [1, 2, 0])  # shift the "node"-axis to the last
                scene = test_data['scene'][0]
                dst_fname = os.path.join(save_folder, scene + '.mat')
                savemat(dst_fname, {'R_pred': pred_mat})
            else:
                raise NotImplementedError

        self.num_graphs += batch_size

    def test_epoch_end(self, step: int):
        all_losses = self.loss_accumulator.get_aggregated()
        all_metrics = self.metrics_accumulator.get_aggregated()

        # Average the losses from all the steps
        all_losses = {k: torch.mean(all_losses[k]) for k in all_losses}

        # Compute aggregated metrics
        all_metrics, input_stats = get_aggregated_metrics(all_metrics, eval_is_relative(self.config.dataset, self.config.phase))

        if self.hf is not None:
            self.hf.close()

        print('Completed inference on {} graphs containing an average total of {} edges'.format(
            self.num_graphs, self.num_edges / self.num_graphs))
        print_loss_metrics(self._logger, step, all_losses, all_metrics, input_stats)

        self.loss_accumulator.reset()
        self.metrics_accumulator.reset()

    @staticmethod
    def _split_data_batch(data, pred, endpoints):
        # Specialized function to split data and predicted values to a list

        data_list = data.to_data_list()
        num_batch = len(data_list)

        node_slices = data.__slices__['x']
        edge_slices = data.__slices__['edge_index']
        num_steps = len(pred)
        num_edges = data.edge_index.shape[1]

        pred_list = []
        endpoints_list = []

        weight_logits = None
        if 'weight_logits' in endpoints:
            assert 'edge_mask' not in endpoints or endpoints['edge_mask'] is None or torch.all(endpoints['edge_mask'])
            weight_logits = [torch.reshape(endpoints['weight_logits'][step], (2, num_edges))
                             for step in range(num_steps)]
        features_in_mag = None
        if 'features_in_mag' in endpoints:
            assert 'edge_mask' not in endpoints or endpoints['edge_mask'] is None or torch.all(endpoints['edge_mask'])
            features_in_mag = [torch.reshape(endpoints['features_in_mag'][step], (2, num_edges))
                               for step in range(num_steps)]

        for batch in range(num_batch):
            # Save predictions of absolute poses
            xStart, xEnd = node_slices[batch:batch + 2]
            pred_i = torch.stack([pred[step][xStart:xEnd].as_matrix() for step in range(num_steps)],
                                 dim=0)
            pred_list.append(pred_i)

            # Save predictions of inlier prediction
            eStart, eEnd = edge_slices[batch:batch + 2]
            weight_logits_b = torch.stack([weight_logits[step][:, eStart:eEnd] for step in range(num_steps)], dim=0) \
                if weight_logits is not None else None
            features_in_mag_b = torch.stack([features_in_mag[step][:, eStart:eEnd] for step in range(num_steps)], dim=0) \
                if features_in_mag is not None else None

            endpoints_list.append({
                'weight_logits': weight_logits_b,
                'features_in_mag': features_in_mag_b
            })

        return data_list, pred_list, endpoints_list

    def compute_loss(self, data, pred, endpoints):
        """To be implemented via subclass"""
        raise NotImplementedError

    def save_summaries(self, writer, step, input_stats=None, endpoints=None, **kwargs):
        """Default implementation does nothing"""
        pass