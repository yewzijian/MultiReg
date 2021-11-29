from collections import defaultdict
import itertools
import logging
from typing import List, Dict

import numpy as np
import torch
import torch_geometric

from common.torch_helpers import to_numpy


def print_loss_metrics(logger, step, losses=None, metrics=None, input_stats=None):

    logger.info('Step {} validation:'.format(step))
    if losses is not None:
        loss_str = ', '.join(['{:.4g} ({})'.format(losses[k], k) for k in losses if k != 'total'])
        loss_str = 'Losses: ' + loss_str + ' | {:.4g} (total)'.format(losses['total'])
        logger.info(loss_str)

    if input_stats is not None:
        meas_metrics_str = 'Input Measurements have errors: [rot] {:.1f} (mean), {:.1f} (med)'.format(
            input_stats['rot_error_mean'], input_stats['rot_error_med']
        )
        if 'trans_error_mean' in input_stats:
            meas_metrics_str += ' | [trans] errors: {:.2f} | {:.2f}'.format(
                input_stats['trans_error_mean'], input_stats['trans_error_med'])

        logger.info(meas_metrics_str)

    if metrics is not None:  # Training on graphs
        rot_metrics_str = 'Rot: ' + \
                          ', '.join(['{:.1f} ({})'.format(metrics[k] * 100, k[9:])
                                    for k in metrics if k.startswith('rot_ecdf')]) + \
                          ' | {:.3f} (mean), {:.3f} (median)'.format(
                              metrics['rot_error_mean'], metrics['rot_error_med'])
        logger.info(rot_metrics_str)

        if 'trans_error_mean' in metrics:
            trans_metrics_str = 'Trans: ' + \
                                ', '.join(['{:.1f} ({})'.format(metrics[k] * 100, k[11:])
                                          for k in metrics if k.startswith('trans_ecdf')]) + \
                                ' | {:.2f} (mean), {:.2f} (median)'.format(
                                    metrics['trans_error_mean'], metrics['trans_error_med'])
            logger.info(trans_metrics_str)


def compute_metrics(data_batch: torch_geometric.data.batch.Batch, pred_batch, is_rel, only_good):
    if is_rel:
        return compute_rel_metrics(data_batch, pred_batch, check_only_good=only_good)
    else:
        return compute_abs_metrics(data_batch, pred_batch)


def compute_abs_metrics(data_batch: torch_geometric.data.batch.Batch, pred_batch, align=True):
    metrics = {}
    assert data_batch.num_graphs == 1, 'This can only handle batch sizes of 1'

    # Compute statistics for measured transforms
    pairs = data_batch.edge_index
    i, j = pairs
    se3_gt_i = data_batch['t_star'][i]
    se3_gt_j = data_batch['t_star'][j]
    rel_gt = se3_gt_j * se3_gt_i.inv()
    if 'valid_gt' in data_batch:
        valid_pairs = data_batch.valid_gt[i] & data_batch.valid_gt[j]
    else:
        valid_pairs = torch.ones(pairs.shape[1], dtype=torch.bool, device=pairs.device)
    # Compute measurement errors
    measured_errs = rel_gt.compare(data_batch['tij_hat'])
    metrics['measured_rot_errors'] = measured_errs['rot_deg'][valid_pairs]
    if 'trans' in measured_errs:
        metrics['measured_trans_errors'] = measured_errs['trans'][valid_pairs]

    # Evaluate predictions
    T_cw_pred = pred_batch[-1]  # only evaluate final iteration
    T_cw_gt = data_batch.t_star
    if hasattr(data_batch, 'valid_gt'):
        # Remove nodes without valid groundtruth
        T_cw_pred = T_cw_pred[data_batch['valid_gt']]
        T_cw_gt = T_cw_gt[data_batch['valid_gt']]
    if align:
        T_cw_pred = align_predictions(T_cw_gt, T_cw_pred)

    pred_err = T_cw_gt.compare(T_cw_pred)
    for k in pred_err:
        metrics[f'{k}_err_mean'] = torch.mean(pred_err[k])
        metrics[f'{k}_err_med'] = torch.median(pred_err[k])

    return metrics


def align_predictions(rot1, rot2, method='median', mask=None):
    """Align two sets of rotations, similar to what is done in the rotation
    averaging papers

    Notes:
        1. Only works for SO3
        2. (Update 2 June 2021) The rotation conventions are the same as that
           used in NeuRoRA/Chatterjee code. Specifically they assume that the
           rotations maps from world to camera, i.e. p_c = rot1 * p_w.

    Args:
        rot1: Groundtruth rotations, mapping points from the world frame to
          the camera frame
        rot2: Predicted rotations
        method: Either 'mean' or 'median'
        mask: If provided, will ignore entries corresponding to mask values of 0

    Returns:
        Aligned rot2
    """

    Transform = type(rot1)

    # Chooses a random frame for initial alignment
    if mask is None:
        j = np.random.randint(rot1.shape[0])
        rot1a = rot1 * rot1[j:j+1].inv()
        rot2a = rot2 * rot2[j:j+1].inv()
    else:
        j = np.random.choice(to_numpy(torch.nonzero(mask)[:, 0]))
        rot1a = rot1 * rot1[j:j+1].inv()
        rot2a = rot2 * rot2[j:j+1].inv()

    d, count = float('inf'), 0
    while d > 1e-5 and count < 20:
        if mask is None:
            align_w = (rot2a.inv() * rot1a).log()
        else:
            align_w = (rot2a[mask].inv() * rot1a[mask]).log()
        if method == 'mean':
            align_w = torch.mean(align_w, dim=0, keepdim=True)
        elif method == 'median':
            align_w = torch.median(align_w, dim=0, keepdim=True).values
        else:
            raise AssertionError('Alignment method not recognized')
        d = torch.norm(align_w)
        correction = Transform.exp(align_w)
        rot2a = rot2a * correction
        count += 1

    aligned = rot2a * rot1[j:j+1]
    return aligned


def compute_rel_metrics(data_batch: torch_geometric.data.batch.Batch, pred_batch, check_only_good):
    """Accumulate statistics

    Args:
        data_batch: Validation/Test data batch
        pred_batch: network outputs
    """
    metrics = {}
    batch_size = data_batch.num_graphs
    predicted_transforms = pred_batch[-1]  # Only evaluate last step

    # Check if predicted rotations are normalize, if not normalize them
    assert predicted_transforms.is_valid(), 'Predicted transforms are not valid SO(3)/SE(3)'

    # Compute statistics for measured transforms
    pairs = data_batch.edge_index
    i, j = pairs
    se3_gt_i = data_batch['t_star'][i]
    se3_gt_j = data_batch['t_star'][j]
    rel_gt = se3_gt_j * se3_gt_i.inv()
    if 'valid_gt' in data_batch:
        valid_pairs = data_batch.valid_gt[i] & data_batch.valid_gt[j]
    else:
        valid_pairs = torch.ones(pairs.shape[1], dtype=torch.bool, device=pairs.device)
    # Compute measurement errors
    measured_errs = rel_gt.compare(data_batch['tij_hat'])
    metrics['measured_rot_errors'] = measured_errs['rot_deg'][valid_pairs]
    if 'trans' in measured_errs:
        metrics['measured_trans_errors'] = measured_errs['trans'][valid_pairs]

    # Compute prediction errors
    # First compute which pairs to evaluate
    if not check_only_good:
        # Check all pairise edges
        pairs_to_eval = []
        for iBatch in range(batch_size):
            # Compute all pairwise combinations
            in_batch = np.nonzero(to_numpy(data_batch.batch == iBatch))[0]
            pairs_to_eval.append(torch.tensor(list(itertools.combinations(in_batch, 2))))
        pairs_to_eval = torch.cat(pairs_to_eval, dim=0)
    else:
        # Hack to compute metrics just on good edges
        pairs_to_eval = data_batch.edge_index[:, data_batch['edge_sigma'] < 0.05].transpose(0, 1)

    # Then evaluate the pairs
    i, j = pairs_to_eval[:, 0], pairs_to_eval[:, 1]
    se3_gt_i = data_batch['t_star'][i]
    se3_gt_j = data_batch['t_star'][j]
    se3_pred_i = predicted_transforms[i]
    se3_pred_j = predicted_transforms[j]
    rel_gt = se3_gt_j * se3_gt_i.inv()
    rel_pred = se3_pred_j * se3_pred_i.inv()
    pred_errs = rel_gt.compare(rel_pred)

    metrics['rot_errors'] = pred_errs['rot_deg']
    if 'trans' in pred_errs:
        metrics['trans_errors'] = pred_errs['trans']

    # Handles pairs without groundtruth
    if 'valid_gt' in data_batch:
        valid_pairs = data_batch.valid_gt[i] & data_batch.valid_gt[j]
        metrics['rot_errors'] = pred_errs['rot_deg'][valid_pairs]
        if 'trans' in pred_errs:
            metrics['trans_errors'] = pred_errs['trans'][valid_pairs]

    return metrics


def get_aggregated_metrics(metrics, is_rel):
    if is_rel:
        return get_aggregated_metrics_rel(metrics)
    else:
        return get_aggregated_metrics_abs(metrics)


def get_aggregated_metrics_rel(metrics):
    rot_thresh_deg = [3, 5, 10, 30, 45]
    trans_thresh = [0.05, 0.1, 0.25, 0.5, 0.75]

    # rot errors
    input_stats = {
        'rot_error_mean': torch.mean(metrics['measured_rot_errors']),
        'rot_error_med': torch.median(metrics['measured_rot_errors']),
    }
    aggr_metrics = {
        'rot_error_mean': torch.mean(metrics['rot_errors']),
        'rot_error_med': torch.median(metrics['rot_errors'])
    }
    for t in rot_thresh_deg:
        aggr_metrics['rot_ecdf_{}'.format(t)] = torch.mean((metrics['rot_errors'] < t).float())

    # trans errors
    if 'trans_errors' in metrics:
        input_stats.update({
            'trans_error_mean': torch.mean(metrics['measured_trans_errors']),
            'trans_error_med': torch.median(metrics['measured_trans_errors']),
        })
        aggr_metrics.update({
            'trans_error_mean': torch.mean(metrics['trans_errors']),
            'trans_error_med': torch.median(metrics['trans_errors'])
        })
        for t in trans_thresh:
            aggr_metrics['trans_ecdf_{}'.format(t)] = torch.mean((metrics['trans_errors'] < t).float())

    return aggr_metrics, input_stats


def get_aggregated_metrics_abs(metrics):
    # rot errors
    input_stats = {
        'rot_error_mean': torch.mean(metrics['measured_rot_errors']),
        'rot_error_med': torch.median(metrics['measured_rot_errors']),
    }
    aggr_metrics = {
        'rot_error_mean': torch.mean(metrics['rot_deg_err_mean']),
        'rot_error_med': torch.mean(metrics['rot_deg_err_med'])
    }

    # trans errors
    if 'trans_errors' in metrics:
        input_stats.update({
            'trans_error_mean': torch.mean(metrics['measured_trans_errors']),
            'trans_error_med': torch.median(metrics['measured_trans_errors']),
        })
        aggr_metrics.update({
            'trans_error_mean': torch.mean(metrics['trans_err_mean']),
            'trans_error_med': torch.mean(metrics['trans_err_medi='])
        })

    return aggr_metrics, input_stats


class Accumulator(object):
    """Generic Accumulator"""
    def __init__(self):
        self._data = defaultdict(float)  # Weighted sum
        self.reset()

    def reset(self):
        """Clears accumulated statistics
        """
        self._data = defaultdict(list)  # Stores metrics separately then concat

    def update(self, data):
        for k in data:
            self._data[k].append(data[k])

    def get_aggregated(self):
        """Return concatenated data
        """

        if len(self._data) == 0 or len(self._data[next(iter(self._data.keys()))]) == 0:
            raise AssertionError('You need to call update() at least once before calling this')
        self._concat_all()

        data = {k: self._data[k][0] for k in self._data}
        return data

    def _concat_all(self):
        for k in self._data:
            if self._data[k][0].dim() == 0:
                self._data[k] = [torch.stack(self._data[k], dim=0)]
            else:
                self._data[k] = [torch.cat(self._data[k], dim=0)]
