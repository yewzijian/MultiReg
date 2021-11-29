from collections import defaultdict
import logging
from typing import Dict, Tuple, Type

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_sum, scatter_max

from common.lie.torch.liegroupbase import LieGroupBase
from common.torch_helpers import all_to_device
from losses import pose_loss
from models.PoseGraphNet_Generic import PoseGraphNet_Generic
from models.PoseGraphNet_common import MetaLayer_mod, SquashRot


class EdgeModel(torch.nn.Module):
    def __init__(self, transform_type: Type[LieGroupBase],
                 node_dim, edge_dim, global_dim, hidden_dim=128):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + transform_type.DIM + edge_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim),
        )

    def forward(self, src: Tuple, dst: Tuple, edge_attr, u, batch):
        """

        Args:
            src: Node feature of source
            dst: Node feature of dst
            edge_attr: Edge feature, a pair, each of size [E, F_e]
            u: [B, F_u], where B is the number of graphs.
            batch: [E] with max entry B - 1.

        Returns:

        """
        edge_feat, Tij_hat = edge_attr

        xfeati, xfeatj = src[0], dst[0]
        Ti, Tj = src[1], dst[1]
        Tij = Tj * Ti.inv()  # Current estimate
        T_err = Tij_hat * Tij.inv()
        feat_cat = torch.cat([xfeati, xfeatj, edge_feat, T_err.vec(), u[batch]], dim=1)

        edge_feat_out = edge_feat + self.edge_mlp(feat_cat)

        return edge_feat_out, Tij_hat


class SimpleNodeModel(torch.nn.Module):
    def __init__(self, transform_type: Type[LieGroupBase],
                 node_dim, edge_dim, global_dim, hidden_dim, aggr='mean'):
        super().__init__()

        self.aggr = aggr
        self.Transform = transform_type

        self.node_mlp_1 = nn.Sequential(
            nn.Linear(2 * node_dim + self.Transform.DIM + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.node_mlp_2 = nn.Sequential(
            nn.Linear(hidden_dim + global_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim + self.Transform.DOF)
            # No ReLU since eps or node update can be negative
        )

        self.squash = SquashRot()

        if self.aggr in ['weighted_sum']:
            # Subnetwork for predicting inlier weights

            self.weights_mlp_1 = nn.Sequential(
                nn.Linear(2 * node_dim + self.Transform.DIM + edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU()
            )
            self.weights_mlp_2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.logger = logging.getLogger(__name__)
        self.logger.info('Using {} aggregation scheme'.format(aggr))

    def forward(self, x: Tuple, edge_index, edge_attr, u, batch):
        """

        Args:
            x:
            edge_index: [2, E] with max entry N - 1.
            edge_attr: [E, F_e]
            u: [B, F_u]
            batch: [N] with max entry B - 1.

        Returns:

        """
        endpoints = {}

        xfeat, T = x
        num_nodes = x[0].shape[0]
        i, j = edge_index[0, :], edge_index[1, :]  # propagation from i --> j
        edge_feat, Tij_hat = edge_attr

        xfeati, xfeatj = xfeat[i], xfeat[j]
        Ti, Tj = T[i], T[j]

        # i -> j
        T_err_ij = Tj * Ti.inv() * Tij_hat.inv()
        features_ij = torch.cat([xfeati, xfeatj, edge_feat, T_err_ij.vec()], dim=1)
        # j -> i
        T_err_ji = Ti * Tj.inv() * Tij_hat
        features_ji = torch.cat([xfeatj, xfeati, edge_feat, T_err_ji.vec()], dim=1)

        features_ijji = torch.cat([features_ij, features_ji], dim=0)
        j_both = torch.cat([j, i], dim=0)

        features_in = self.node_mlp_1(features_ijji)
        endpoints['features_in_mag'] = torch.norm(features_in, dim=-1, keepdim=True)

        if self.aggr == 'max':
            aggr_in, _ = scatter_max(features_in, j_both, dim=0, dim_size=num_nodes)
        elif self.aggr == 'mean':
            aggr_in = scatter_mean(features_in, j_both, dim=0, dim_size=num_nodes)
        elif self.aggr == 'weighted_sum':
            # Compute weights using robust pooling, i.e. max
            features_in_w = self.weights_mlp_1(features_ijji)
            aggr_in_w, _ = scatter_max(features_in_w, j_both, dim=0, dim_size=num_nodes)
            weight_logits = self.weights_mlp_2(
                torch.cat([features_in_w, aggr_in_w[j_both, :]], dim=-1))
            weights = torch.sigmoid(weight_logits)

            # weighted scatter sum
            aggr_in = scatter_sum(features_in * weights, j_both, dim=0, dim_size=num_nodes)
            endpoints['weight_logits'] = weight_logits

            aggr_in = aggr_in / (torch.norm(aggr_in, dim=-1, keepdim=True) + 1e-4)
        else:
            raise AssertionError('aggr method not supported')

        # Post aggregation MLP
        out = self.node_mlp_2(torch.cat([aggr_in, xfeat, u[batch]], dim=-1))
        xfeat_update, T_update = out[:, :-self.Transform.DOF], out[:, -self.Transform.DOF:]
        T_update = self.squash(T_update)
        xfeat = x[0] + xfeat_update
        T_out = x[1].boxplus_left(T_update, pseudo=False)
        x_out = (xfeat, T_out)

        endpoints['theta_update'] = torch.norm(T_update[..., -3:], dim=-1)
        return x_out, endpoints


class GlobalModel(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim):
        super(GlobalModel, self).__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, global_dim),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        xfeat, T = x
        if xfeat.shape[1] == 0:
            return self.global_mlp(u)
        else:
            out = torch.cat([u, scatter_mean(xfeat, batch, dim=0)], dim=1)
            return self.global_mlp(out)


class PoseGraphNet(PoseGraphNet_Generic):
    """Very simple pose graph optimizer based on the left version of the
    box-plus operator"""
    def __init__(self, transform_type: Type[LieGroupBase], config: Dict):
        super().__init__(transform_type, config)

        # Number of dimensions for node and edges, as well as the intermediate
        # layers
        self.node_dim = 16
        self.edge_dim = 0  # Edge features not useful
        self.global_dim = 4
        self.hidden_dim = config['hidden_dim']

        # Main graph neural network. We use a custom MetaLayer which
        # is customized to handle transforms separately from the features.
        # As in the original Metalayer edge features can be updated, allowing
        # it to update its uncertainty if required.
        node_model = SimpleNodeModel(transform_type=self.Transform,
                                     node_dim=self.node_dim,
                                     edge_dim=self.edge_dim,
                                     global_dim=self.global_dim,
                                     hidden_dim=self.hidden_dim,
                                     aggr=config['aggr'])

        if self.edge_dim > 0:
            edge_model = EdgeModel(transform_type=self.Transform,
                                   node_dim=self.node_dim,
                                   edge_dim=self.edge_dim,
                                   global_dim=self.global_dim,
                                   hidden_dim=self.hidden_dim)
        else:
            edge_model = None

        if self.global_dim > 0:
            global_model = GlobalModel(node_dim=self.node_dim,
                                       edge_dim=self.edge_dim,
                                       global_dim=self.global_dim,
                                       hidden_dim=self.hidden_dim)
        else:
            global_model = None

        self.meta_op = MetaLayer_mod(edge_model=edge_model,
                                     node_model=node_model,
                                     global_model=global_model)

    def forward(self, data):
        """Forward pass

        Args:
            data: torch_geometric data containing the following additional fields:
              - connected_index: Edges of connected components
              - edge_index: Edges [i, j]
              - edge_inliers: Inlier mask (1: inliers, 0.5: don't care, 0: outliers)
              - edge_sigma: Whether edge is 'good'. Only used for evaluation on ScanNet
              - source_clouds: Indicate which scene and point cloud the data comes from.
                  Useful for visualizing the reconstructed scene, e.g. for ScanNet.
              - t_star: Groundtruth transforms, transforms points in world to camera frame
              - tij_gt: Transforms points in frame i to frame j. Used for computing loss/evaluation
                        tij_gt = t_star[j] * t_star[i].inv()
              - tij_hat: Measured relative transforms, same convention as tij_gt

        Returns:
            out: List (num_iter) of transforms
            endpoints: Other outputs, e.g. inlier weights
        """

        out = []  # Stores optimized absolute poses for each iteration
        endpoints = defaultdict(list)

        assert isinstance(data.t_star, torch.Tensor) and \
               isinstance(data.tij_hat, torch.Tensor) and \
               isinstance(data.tij_gt, torch.Tensor)

        data = all_to_device(data, self.device)
        data.t_star = self.Transform(data.t_star)
        data.tij_hat = self.Transform(data.tij_hat)
        data.tij_gt = self.Transform(data.tij_gt)

        edge_index, batch = data.edge_index, data.batch
        tij_hat = data.tij_hat
        batch_size = data.num_graphs
        num_nodes = data.num_nodes
        num_edges = edge_index.shape[1]

        # Initialize/encode nodes and edges
        x = (torch.zeros((num_nodes, self.node_dim), device=self.device),
             self.Transform.identity(num_nodes).to(self.device))
        edge_attr = torch.zeros((num_edges, self.edge_dim), device=self.device)
        edge_attr = (edge_attr, tij_hat)
        u = torch.zeros((batch_size, self.global_dim), device=self.device)

        # Learned optimization
        for iIter in range(self.num_iter):
            # Process
            x, edge_attr, u, iter_endpoints = self.meta_op(
                x, edge_index, edge_attr, u=u, batch=batch)

            out.append(x[1])
            for k in iter_endpoints:
                endpoints[k].append(iter_endpoints[k])

        return out, endpoints

    def save_summaries(self, writer, step, input_stats=None, endpoints=None, **kwargs):
        with torch.no_grad():
            if endpoints is not None:
                if 'weight_logits' in endpoints:
                    for i in range(len(endpoints['weight_logits'])):
                        writer.add_histogram('weights/{}'.format(i),
                                             torch.sigmoid(endpoints['weight_logits'][i]), step)

                if 'theta_update' in endpoints:
                    for i in range(len(endpoints['theta_update'])):
                        writer.add_histogram('theta_update/{}'.format(i),
                                             endpoints['theta_update'][i], step)

            if input_stats is not None:
                for k in input_stats:
                    writer.add_scalar('input_stats/{}'.format(k), input_stats[k], step)

            # Generic losses/metrics
            for k in kwargs:
                if k in ['losses', 'metrics']:
                    values = kwargs[k]
                    for v in values:
                        writer.add_scalar('{}/{}'.format(k, v), values[v], step)

            if 'optimizer' in kwargs:
                writer.add_scalar('misc/lr', kwargs['optimizer'].param_groups[0]['lr'], step)

            if 'model' in kwargs:
                for k, weights in kwargs['model'].named_parameters():
                    if torch.numel(weights) > 0:
                        writer.add_histogram('network_weights/{}'.format(k), weights, step)
        writer.flush()

    def compute_loss(self, data, pred, endpoints):
        """Computes losses. This includes 3 (actually 2) components:
           'rel': Loss on relative poses, L_{rel}
           'inliers': BCE loss on predicted inlier weights, L_{bce}
           'abs': Loss on absolute poses - Unused by setting weight to zero
        """

        # abs: Absolute loss
        # rel: Relative losses (all pairs)
        # inliers: BCE loss on inlier prediction
        losses = {}
        weights = {}
        weight_multiplier = {k[2:]: self.config[k] for k in self.config if k.startswith('w_')}
        alpha = 0.5
        num_iter = len(pred)

        # For rel: we only consider edges within connected components
        i_all, j_all = data.connected_index
        tij_gt_all = data.t_star[j_all] * data.t_star[i_all].inv()

        # For inliers: Need to also take into account the edges which are discarded
        # during dropout. In addition, need to duplicate by 2 for the
        # bidirectional edges.
        if 'edge_mask' in endpoints and endpoints['edge_mask'] is not None:
            edge_mask = endpoints['edge_mask']
            edge_inliers = data.edge_inliers[edge_mask].repeat(2).float()
        else:
            edge_inliers = data.edge_inliers.repeat(2).float()
        bce_weights = (edge_inliers != 0.5).float()  # don't care for ambiguous cases
        bce_criterion = torch.nn.BCEWithLogitsLoss(weight=bce_weights)

        # Prediction accuracy
        for iIter in range(num_iter):

            iter_weight = alpha ** (len(pred) - 1 - iIter)

            if weight_multiplier['abs'] > 0.0:  # absolute loss
                key = 'iter_{}_abs'.format(iIter)
                weights[key] = iter_weight * weight_multiplier['abs']
                losses[key] = pose_loss(pred[iIter], data.t_star)

            if weight_multiplier['rel'] > 0.0:  # pairwise relative transformation error
                key = 'iter_{}_rel'.format(iIter)
                weights[key] = iter_weight * weight_multiplier['rel']
                losses[key] = pose_loss(
                    pred[iIter][j_all] * pred[iIter][i_all].inv(),
                    tij_gt_all)

            if 'weight_logits' in endpoints:
                key = 'iter_{}_inliers'.format(iIter)
                weights[key] = iter_weight * weight_multiplier['inliers']
                losses[key] = bce_criterion(
                    torch.squeeze(endpoints['weight_logits'][iIter], -1),
                    edge_inliers)

        losses['total'] = torch.sum(torch.stack([weights[k] * losses[k] for k in losses], dim=0))
        return losses
