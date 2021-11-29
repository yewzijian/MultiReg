import itertools
import logging
from typing import List, Dict

import hdf5storage
import networkx as nx
import numpy as np
import os
import torch

from torch_geometric.data import Data, Dataset

from common.lie.numpy import SO3, SO3q, SE3, LieGroupBase
from common.lie.torch import SO3 as SO3_torch, SE3 as SE3_torch


class MultiviewDataset(Dataset):

    def __init__(self, config, phase, transform):
        super().__init__()

        self.config = config
        self.phase = phase
        self.transform=transform

        self.data = []  # For storing individual view-graphs
        self.scene_data = {}  # For storing scene data
        self.transform_type = self.get_transform_type()
        self.bidirectional = bool(config['bidirectional'])
        self.min_nodes, self.max_nodes = None, None

        self.thresh_rot_low, self.thresh_trans_low = 5.0, 0.05  # lower than this: inliers
        self.thresh_rot_high, self.thresh_trans_high = 15.0, 0.15  # higher than this: outliers. In-between: don't care

        self.logger = logging.getLogger(self.__class__.__name__)

        self.dataset_path = self.get_dataset_path()
        self.scene_list = self.get_scene_list()
        self.load_data()
        self.logger.info('Loaded data {}({}) containing {} scenes and {} instances'.format(
            self.config['dataset'], self.phase, len(self.scene_list), len(self)))

    def load_data(self):
        for scene in self.scene_list:
            in_file = os.path.join(self.dataset_path, scene + '.npz')
            scene_data = np.load(in_file)
            num_frames = scene_data['Tij_meas'].shape[0]

            self.scene_data[scene] = {k: scene_data[k] for k in scene_data.files}

            if self.phase == 'train':
                self.data.extend([(scene, i) for i in range(num_frames)])
            else:
                self.data.append((scene, 0))

    def sample_nodes(self, scene_data):
        """Simplest scheme of sampling nodes. Does not care whether the graph
        remains connected"""
        scene_num_frames = scene_data['Tij_meas'].shape[0]
        num_sampled_nodes = min(np.random.randint(self.min_nodes, self.max_nodes + 1),
                                scene_num_frames)
        nodes = np.random.choice(np.arange(scene_num_frames), size=num_sampled_nodes,
                                 replace=False)
        return nodes

    def sample_nodes_contiguous(self, root, scene_data):
        """Sample nodes to use for training. We use a simple scheme where we
        1. Randomize number of nodes to use
        2. Use root as center and pick the adjacent frames
        3. Permute the node order (although it shouldn't matter if the network
           is permutation invariant)
        """

        # Randomize number of nodes to use
        scene_num_frames = scene_data['Tij_meas'].shape[0]
        num_nodes = np.random.randint(min(scene_num_frames, self.min_nodes),
                                      min(scene_num_frames, self.max_nodes) + 1)
        # Pick frames, making sure they still all within valid range
        start_idx = root - num_nodes // 2
        end_idx = start_idx + num_nodes
        shift = 0
        if start_idx < 0:
            shift = -start_idx
        elif end_idx > scene_num_frames:
            shift = scene_num_frames - end_idx
        start_idx, end_idx = start_idx + shift, end_idx + shift
        assert start_idx >= 0 and end_idx <= scene_num_frames

        nodes = np.arange(start_idx, end_idx)
        if self.phase != 'test':
            return np.random.permutation(nodes)
        else:
            # We return it in the original order so that the translation is
            # computed consistently for the test data.
            return nodes

    def sample_nodes_connected(self, scene_data):
        """Sample random nodes throughout the scene, while ensuring
        the whole graph is connected when we consider the correct pairwise transforms
        """
        scene_num_frames = scene_data['Tij_meas'].shape[0]

        if self.min_nodes >= scene_num_frames:
            # No need to sample
            subset = np.arange(scene_num_frames)
        else:
            adj_matrix = np.logical_and(scene_data['aerr'] < self.thresh_rot_high,
                                        scene_data['terr'] < self.thresh_trans_high)
            num_sampled_nodes = min(np.random.randint(self.min_nodes, self.max_nodes + 1),
                                    scene_num_frames)

            subset = np.random.choice(np.arange(scene_num_frames), size=num_sampled_nodes,
                                      replace=False)
            adj_matrix_subset = adj_matrix[subset, :]
            adj_matrix_subset = adj_matrix_subset[:, subset]
            G = nx.convert_matrix.from_numpy_matrix(adj_matrix_subset)
            largest_component = max(nx.connected_components(G), key=len)
            largest_component_nodes = list(largest_component)
            subset = subset[largest_component_nodes]

        if self.phase == 'test':
            return subset
        else:
            return np.random.permutation(subset)

    @staticmethod
    def extract_subgraph(scene_data, nodes):
        num_nodes = len(nodes)
        edge_index = np.array(list(itertools.combinations(range(num_nodes), 2))).transpose()
        i, j = nodes[edge_index[0]], nodes[edge_index[1]]
        t_star = scene_data['Tstar'][nodes]
        tij_hat = scene_data['Tij_meas'][i, j]
        tij_gt = scene_data['Tij_gt'][i, j]
        aerr = scene_data['aerr'][i, j]
        terr = scene_data['terr'][i, j]
        if 'sigma' in scene_data:
            # Median point distance in overlapping region. Used only for
            # evaluating using LMPR scheme where only "good" edges are evaluated
            edge_sigma = scene_data['sigma'][i, j]
        else:
            edge_sigma = np.zeros_like(aerr)  # For handling other datasets, e.g. 3DMatch
        return edge_index, t_star, tij_hat, tij_gt, aerr, terr, edge_sigma

    @staticmethod
    def get_edge_attr(num_edges, inliers=None):
        if inliers is None:
            return np.zeros((num_edges, 1))
        else:
            return inliers.astype(np.float32)[:, None]

    def compute_edge_inliers(self, sample: Dict):
        """Compute edge inliers (0: outlier, 0.5: don't care, 1: inlier)
        """
        errors = sample['tij_gt'].compare(sample['tij_hat'])

        # Initialize to 0.5 (don't care)
        edge_inliers = np.full(errors['rot_deg'].shape, 0.5, dtype=np.float32)

        if 'trans' in errors:
            inliers = np.logical_and(errors['rot_deg'] < self.thresh_rot_low,
                                     errors['trans'] < self.thresh_trans_low)
            outliers = np.logical_or(errors['rot_deg'] > self.thresh_rot_high,
                                     errors['trans'] > self.thresh_trans_high)
        else:
            inliers = errors['rot_deg'] < self.thresh_rot_low
            outliers = errors['rot_deg'] > self.thresh_rot_high

        edge_inliers[inliers] = 1.0
        edge_inliers[outliers] = 0.0
        sample['edge_inliers'] = edge_inliers

        return sample

    def get_components(self, sample: Dict):
        """Computes the edges between all vertices which are in the same component"""
        G = nx.Graph()
        inlier_edges = sample['edge_index'][:, sample['edge_inliers'] >= 0.5].T
        G.add_edges_from(inlier_edges)
        connected_index = []
        for cc in nx.connected_components(G):
            connected_index += list(itertools.combinations(cc, 2))
        connected_index = np.array(connected_index).T
        if 'valid_gt' in sample:
            # This ensures the relative loss isn't applied on the invalid nodes,
            # but actually this piece of code isn't reached in practice, the inlier edges only
            # reach nodes with groundtrtuh
            valid_pairs = np.all(sample['valid_gt'][connected_index], axis=0)
            connected_index = connected_index[:, valid_pairs]
        sample['connected_index'] = connected_index
        return sample

    def post_process(self, sample: Dict, get_ccomp: bool = True):
        """Post processing code that handles:
        1. Compute inliers and connected components
        2. Adding of edges in the opposite direction (if necessary)
        3. Convert to torchgeometric data instance

        Args:
            sample:
            get_ccomp: Whether to compute the pairs (for computing losses) using
              connected components
        """

        sample = self.compute_edge_inliers(sample)

        if get_ccomp:
            sample = self.get_components(sample)
        else:
            # Assume fully connected
            nodes = np.nonzero(sample['has_gt'])[0] if 'has_gt' in sample else \
                np.arange(sample['t_star'].shape[0])
            connected_index = list(itertools.combinations(nodes, 2))
            sample['connected_index'] = np.array(connected_index).T

        if self.bidirectional:
            sample = make_bidirectional(sample)

        sample = convert_to_graph(sample)
        return sample

    def __getitem__(self, item):
        if self.phase != 'train':
            np.random.seed(item)  # for consistency in testing

        scene, ref_node = self.data[item]
        scene_data = self.scene_data[scene]

        # Choose frames to use. For evaluation, we simply use the entire graph
        nodes = self.sample_nodes_contiguous(ref_node, scene_data)

        # Extract subgraph
        edge_index, t_star, tij_hat, tij_gt, aerr, terr, edge_sigma = \
            self.extract_subgraph(scene_data, nodes)
        t_star = SE3.from_matrix(t_star)
        tij_hat = SE3.from_matrix(tij_hat)
        tij_gt = SE3.from_matrix(tij_gt)

        # Prepare outputs, we always set reference at node 0
        # (the frames are already randomized during sample_nodes)
        t_star = t_star[0:1].inv() * t_star
        source_clouds = [(scene, n) for n in nodes]

        # # Flip t_star convention for consistency with Learn2Sync
        t_star = t_star.inv()

        sample = {
            'edge_index': edge_index,
            'tij_hat': tij_hat,
            'tij_gt': tij_gt,
            'edge_sigma': edge_sigma,
            't_star': t_star,
            'source_clouds': source_clouds
        }

        # Apply transformations
        if self.transform:
            sample = self.transform(sample)
        sample = self.post_process(sample)

        return sample

    def __len__(self):
        return len(self.data)

    def get_transform_type(self):
        return SE3_torch

    def get_dataset_path(self):
        """Return dataset path which will be stored in self.data_path"""
        return None

    def get_scene_list(self) -> List:
        """Should return a list of scenes based on the configuration"""
        return []


class NeuroraSynthetic(MultiviewDataset):
    """NeuRoRA synthetic dataset"""
    def __init__(self, config, phase, transform, **kwargs):
        super().__init__(config, phase, transform)

    def get_dataset_path(self):
        h5_path = os.path.join(self.config['dataset_path'],
                               'neurora/gt_graph_random_large_outliers.h5')
        assert os.path.exists(h5_path), \
            'Neurora synthetic dataset not found at {}'.format(h5_path)
        return h5_path

    def get_scene_list(self):
        return []

    def get_transform_type(self):
        return SO3_torch

    def load_data(self):
        if self.phase == 'train':
            data_ind = np.arange(240, 1200)
        elif self.phase == 'val':
            data_ind = np.arange(120, 240)
        elif self.phase == 'test':
            data_ind = np.arange(0, 120)
        elif self.phase == 'all':
            data_ind = np.arange(1200)
        else:
            raise AssertionError('Invalid phase')

        self.data = [(i, None) for i in data_ind]

    def __getitem__(self, item):
        if self.data[item][1] is None:
            ind = self.data[item][0]
            data = self._load_single_graph(ind)
            self.data[item] = (ind, data)
        return self.data[item][1]

    def _load_single_graph(self, ind):
        path = '/data/' + str(ind + 1)
        edge_index = hdf5storage.read(path=path + '/edge_index',
                                      filename=self.dataset_path)
        rel_quat = hdf5storage.read(path=path + '/edge_feature',
                                    filename=self.dataset_path).astype(np.float64)
        abs_quat = hdf5storage.read(path='/data/' + str(ind + 1) + '/y',
                                    filename=self.dataset_path).astype(np.float64)
        t_star = SO3q.from_quaternion(abs_quat, normalize=True)
        tij_hat = SO3q.from_quaternion(rel_quat, normalize=True)
        edge_index = edge_index.transpose().astype(np.int64)

        # Remove one of the directions of the edges (this is handled internally)
        to_keep = edge_index[0] < edge_index[1]
        edge_index = edge_index[:, to_keep]
        tij_hat = tij_hat[to_keep]

        # Convert to matrix form
        tij_hat = SO3.from_quaternion(tij_hat.data)
        t_star = SO3.from_quaternion(t_star.data)

        # Groundtruth edge transforms
        ti_gt, tj_gt = t_star[edge_index[0, :]], t_star[edge_index[1, :]]
        tij_gt = tj_gt * ti_gt.inv()

        sample = {
            'edge_index': edge_index,
            'tij_hat': tij_hat,
            'tij_gt': tij_gt,
            'edge_sigma': np.array([]),
            't_star': t_star,
            'source_clouds': [(str(ind), n) for n in range(t_star.shape[0])],
        }

        # # We do not augment the data for neurora since the dataset is large enough
        # if self.transform:
        #     sample = self.transform(sample)
        sample = self.post_process(sample)

        return sample


class ThreeDMatch(MultiviewDataset):

    def __init__(self, config, phase, transform, **kwargs):
        super().__init__(config, phase, transform)

        self.min_nodes, self.max_nodes = 30, 30

    def get_scene_list(self):
        with open(os.path.join(os.path.dirname(__file__),
                               '3dmatch_{}.txt'.format(self.phase))) as fid:
            scene_list = [line.strip() for line in fid.readlines()]

        return scene_list

    def get_dataset_path(self):
        if self.phase in ['train', 'val']:
            subfolder_phase = 'train'
        else:
            subfolder_phase = 'test'

        dataset_path = os.path.join(self.config['dataset_path'], '3dmatch',
                                    subfolder_phase, 'pairwise_poses', 'RegBlock')
        return dataset_path


class Scannet(MultiviewDataset):
    """30 keyframe data processed following LMPR paper"""
    def __init__(self, config, phase, transform, **kwargs):
        super().__init__(config, phase, transform)

        self.min_nodes, self.max_nodes = 30, 30

    def get_scene_list(self):
        with open(os.path.join(os.path.dirname(__file__),
                               'scannet_{}.txt'.format(self.phase))) as fid:
            scene_list = [line.strip() for line in fid.readlines()]

        return scene_list

    def get_dataset_path(self):
        dataset_path = os.path.join(self.config['dataset_path'],
                                    'scannet/pairwise_poses/RegBlock')
        return dataset_path


def make_bidirectional(sample: Dict):
    """Add the edge for the other direction"""
    sample['tij_hat'] = SE3.stack([sample['tij_hat'], sample['tij_hat'].inv()])
    sample['tij_gt'] = SE3.stack([sample['tij_gt'], sample['tij_gt'].inv()])
    sample['edge_index'] = np.concatenate([sample['edge_index'], sample['edge_index'][::-1, :]], axis=1)
    sample['edge_sigma'] = np.tile(sample['edge_sigma'], 2)

    if 'edge_attr' in sample:
        sample['edge_attr'] = np.tile(sample['edge_attr'], (2, 1))
    elif 'edge_inliers' in sample:
        sample['edge_inliers'] = np.tile(sample['edge_inliers'], 2)
    return sample


def convert_to_graph(sample: Dict):
    dtype = torch.get_default_dtype()

    num_nodes = sample['t_star'].shape[0]

    for k in sample:
        if isinstance(sample[k], np.ndarray):
            sample[k] = torch.from_numpy(sample[k]) if sample[k].dtype != np.float64 else \
                torch.from_numpy(sample[k]).type(dtype)
        elif isinstance(sample[k], LieGroupBase):
            sample[k] = torch.from_numpy(sample[k].data).type(dtype)
    sample['x'] = torch.zeros(num_nodes)  # Dummy to easily infer num_nodes
    sample = Data(**sample)

    return sample
