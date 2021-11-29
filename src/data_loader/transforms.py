from typing import Dict

import numpy as np

PERTURB_NOISE_ROT = 20 / 180 * np.pi
PERTURB_NOISE_TRANS = 0.2


class CorruptEdges:
    """Corrupt a certain proportion of edges. Note that sigma is not updated,
    but this is ok since our network does not make use of it.
    """
    def __init__(self, p):
        """

        Args:
            p: Percentage of edges to replace with random transformations
               Note that we pick all edges with equal probability, regardless
               of whether they are inliers or are already outliers
        """
        self.p = p

    def __call__(self, sample: Dict):
        num_edges = sample['tij_hat'].shape[0]
        edges_to_corrupt = np.random.rand(num_edges) < self.p
        num_corrupt = np.sum(edges_to_corrupt)
        LieGroup = type(sample['tij_hat'])
        sample['tij_hat'][edges_to_corrupt] = LieGroup.rand(num_corrupt)
        return sample


class Perturb:
    """Apply small perturbations the absolute transformations, and transforms
    the relative poses accordingly
    Note that frame 0 will always be considered the reference frame and will
    not be perturbed to keep it at identity
    """

    def __init__(self):
        self.rot_mag = PERTURB_NOISE_ROT
        self.trans_mag = PERTURB_NOISE_TRANS
        self.ref_idx = 0

    def __call__(self, sample: Dict):

        LieGroup = type(sample['tij_hat'])

        t_star = sample['t_star']
        tij_hat = sample['tij_hat']
        tij_gt = sample['tij_gt']
        num_nodes = t_star.shape[0]
        edge_index = sample['edge_index']
        i, j = edge_index[0, :], edge_index[1, :]

        jitter = LieGroup.rand(num_nodes, self.rot_mag, self.trans_mag)
        jitter[self.ref_idx, ...] = LieGroup.identity()

        # Applies transform on absolute poses
        t_star = jitter * t_star

        # Applies transforms on measured and groundtruth relative poses
        tij_hat = jitter[j] * tij_hat * jitter[i].inv()
        tij_gt = jitter[j] * tij_gt * jitter[i].inv()

        # Update the absolute and relative transforms
        sample['t_star'] = t_star
        sample['tij_hat'] = tij_hat
        sample['tij_gt'] = tij_gt
        return sample


class PruneEdges:
    def __init__(self, thresh):
        """Threshold on the median outlier distance in the overlapping region.
         Edges with distances above this will be removed
        """
        self.sigma_thresh = thresh

    def __call__(self, sample):
        is_good = sample['edge_sigma'] < 0.05
        sample['edge_index'] = sample['edge_index'][:, is_good]
        sample['edge_attr'] = sample['edge_attr'][is_good, :]
        sample['tij_hat'] = sample['tij_hat'][is_good]
        sample['tij_gt'] = sample['tij_gt'][is_good]
        sample['edge_inliers'] = sample['edge_inliers'][is_good]
        sample['edge_sigma'] = sample['edge_sigma'][is_good]
        return sample
