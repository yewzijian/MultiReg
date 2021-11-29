"""Processes the 1DSfM dataset into a format which is easier to read from MATLAB
"""
import argparse
import os
import shutil

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.io import loadmat, savemat

from common.lie.numpy import SO3

parser = argparse.ArgumentParser(description='Process Neurora Real Dataset')
# Data paths
parser.add_argument('--input_path', default='../../data/rotreal/raw',
                    type=str, help='Path to dataset files')
parser.add_argument('--output_path', default='../../data/rotreal/processed')
args = parser.parse_args()


def process_1dsfm():
    scenes = ['Alamo', 'Ellis_Island', 'Gendarmenmarkt', 'Madrid_Metropolis',
              'Montreal_Notre_Dame', 'Notre_Dame', 'NYC_Library',
              'Piazza_del_Popolo', 'Piccadilly', 'Roman_Forum',
              'Tower_of_London', 'Trafalgar', 'Union_Square', 'Vienna_Cathedral',
              'Yorkminster']
    for scene in scenes:
        src_folder = os.path.join(args.input_path, scene)
        dst_fname = os.path.join(args.output_path, '{}.mat'.format(scene))

        # Reads connected component information
        with open(os.path.join(src_folder, 'cc.txt')) as fid:
            cc = [int(t) for t in fid.read().split()]

        # Read groundtruth rotations
        num_cam, pose_data = load_bundler_poses(os.path.join(src_folder, 'gt_bundle.out'))
        R_cw_gt = np.stack([p['rot'] for p in pose_data], axis=0)

        # Read relative poses
        edge_index = []
        R_ji = []
        with open(os.path.join(src_folder, 'EGs.txt')) as fid:
            for line in fid:
                tokens = line.split()
                edge_index.append((int(tokens[0]), int(tokens[1])))
                R_ji.append(np.reshape([float(t) for t in tokens[2:11]], (3, 3)))
        R_ji = SO3.from_matrix(np.stack(R_ji))
        R_ji = R_ji.inv()
        edge_index = np.array(edge_index).transpose()
        assert np.all(np.isin(edge_index, cc))

        # Renumber nodes
        nodes_mapping = np.full((num_cam), -1)
        nodes_mapping[cc] = np.arange(len(cc))
        R_cw_gt = R_cw_gt[cc]
        edge_index = nodes_mapping[edge_index]

        # Some cameras do not have groundtruth
        has_gt = np.sum(R_cw_gt, axis=(1, 2)) != 0.0
        R_cw_gt[~has_gt, :, :] = np.nan

        # Compute errors
        R_cw_gt = SO3(R_cw_gt)
        i, j = edge_index
        R_ji_gt = R_cw_gt[j] * R_cw_gt[i].inv()
        aerr = R_ji_gt.compare(R_ji)['rot_deg']

        print('{} dataset: # nodes {}, # edges {}, edge density: {:.3f}%'.format(
            scene, len(R_cw_gt),
            edge_index.shape[1], edge_index.shape[1] / (len(R_cw_gt) * ((len(R_cw_gt) - 1))/2) * 100
        ))
        print('rotation error: {:.3f} deg (mean), {:.3f} deg (median)'.format(
            np.nanmean(aerr), np.nanmedian(aerr)
        ))
        print('Percentage inliers @ 5deg: {:.2f}'.format(np.mean(aerr < 5)))

        R_gt_data = R_cw_gt.data
        R_gt_data[~has_gt, :, :] = 0

        savemat(dst_fname,
                {'Rgt': np.transpose(R_gt_data, (1, 2, 0)),
                 'RR': np.transpose(R_ji.data, (1, 2, 0)),
                 'I': edge_index+1})  # expect 1-indexing}


def load_bundler_poses(fname):
    with open(fname) as fid:
        header = fid.readline()
        assert header.startswith('# Bundle file')
        num_cameras, num_points = list(map(int, fid.readline().split()))

        camera_data = []
        for iCam in range(num_cameras):
            # Intrinsics
            f, k1, k2 = [float(x) for x in fid.readline().split()]
            rot0 = [float(x) for x in fid.readline().split()]
            rot1 = [float(x) for x in fid.readline().split()]
            rot2 = [float(x) for x in fid.readline().split()]
            rot = np.stack([rot0, rot1, rot2], axis=0)
            trans = [float(x) for x in fid.readline().split()]

            camera_data.append({
                'intrinsics': (f, k1, k2),
                'rot': rot,
                'trans': trans,
            })

    return num_cameras, camera_data


if __name__ == '__main__':

    os.makedirs(args.output_path, exist_ok=True)

    process_1dsfm()

