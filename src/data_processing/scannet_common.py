import glob
import os

import numpy as np


DEPTH_FNAME_FORMAT = 'frame-{:06d}.depth.pgm'
POSE_FNAME_FORMAT = 'frame-{:06d}.pose.txt'
_DIR = os.path.dirname(os.path.abspath(__file__))


def get_scannet_splits():
    """Returns the list of scenes to be processed. We use the same scenes as [2]
    """
    with open(os.path.join(_DIR, '../data_loader/scannet_train.txt')) as fid:
        train_scenes = [s.strip() for s in fid.readlines()]
    with open(os.path.join(_DIR, '../data_loader/scannet_val.txt')) as fid:
        val_scenes = [s.strip() for s in fid.readlines()]
    with open(os.path.join(_DIR, '../data_loader/scannet_test.txt')) as fid:
        test_scenes = [s.strip() for s in fid.readlines()]
    return train_scenes, val_scenes, test_scenes


def parse_scannet_info(path):
    """Loads the scannet metadata which include useful information such as
    camera intrinsics
    """
    with open(path, 'r') as fin:
        lines = fin.readlines()
        info_dict = {}
        for line in lines:
            key, val = line.strip().split('=')
            key, val = key.strip(), val.strip()

            if key in ['m_versionNumber', 'm_colorWidth', 'm_colorHeight',
                       'm_depthWidth', 'm_depthHeight', 'm_depthShift',
                       'm_frames.size']:
                val = int(val)

            elif key in ['m_calibrationDepthIntrinsic', 'm_calibrationColorIntrinsic']:
                val = list(map(float, val.split(' ')))
                val = np.reshape(val, (4, 4))
                assert np.all(val[:3, 3] == 0) and np.all(val[3, :3] == 0) \
                       and val[3, 3] == 1
                val = val[:3, :3]

            elif key in ['m_calibrationDepthExtrinsic', 'm_calibrationColorExtrinsic']:
                val = list(map(float, val.split(' ')))
                val = np.reshape(val, (4, 4))
                assert val[3, 3] == 1

            info_dict[key[2:]] = val

    return info_dict


def read_pose(fname: str) -> np.ndarray:
    """Returns pose, which transforms points from camera to world frame.
    Note that this is inverse of the convention used in Learn2Sync
    """
    pose_wc = np.loadtxt(fname)
    return pose_wc


def read_poses_all(folder: str) -> np.ndarray:
    """Read all poses in the folder
    Does not do any error checking, i.e. certain pose might be infinity.
    """
    num_frames = len(glob.glob(os.path.join(folder, '*.pgm')))
    poses = []
    for i in range(num_frames):
        pose = read_pose(os.path.join(folder, POSE_FNAME_FORMAT.format(i)))
        poses.append(pose)

    poses = np.stack(poses, axis=0)
    return poses


