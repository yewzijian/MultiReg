"""Generate scannet dataset with FCGF matches
For comparison with [1], we sample frames which are 20 frames apart.
This contrasts with [2] where frames are only up to 6 frames apart.

Processing steps:
1. For each scene, we sample 100 frames (which are each 20 frames apart).
2. Extract descriptors
3. Perform pairwise matching using FCGF to get putative correspondences
4. Compute pairwise pose using the method described in [1].

References
[1] Gojcic et al. "Learning Multiview 3D Point Cloud Registration", CVPR 2020
[2] Huang et al. "Learning Transformation Synchronization", CVPR2019
"""
import argparse
import logging
import os
import shutil
import sys

sys.path.append('..')

import coloredlogs
import numpy as np
import open3d as o3d
from tqdm import tqdm

from common.lie.numpy import SE3
from data_processing.process_common import load_depth_and_convert_to_cloud, \
    extract_features_batch, extract_correspondences_batch, compute_pose_batch, \
    CLOUDS, FNAME_PLY_FORMAT, FNAME_POSE_FORMAT, save_info, generate_trainval_data_batch, load_info
from data_processing.scannet_common import get_scannet_splits, \
    parse_scannet_info, read_poses_all, DEPTH_FNAME_FORMAT


parser = argparse.ArgumentParser(description='Process Scannet Dataset')
# Data paths
parser.add_argument('--input_path', default='../../data/scannet/raw_data',
                    type=str, help='Path to extracted scannet files')
parser.add_argument('--output_path', default='../../data/scannet')
parser.add_argument('--lmpr_scannet_eval_path',
                    help='Set this path to point to the evaluation data from LMPR')
# Enables/disables which steps to run
parser.add_argument('--sample_frames', type=int, default=1, help='Sample frames')
parser.add_argument('--extract_features', type=int, default=1)
parser.add_argument('--extract_correspondences', type=int, default=1)
parser.add_argument('--compute_pose', type=int, default=1)
parser.add_argument('--generate_trainval_data', type=int, default=1)
# Dataset settings
parser.add_argument('--n_correspondences', type=int, default=5000,
                    help='number of points to be sampled in the correspondence estimation')
parser.add_argument('--num_keyframes', type=int, default=30,
                    help='Number of keyframes for scannet dataset (should be 30 as in LMPR).'
                         'This is also used to reduce the number of pairwise matching, training '
                         'keyframes more than this number apart will not be matched')
parser.add_argument('--gap', type=int, default=20)
# Method settings
parser.add_argument('--method', type=str, default='FCGF_LMPR',
                    choices=['FCGF_RANSAC', 'FCGF_LMPR'],
                    help='Method to estimate pairwise poses')
parser.add_argument('--use_mutuals', type=int, default=1,
                    help='Whether to only use mutual putative matches')
args = parser.parse_args()


def pick_frames(poses, is_train=True):
    """Pick valid frames, each ~GAP apart. Since some frames may have
    invalid poses, we first extract the longest segment we can without a
    gap larger than GAP (for 'train')
    We then sample from this longest segment. For 'train', some sampled
    frames might be invalid, which we replace with the closest valid frame.
    """
    is_valid = np.logical_not(np.any(np.isinf(poses), axis=(1, 2)))
    valid_frame_idx = np.nonzero(is_valid)[0]
    segments = np.split(valid_frame_idx, np.where(np.diff(valid_frame_idx) > args.gap)[0] + 1)
    longest_segment = max(segments, key=len)

    if is_train:
        start = np.random.randint(args.gap)
        frames = np.arange(start + longest_segment[0], longest_segment[-1], args.gap)
    else:
        assert len(longest_segment) > args.num_keyframes * args.gap
        iStart = np.random.randint(len(longest_segment) - (args.num_keyframes - 1) * args.gap)
        start = longest_segment[iStart]
        frames = np.arange(start, start + args.gap * args.num_keyframes, args.gap)

    valid_idx = np.nonzero(is_valid)[0]
    for i in np.nonzero(~is_valid[frames])[0]:
        # Adjust invalid frame indices to nearest valid frames
        bad_idx = frames[i]
        right = np.searchsorted(valid_idx, bad_idx)
        left_idx, right_idx = valid_idx[right - 1], valid_idx[right]
        frames[i] = left_idx if (bad_idx - left_idx < right_idx - bad_idx) else right_idx
    return frames


def sample_frames_lmpr(scene, is_train=True):
    src_folder = os.path.join(args.input_path, scene)
    dst_folder = os.path.join(args.output_path, CLOUDS, scene)
    if os.path.exists(dst_folder):
        logger.info('Skip extracting frames for {} as already exists'.format(scene))
        return 99999
    else:
        logger.info('Extracting frames for {}'.format(scene))

    os.makedirs(dst_folder)
    info = parse_scannet_info(os.path.join(src_folder, '_info.txt'))
    total_frames = info['frames.size']

    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(info['depthWidth'], info['depthHeight'],
                              info['calibrationDepthIntrinsic'][0, 0],  # fx
                              info['calibrationDepthIntrinsic'][1, 1],  # fy
                              info['calibrationDepthIntrinsic'][0, 2],  # cx
                              info['calibrationDepthIntrinsic'][1, 2])  # cy

    # Select frames
    poses = read_poses_all(src_folder)
    assert poses.shape[0] == total_frames
    frames = pick_frames(poses, is_train)
    if is_train:
        assert len(frames) >= args.num_keyframes
    else:
        assert len(frames) == args.num_keyframes

    # Extract frames
    # Extract
    iFrame = 0  # number of frames added
    for frame in tqdm(frames, ncols=80, leave=False):
        # Read depth and convert to point cloud
        depth_fname = os.path.join(src_folder, DEPTH_FNAME_FORMAT.format(frame))
        pcd = load_depth_and_convert_to_cloud(depth_fname, intrinsics)

        # Save cloud
        out_fname_ply = os.path.join(dst_folder, FNAME_PLY_FORMAT.format(iFrame))
        o3d.io.write_point_cloud(out_fname_ply, pcd)

        # Save pose and other metadata
        out_fname_info = os.path.join(dst_folder, FNAME_POSE_FORMAT.format(iFrame))
        save_info(out_fname_info, scene, frame, poses[frame])
        iFrame += 1

    return iFrame


def copy_eval_frames_lmpr(scene):
    raw_folder = os.path.join(args.input_path, scene)
    src_folder = os.path.join(args.lmpr_scannet_eval_path, scene)
    dst_folder = os.path.join(args.output_path, CLOUDS, scene)

    if os.path.exists(dst_folder):
        logger.info('Skip copying frames for {} as already exists'.format(scene))
        return

    # Load all poses to find out which frames they come from
    poses_all = read_poses_all(raw_folder)

    os.makedirs(dst_folder, exist_ok=True)
    frame_nums = []
    for i in range(30):
        ply_fname = 'cloud_bin_{}.ply'.format(i)
        info_fname =  'cloud_bin_{}.info.txt'.format(i)

        shutil.copyfile(os.path.join(src_folder, ply_fname), os.path.join(dst_folder, ply_fname))
        info = load_info(os.path.join(src_folder, info_fname))
        pose_wc = SE3.from_matrix(info['pose'], normalize=True).inv().as_matrix()
        pose_diff = np.sum(np.abs(poses_all - pose_wc[None, :, :]), (1, 2))
        frame_num = np.argmin(pose_diff)
        assert pose_diff[frame_num] < 1e-4
        save_info(os.path.join(dst_folder, info_fname), scene, frame_num, pose_wc)
        frame_nums.append(frame_num)
    logger.info('Scene {} source frames: {}'.format(scene, frame_nums))


def sample_frames_lmpr_batch(train_scenes, val_scenes, test_scenes):

    logger.info('Sampling keyframes, frames will be stored in {}'.format(
        os.path.join(args.output_path, CLOUDS)))

    np.random.seed(0)  # So that the same frames are chosen each time
    min_len = float('inf')
    for scene in train_scenes:
        num_kf = sample_frames_lmpr(scene, is_train=True)
        min_len = min(min_len, num_kf)
    logger.info('Finished sampling keyframes. '
                'Shortest training sequence has {} keyframes'.format(min_len))

    for scene in val_scenes:
        num_kf = sample_frames_lmpr(scene, is_train=False)

    # For test scenes, we copy the ply and info files from the author.
    # However, we flip the convention to T_wc to be consistent with the other
    # datasets
    if args.lmpr_scannet_eval_path is not None:
        for scene in test_scenes:
            copy_eval_frames_lmpr(scene)
    else:
        logger.warning('Skipping test scenes processing as LMPR data path not provided')


def main():
    train_scenes, val_scenes, test_scenes = get_scannet_splits()
    if args.lmpr_scannet_eval_path is None:
        test_scenes = []
    all_scenes = train_scenes + val_scenes + test_scenes

    if args.sample_frames:
        sample_frames_lmpr_batch(train_scenes, val_scenes, test_scenes)

    if args.extract_features:
        extract_features_batch(args.output_path, all_scenes)

    if args.extract_correspondences:
        extract_correspondences_batch(args.output_path, args.n_correspondences,
                                      args.num_keyframes,
                                      all_scenes)

    if args.compute_pose:
        compute_pose_batch(args.output_path, all_scenes,
                           method=args.method, use_mutuals=args.use_mutuals)

    if args.generate_trainval_data:
        generate_trainval_data_batch(args.output_path, all_scenes, method=args.method,
                                     use_mutuals=args.use_mutuals)


if __name__ == "__main__":
    # Initialize the logger
    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger)
    logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')

    main()
