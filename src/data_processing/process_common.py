"""
Common functions for processing datasets
Includes extracting of FCGF features, pose computation using LMPR/RANSAC
"""

import glob
import itertools
import logging
import multiprocessing
import os
import sys

import cv2
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
from functools import partial
from sklearn.neighbors import NearestNeighbors
import torch
from tqdm import tqdm

from common.lie.numpy import SE3
import data_processing.lmpr.config
from data_processing.lmpr.descriptor.fcgf import FCGFNet
from data_processing.lmpr.checkpoints import CheckpointIO
from data_processing.lmpr.utils import load_config

_DIR = os.path.dirname(os.path.abspath(__file__))

VOXEL_SIZE = 0.025
FCGF_CHECKPOINT = os.path.join(_DIR, 'lmpr/pretrained/fcgf32_voxel25.pth')
REGBLOCK_CONFIG_PATH = os.path.join(_DIR, 'lmpr/config.yaml')  # Config of LMPR pairwise registration block
REGBLOCK_CHECKPOINT = os.path.join(_DIR, 'lmpr/pretrained/pairwise_reg.pt')

# Parameters for FastGR matching
FGR_VOXEL_SIZE = 0.02

# Subfolders
CLOUDS = 'raw_data'
FEATURES = 'features'
MATCHES = 'correspondences'
PAIRWISE_POSES = 'pairwise_poses'

# Filenames
FNAME_PLY_FORMAT = 'cloud_bin_{}.ply'
FNAME_POSE_FORMAT = 'cloud_bin_{}.info.txt'
FNAME_FEAT_FORMAT = '{}_{:03d}.npz'  # features file name
FNAME_CORR_FORMAT = '{}_{:03d}_{:03d}.npz'  # correspondences file name
FNAME_RELPOSE_FORMAT = 'relpose_{:03d}_{:03d}.npy'  # Computed pairwise pose
NUM_PROCESSES = 10  # number of threads to use for matching
METHOD_TO_FOLDER = {  # Maps method, use_mutuals to folder name
    ('FCGF_LMPR', True): 'RegBlock',
    ('FCGF_RANSAC', True): 'RANSAC',
    ('FGR', True): 'FastGR',
}

_logger = logging.getLogger(__name__)


def create_cloud(xyz: np.ndarray):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return cloud


def load_depth_and_convert_to_cloud(depth_fname: str, intrinsics):
    depth_array = cv2.imread(depth_fname, cv2.IMREAD_ANYDEPTH)
    depth_im = o3d.geometry.Image(depth_array)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_im, intrinsics)
    return pcd


def save_info(out_fname, scene, frame_no, pose):
    with open(out_fname, 'w') as fid:
        fid.write('{}\t{}\n'.format(scene, frame_no))
        for i in range(4):
            fid.write('\t'.join(map(str, pose[i, :])) + '\n')


def load_info(fname):
    with open(fname, 'r') as fid:
        fid = open(fname, 'r')
        first_line = fid.readline()
        tokens = first_line.split()
        pose = np.loadtxt(fid)

        info = {'scene': tokens[0],
                'orig_frame': tokens[1],
                'pose': pose}
    return info


def create_fcgf_model():
    device = torch.device('cuda')
    model = FCGFNet(in_channels=1, out_channels=32,
                    bn_momentum=0.05,
                    normalize_feature=True,
                    conv1_kernel_size=7,
                    D=3)
    _logger.info('Created model of type {}'.format(type(model)))
    _logger.info('Loading pretrained weights from {}'.format(FCGF_CHECKPOINT))
    state = torch.load(FCGF_CHECKPOINT)
    model.load_state_dict(state['state_dict'])
    model.to(device)
    model.eval()
    return model, device


def extract_fcgf(model, xyz, device):
    sel = ME.utils.sparse_quantize(xyz / VOXEL_SIZE, return_index=True)
    xyz_down = xyz[sel, :]  # Selected coordinates
    feats = np.ones([xyz_down.shape[0], 1])  # dummy: just contains ones
    coords = np.floor(xyz_down / VOXEL_SIZE)
    coordsC, featsC = ME.utils.sparse_collate(
        [torch.from_numpy(coords)],
        [torch.from_numpy(feats).float()])
    sinput = ME.SparseTensor(featsC, coords=coordsC).to(device)

    return xyz_down, model(sinput).F


def extract_features_batch(data_path, scenes):

    source_path = os.path.join(data_path, CLOUDS)
    target_path = os.path.join(data_path, FEATURES)

    os.makedirs(target_path, exist_ok=True)
    list_file = os.path.join(target_path, 'list.txt')
    f = open(list_file, 'w')

    model, device = create_fcgf_model()
    model.eval()

    for scene in scenes:
        num_ply_files = len(glob.glob(os.path.join(source_path, scene, '*.ply')))
        os.makedirs(os.path.join(target_path, scene), exist_ok=True)

        f.write('%s %d\n' % (scene, num_ply_files))
        for i in tqdm(range(num_ply_files), leave=False):
            save_fn = FNAME_FEAT_FORMAT.format(scene, i)
            in_fname = os.path.join(source_path, scene, FNAME_PLY_FORMAT.format(i))
            if os.path.exists(os.path.join(target_path, scene, save_fn)):
                _logger.debug('Features file already exist moving to the next example: {} - {}'.format(
                    scene, save_fn + '.npz'))
            else:
                # Extract features from a file
                pcd = o3d.io.read_point_cloud(in_fname)

                xyz_down, feature = extract_fcgf(model,
                                                 xyz=np.array(pcd.points),
                                                 device=device)

                np.savez_compressed(os.path.join(target_path, scene, save_fn),
                                    points=np.array(pcd.points),
                                    xyz=xyz_down,
                                    feature=feature.detach().cpu().numpy())
    f.close()


def extract_correspondences(data_path: str, num_correspondences: int, max_frames_apart: int, scene: str):
    logging.info('Matching keypoints for {}'.format(scene))
    src_folder = os.path.join(data_path, FEATURES, scene)
    dst_folder = os.path.join(data_path, MATCHES, scene)
    os.makedirs(os.path.join(dst_folder), exist_ok=True)

    # Read all features
    fnames = [f for f in os.listdir(src_folder) if f.endswith('.npz')]
    num_clouds = len(fnames)
    pairs = list(itertools.combinations(range(num_clouds), 2))
    np.random.seed(0)

    for (idx0, idx1) in pairs:
        if max_frames_apart > 0 and idx1 - idx0 >= max_frames_apart:
            # We only match frames which are within a certain time apart,
            # since we won't be considering graphs above this size
            continue

        out_path = os.path.join(dst_folder, FNAME_CORR_FORMAT.format(scene, idx0, idx1))
        if os.path.exists(out_path):
            logging.debug('Skipping feature matching as already exists for ' + out_path)
            continue

        pc_0_data = np.load(os.path.join(src_folder, FNAME_FEAT_FORMAT.format(scene, idx0)))
        feat0, kp0 = pc_0_data['feature'], pc_0_data['xyz']
        pc_1_data = np.load(os.path.join(src_folder, FNAME_FEAT_FORMAT.format(scene, idx1)))
        feat1, kp1 = pc_1_data['feature'], pc_1_data['xyz']

        # Sample 5000 points (if not enough, sample with replacement as in [1])
        inds0 = np.random.choice(len(kp0), num_correspondences,
                                 replace=False if len(kp0) >= num_correspondences else True)
        inds1 = np.random.choice(len(kp1), num_correspondences,
                                 replace=False if len(kp1) >= num_correspondences else True)
        kp0, feat0 = kp0[inds0], feat0[inds0]
        kp1, feat1 = kp1[inds1], feat1[inds1]

        # find the correspondence using nearest neighbor search in the feature space (two way)
        # For every point in cloud0, find best point in cloud1
        nn_search = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn_search.fit(feat1)
        nn_dists0, nn_indices0 = nn_search.kneighbors(X=feat0, n_neighbors=2, return_distance=True)
        # For every point in cloud1, find best point in cloud0
        nn_search.fit(feat0)
        nn_dists1, nn_indices1 = nn_search.kneighbors(X=feat1, n_neighbors=2, return_distance=True)
        # Compute mutual match
        mutuals = (nn_indices0[nn_indices1[:, 0], 0] == np.arange(len(kp1)))  # size = (n1,)
        ratios = nn_dists0[:, 0] / nn_dists0[:, 1]

        # Concatenate the correspondence coordinates
        xs = np.concatenate([kp0[nn_indices1[:, 0]], kp1], axis=1)  # (n0, 6)
        np.savez_compressed(out_path, x=xs, mutuals=mutuals, ratios=ratios)

    logging.info('Finished matching for {}'.format(scene))


def extract_correspondences_batch(data_path, num_correspondences, max_frames_apart, scenes):
    pool = multiprocessing.Pool(processes=NUM_PROCESSES)
    func = partial(extract_correspondences, data_path, num_correspondences, max_frames_apart)
    pool.map(func, scenes)
    pool.close()
    pool.join()


def compute_pose_regblock(data_path, model, use_mutuals, scene):

    _logger.info('Computing poses using Regblock for {}'.format(scene))
    matches_folder = os.path.join(data_path, MATCHES, scene)
    dst_folder = os.path.join(data_path, PAIRWISE_POSES, METHOD_TO_FOLDER[('FCGF_LMPR', use_mutuals)], scene)
    os.makedirs(dst_folder, exist_ok=True)

    # Compute relative poses
    matches_fpaths = glob.glob(os.path.join(matches_folder, '*.npz'))
    for matches_fpath in tqdm(matches_fpaths, ncols=80):
        idx0 = int(matches_fpath.split('_')[-2])
        idx1 = int(matches_fpath.split('_')[-1].split('.')[0])

        out_fname = os.path.join(dst_folder, FNAME_RELPOSE_FORMAT.format(idx0, idx1))
        if os.path.exists(out_fname):
            continue

        # Load correspondence file
        matches_data = np.load(matches_fpath)
        pts01 = matches_data['x'] if 'x' in matches_data else matches_data['correspondences']
        mutuals = matches_data['mutuals'].flatten().astype(np.bool)

        # Forward pass through the network to compute pose
        xs = torch.from_numpy(pts01).float().to(model.device)
        data = {'xs': xs[None, None, mutuals, :]} if use_mutuals else \
            {'xs': xs[None, None, :, :]}  # use only mutuals if desired
        est_data = model.filter_correspondences(data)

        rot = est_data['rot_est'][-1][0, ...].cpu().numpy()
        trans = est_data['trans_est'][-1][0, ...].cpu().numpy()
        rel_pose = np.eye(4)  # transforms from xyz0 to xyz1
        rel_pose[0:3, 0:3] = rot
        rel_pose[0:3, 3:4] = trans
        rel_pose = SE3.from_matrix(rel_pose, normalize=True)

        # Save out transformation matrix
        np.save(out_fname, rel_pose.as_matrix())


def compute_pose_ransac(data_path, use_mutuals, scene):

    _logger.info('Computing poses using RANSAC for {}'.format(scene))
    matches_folder = os.path.join(data_path, MATCHES, scene)
    dst_folder = os.path.join(data_path, PAIRWISE_POSES, METHOD_TO_FOLDER[('FCGF_RANSAC', use_mutuals)], scene)
    os.makedirs(dst_folder, exist_ok=True)

    # Compute relative poses
    matches_fpaths = glob.glob(os.path.join(matches_folder, '*.npz'))
    for matches_fpath in tqdm(matches_fpaths, ncols=80):
        idx0 = int(matches_fpath.split('_')[-2])
        idx1 = int(matches_fpath.split('_')[-1].split('.')[0])

        out_fname = os.path.join(dst_folder, FNAME_RELPOSE_FORMAT.format(idx0, idx1))
        if os.path.exists(out_fname):
            continue

        # Load correspondence file
        matches_data = np.load(matches_fpath)
        pts01 = matches_data['x'] if 'x' in matches_data else matches_data['correspondences']
        mutuals = matches_data['mutuals'].flatten().astype(np.bool)

        # Forward pass through the network to compute pose
        if use_mutuals:
            pts01 = pts01[mutuals, :]

        # Use Open3d's RANSAC function to compute transformation
        matches = np.tile(np.arange(len(pts01))[:, None], (1, 2))
        result_ransac = o3d.registration.registration_ransac_based_on_correspondence(
            source=create_cloud(pts01[:, 0:3]),
            target=create_cloud(pts01[:, 3:6]),
            corres=o3d.utility.Vector2iVector(matches),
            max_correspondence_distance=VOXEL_SIZE * 2,
            estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4)
        rel_pose = result_ransac.transformation

        # Save out transformation matrix
        np.save(out_fname, rel_pose)


def compute_pose_fastgr(data_path, scene):
    """Computes relative pose using FastGR. Parameters follow that of
    "Learning Transformation Synchronization"
    """
    logging.info('Starting FastGR matching for {}'.format(scene))

    src_folder = os.path.join(data_path, CLOUDS, scene)
    dst_folder = os.path.join(data_path, PAIRWISE_POSES, METHOD_TO_FOLDER[('FGR', True)], scene)
    os.makedirs(dst_folder, exist_ok=True)

    voxel_size = FGR_VOXEL_SIZE
    fnames = [f for f in os.listdir(src_folder) if f.endswith('.ply')]
    num_clouds = len(fnames)
    max_frames_apart = 30

    # Load point clouds and compute normals
    pcds, pcds_down, pcds_fpfh = [], [], []
    for i in range(num_clouds):
        pcd = o3d.io.read_point_cloud(os.path.join(src_folder, FNAME_PLY_FORMAT.format(i)))
        pcd_down = pcd.voxel_down_sample(voxel_size)

        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=60))
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        pcd_fpfh = o3d.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

        pcds.append(pcd)
        pcds_down.append(pcd_down)
        pcds_fpfh.append(pcd_fpfh)

    # Perform fast global registration
    pairs = list(itertools.combinations(range(num_clouds), 2))
    for (idx0, idx1) in pairs:
        if max_frames_apart > 0 and idx1 - idx0 >= max_frames_apart:
            # We only match frames which are within a certain time apart,
            # since we won't be considering graphs above this size
            continue

        out_fname = os.path.join(dst_folder, FNAME_RELPOSE_FORMAT.format(idx0, idx1))
        if os.path.exists(out_fname):
            continue

        result_fast = o3d.registration.registration_fast_based_on_feature_matching(
            pcds_down[idx0], pcds_down[idx1], pcds_fpfh[idx0], pcds_fpfh[idx1],
            o3d.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=voxel_size * 5))

        # Refine using ICP
        result_icp = o3d.registration.registration_icp(pcds[idx0], pcds[idx1], voxel_size * 1.5,
                                                       result_fast.transformation,
                                                       o3d.registration.TransformationEstimationPointToPlane())
        rel_pose = result_icp.transformation

        # Save out transformation matrix
        np.save(out_fname, rel_pose)

    logging.info('Finished FastGR matching for {}'.format(scene))


def compute_pose_batch(data_path, scenes, method, use_mutuals: bool):
    if method == 'FCGF_LMPR':
        # Get model
        cfg = load_config(REGBLOCK_CONFIG_PATH)
        model = data_processing.lmpr.config.get_model(cfg)
        model.eval()
        # Load checkpoints
        checkpoint_io = CheckpointIO('', model=model)
        checkpoint_io.load(REGBLOCK_CHECKPOINT)
        os.makedirs(os.path.join(data_path, PAIRWISE_POSES), exist_ok=True)

        with torch.no_grad():
            for scene in scenes:
                compute_pose_regblock(data_path, model, use_mutuals, scene)

    elif method == 'FCGF_RANSAC':
        for scene in scenes:
            compute_pose_ransac(data_path, use_mutuals, scene)

    elif method == 'FGR':
        pool = multiprocessing.Pool(processes=NUM_PROCESSES//2)
        func = partial(compute_pose_fastgr, data_path)
        pool.map(func, scenes)
        pool.close()
        pool.join()

    else:
        raise NotImplementedError('Invalid pose estimation method')


def generate_traj(data_path, method, use_mutuals, scene):
    data_folder = os.path.join(data_path, PAIRWISE_POSES, METHOD_TO_FOLDER[(method, use_mutuals)], scene)
    pose_fpaths = glob.glob(os.path.join(data_folder, '*.npy'))

    out_fname = os.path.join(data_folder, 'traj.txt')
    if os.path.exists(out_fname):
        _logger.info('Skipping {} as already generated'.format(scene))
        return
    out_file = open(out_fname, 'w')

    for pose_fpath in pose_fpaths:
        idx0 = int(pose_fpath.split('_')[-2])
        idx1 = int(pose_fpath.split('_')[-1].split('.')[0])
        pose = np.load(pose_fpath)
        inv_pose = SE3.from_matrix(pose).inv().as_matrix()

        out_file.write('{}\t{}\tTrue\n'.format(idx0, idx1))  # We don't compute overlap, so add dummy
        for row in inv_pose:
            out_file.write('\t'.join(map(str, row)) + '\n')

    out_file.close()


def generate_traj_batch(data_path, scenes, method, use_mutuals):
    """Generates traj.txt to compare with LMPR evaluation code"""
    _logger.info('Generating traj.txt')
    for scene in tqdm(scenes, ncols=80, leave=False):
        generate_traj(data_path, method, use_mutuals, scene)


def compute_median_point_distance(points_i, points_j, transform_ij: SE3) -> float:
    """Computes median point distance in overlapping region. This is used in
    Learn2Sync and LMPR as a heuristic of the registration quality
    """
    points_i_transformed = transform_ij.transform(points_i)

    tree_j = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(np.asarray(points_j))
    distances, indices = tree_j.kneighbors(points_i_transformed)  # [np, 1], [np, 1]
    idx = np.where(distances < 0.2)[0]
    sigma = np.inf if len(idx) == 0 else np.median(distances[idx])
    return sigma


def generate_trainval_data(data_path, method, use_mutuals, scene):
    _logger.info('Generating trainval data for {}'.format(scene))
    cloud_folder = os.path.join(data_path, CLOUDS, scene)
    meas_poses_folder = os.path.join(data_path, PAIRWISE_POSES, METHOD_TO_FOLDER[(method, use_mutuals)], scene)
    result_fname = os.path.join(data_path, PAIRWISE_POSES, METHOD_TO_FOLDER[method, use_mutuals],
                                '{}.npz'.format(scene))
    if os.path.exists(result_fname):
        _logger.info('Skipping generating of npz file for {} as already exists.'.format(scene))
        return
    num_clouds = len(glob.glob(os.path.join(cloud_folder, '*.ply')))

    clouds = [np.zeros([])] * num_clouds
    Tstar = np.zeros((num_clouds, 4, 4))
    Tij_meas = np.zeros((num_clouds, num_clouds, 4, 4))
    Tij_gt = np.zeros((num_clouds, num_clouds, 4, 4))
    aerr = np.full((num_clouds, num_clouds), np.inf)
    terr = np.full((num_clouds, num_clouds), np.inf)
    sigma = np.full((num_clouds, num_clouds), np.inf)

    # Collates the groundtruth absolute poses
    for i in range(num_clouds):
        cloud_meta = load_info(os.path.join(cloud_folder, FNAME_POSE_FORMAT.format(i)))
        Tstar[i, :, :] = cloud_meta['pose']
        clouds[i] = np.asarray(
            o3d.io.read_point_cloud(os.path.join(cloud_folder, FNAME_PLY_FORMAT.format(i))).points)
    Tstar_se3 = SE3.from_matrix(Tstar, normalize=True)
    Tstar = Tstar_se3.as_matrix()

    # Collates pairwise measured poses
    poses_files = glob.glob(os.path.join(meas_poses_folder, '*.npy'))
    for poses_fpath in poses_files:
        i = int(poses_fpath.split('_')[-2])
        j = int(poses_fpath.split('_')[-1].split('.')[0])
        pose_data = np.load(poses_fpath)

        rel_meas = SE3.from_matrix(pose_data)
        rel_gt = Tstar_se3[j].inv() * Tstar_se3[i]
        meas_err = rel_gt.compare(rel_meas)
        median_pt_dist = compute_median_point_distance(clouds[i], clouds[j], rel_meas)

        Tij_meas[i, j] = rel_meas.as_matrix()
        Tij_gt[i, j] = rel_gt.as_matrix()
        aerr[i, j] = meas_err['rot_deg']
        terr[i, j] = meas_err['trans']
        sigma[i, j] = median_pt_dist

        Tij_meas[j, i] = rel_meas.inv().as_matrix()
        Tij_gt[j, i] = rel_gt.inv().as_matrix()
        aerr[j, i] = meas_err['rot_deg']
        terr[j, i] = meas_err['trans']
        sigma[j, i] = median_pt_dist

    # Output to file
    np.savez(result_fname,
             Tstar=Tstar, Tij_meas=Tij_meas, Tij_gt=Tij_gt, aerr=aerr, terr=terr,
             sigma=sigma)
    _logger.info('Done generating trainval data for {}'.format(scene))


def generate_trainval_data_batch(data_path, scenes, method, use_mutuals):
    _logger.info('Generating train/val data...')
    pool = multiprocessing.Pool(processes=NUM_PROCESSES)
    func = partial(generate_trainval_data, data_path, method, use_mutuals)
    pool.map(func, scenes)
    pool.close()
    pool.join()
