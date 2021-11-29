import numpy as np
import torch
import itertools

from common.lie.torch import SE3q, SO3q, SE3, SO3
from common.torch_helpers import to_numpy


def pose_loss(pred, target, reduction='mean'):
    assert reduction in ['mean', 'none']

    pred_vec = pred.vec()
    target_vec = target.vec()

    # rot err
    if isinstance(pred, SE3) or isinstance(pred, SO3):
        rot_err = torch.abs(pred_vec[:, :9] - target_vec[:, :9])
        rot_loss = torch.sum(rot_err, dim=-1)
    else:
        # rotations are represented using quaternions: Need to take into account
        # that q and -q denotes the same rotation
        rot_err1 = torch.abs(pred_vec[:, :4] - target_vec[:, :4])
        rot_loss1 = torch.sum(rot_err1, dim=-1)
        rot_err2 = torch.abs(pred_vec[:, :4] + target_vec[:, :4])
        rot_loss2 = torch.sum(rot_err2, dim=-1)
        rot_loss = torch.min(rot_loss1, rot_loss2)

    # trans err
    if isinstance(pred, SE3) or isinstance(pred, SE3q):
        trans_err = torch.abs(pred_vec[:, -3:] - target_vec[:, -3:])
        trans_loss = torch.sum(trans_err, dim=-1)
        total_loss = rot_loss + trans_loss  # equal weighting for simplicity
    else:
        total_loss = rot_loss

    if reduction == 'mean':
        total_loss = torch.mean(total_loss)

    return total_loss


def pose_loss2(pred, target, reduction='mean'):
    assert reduction in ['mean', 'none']

    err = pred * target.inv()
    err_vec = err.vec()
    identity_vec = type(err).identity().vec().to(target.data.device)
    err_vec = err_vec - identity_vec[None, :]

    # rot err
    if isinstance(pred, SE3) or isinstance(pred, SO3):
        rot_loss = torch.sum(torch.abs(err_vec[:, :9]), dim=-1)
    else:
        raise NotImplementedError

    # trans err
    if isinstance(pred, SE3) or isinstance(pred, SE3q):
        trans_loss = torch.sum(torch.abs(err_vec[:, -3:]), dim=-1)
        total_loss = rot_loss + trans_loss  # equal weighting for simplicity
    else:
        total_loss = rot_loss

    if reduction == 'mean':
        total_loss = torch.mean(total_loss)

    return total_loss


def pose_loss_angularErr(pred, target, reduction='mean'):
    """Angular loss is angle(R1@R2.T)
    Does not perform as well as simple L1 loss on the 3x4 matrix, see
    Run scannet1/201019-095359(attention_dualThresh_rotAngleLoss)
    """

    assert reduction in ['mean', 'none']

    pred_mat = pred.as_matrix()
    target_mat = target.as_matrix()

    rot_err = target_mat[..., :3, :3] @ pred_mat[..., :3, :3].transpose(-2, -1)
    rot_trace = rot_err[..., 0, 0] + rot_err[..., 1, 1] + rot_err[..., 2, 2]
    rot_loss = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-0.999, max=0.999))

    trans_loss = torch.abs(pred_mat[..., :3, 3] - target_mat[..., :3, 3])

    loss = torch.mean(rot_loss) + torch.mean(trans_loss)
    return loss


# Unstable in practice
def pose_loss_log(pred, target, reduction='mean'):
    assert reduction in ['mean', 'none']

    target_rot_log = target.rot.log()
    pred_rot_log = pred.rot.log()
    rot_err = torch.abs(target_rot_log - pred_rot_log)
    trans_err = torch.abs(target.data[:, :3, 3] - pred.data[:, :3, 3])

    rot_loss = torch.sum(rot_err, dim=-1)
    trans_loss = torch.sum(trans_err, dim=-1)

    if reduction == 'mean':
        rot_loss = torch.mean(rot_loss)
        trans_loss = torch.mean(trans_loss)

    total_loss = rot_loss + trans_loss
    return total_loss


def pose_loss_split(pred, target, sigma_rt, reduction='mean'):

    trans_err = None
    if isinstance(pred, SE3q) or isinstance(pred, SO3q):
        rot_err = pred.data[:, :4] - target.data[:, :4]
        if isinstance(pred, SE3q):
            trans_err = pred.data[:, 4:7] - target.data[:, 4:7]
    else:
        pred_rot = pred.data[:, :3, :3]
        target_rot = target.data[:, :3, :3]

        rot_err = pred_rot.transpose(-1, -2) @ target_rot
        rot_err = rot_err[:, [0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]  # Consider off diagonals
        if isinstance(pred, SE3):
            trans_err = pred.data[:, :3, 3] - target.data[:, :3, 3]

    rot_loss = torch.sum(torch.abs(rot_err), dim=-1)
    if reduction == 'mean':
        rot_loss = torch.mean(rot_loss)

    if trans_err is not None:
        trans_loss = torch.sum(torch.abs(trans_err), dim=-1)
        if reduction == 'mean':
            trans_loss = torch.mean(trans_loss)
        loss = rot_loss * torch.exp(-sigma_rt[0]) + trans_loss * torch.exp(-sigma_rt[1]) + torch.sum(sigma_rt)
    else:
        loss = rot_loss  # No weighting

    return loss
