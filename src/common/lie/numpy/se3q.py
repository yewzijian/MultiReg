from typing import Dict, Union

import numpy as np

from common.lie.numpy.liegroupbase import _EPS, LieGroupBase
from common.lie.numpy import se3_common as se3c
from common.lie.numpy.so3q import SO3q


class SE3q(LieGroupBase):
    """Similar to SE3 but uses a quaternion + translation to represent the
    transformation"""

    DIM = 7
    DOF = 6
    N = 4  # Group transformation is 4x4 matrices
    name = 'SE3qNumpy'

    @staticmethod
    def identity(size: int = None, dtype=None, device=None) -> 'SE3q':
        if size is None:
            return SE3q(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype, device=device))
        else:
            return SE3q(np.tile(np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device),
                                (size, 1)))

    @staticmethod
    def rand(size: int = None, rot_mag=np.pi, trans_mag=1.0) -> 'SE3q':
        """Generates a random rotation

        Note that the rotation is only uniformly random if rot_mag is np.pi.
        """
        rot = SO3q.rand(size, rot_mag)
        trans = np.random.randn(3) * trans_mag if size is None else \
            np.random.randn(size, 3) * trans_mag

        return SE3q._from_rt(rot, trans)

    @staticmethod
    def _from_rt(rot: Union[SO3q, np.ndarray], trans: np.ndarray) -> 'SE3q':
        """Convenience function to concatenates the rotation and translation
        part into a SE(3) matrix

        Args:
            rot: ([*,] 4) or SO3q
            trans: ([*,] 3)

        Returns:
            SE(3) matrix
        """
        rotq = rot if isinstance(rot, np.ndarray) else rot.data
        vec = np.concatenate([rotq, trans], axis=-1)
        return SE3q(vec)

    @staticmethod
    def from_rtvec(vec: np.ndarray, normalize: bool = False) -> 'SE3q':
        """Constructs from 7D vector"""
        if normalize:
            normalized = se3c.normalize_quat_trans(vec)
            assert np.allclose(normalized, vec, atol=1e-3), 'Provided vec is too far from valid'
            return SE3q(normalized)
        else:
            assert se3c.is_valid_quat_trans(vec)
            return SE3q(vec)

    @staticmethod
    def from_matrix(mat: np.ndarray, normalize: bool = False, check: bool = True) -> 'SE3q':
        assert mat.shape[-2:] == (4, 4), 'Matrix should be of shape ([*,] 4, 4)'
        if normalize:
            normalized = se3c.normalize_matrix(mat)
            # Ensure that the matrix isn't nonsense in the first place
            assert np.allclose(normalized, mat, atol=1e-3), 'Original SE3 is too far from being valid'
            rot = SO3q.from_matrix(normalized[..., :3, :3], check=False).data
            trans = normalized[..., :3, 3]
        else:
            rot = SO3q.from_matrix(mat[..., :3, :3], check=check).data
            trans = mat[..., :3, 3]
        return SE3q._from_rt(rot, trans)

    def inv(self) -> 'SE3q':
        irot = SO3q(self.data[..., :4]).inv()
        trans = self.trans  # ([N, ] 3)
        itrans = -irot.transform(trans[..., None, :])[:, 0, :]
        return SE3q._from_rt(irot, itrans)

    def __mul__(self, other: 'SE3q') -> 'SE3q':
        """Quaternion multiplication.

        Computes qout = q1 * q2, where * is the Hamilton product between the two
        quaternions. Note that the Hamiltonian product is not commutative.
        """
        rot = self.rot * other.rot
        trans = self.rot.transform(other.data[..., None, 4:])[..., 0, :] + self.trans
        rtvec = np.concatenate([rot.data, trans], axis=-1)
        return SE3q(rtvec)

    def transform(self, pts: np.ndarray) -> np.ndarray:
        assert len(self.shape) == pts.ndim - 2
        transformed = self.rot.transform(pts) + self.data[..., None, 4:]
        return transformed

    @staticmethod
    def exp(vec: np.ndarray) -> 'SE3q':
        """Group exponential. Converts an element of tangent space (twist) to the
        corresponding element of the group SE(3).

        To be specific, computes expm(hat(vec)) with expm being the matrix
        exponential and hat() being the hat operator of SE(3).

        Args:
            vec: Twist vector ([N, ] 6)

        Returns:
            SE(3) matrix of size ([N, ] 7)

        Credits: Implementation is inspired by that in Sophus library
                 https://github.com/strasdat/Sophus/blob/master/sophus/se3.hpp
        """
        orig_shape = vec.shape
        if vec.ndim == 1:
            vec = vec[None, :]

        v, omega = vec[..., :3], vec[..., 3:]
        rot, theta = SO3q.exp_and_theta(omega)

        V = np.zeros_like(vec, shape=(*vec.shape[:-1], 3, 3))
        small_theta_mask = theta < _EPS

        if np.sum(small_theta_mask) > 0:
            V[small_theta_mask] = rot[small_theta_mask].as_matrix()
        if np.sum(~small_theta_mask) > 0:
            mask = ~small_theta_mask
            Omega = SO3q.hat(omega[mask])
            Omega_sq = Omega @ Omega

            theta_masked = theta[mask, None, None]
            theta2, theta3 = theta_masked ** 2, theta_masked ** 3
            s, c = np.sin(theta_masked), np.cos(theta_masked)

            V[~small_theta_mask] = np.identity(3) \
                                   - (c - 1.0) / theta2 * Omega \
                                   + (theta_masked - s) / theta3 * Omega_sq

        trans = V @ v[..., None]
        rtvec = SE3q._from_rt(rot, trans[..., 0])
        rtvec.data = np.reshape(rtvec.data, (*orig_shape[:-1], 7))
        return rtvec

    def log(self) -> np.ndarray:
        """Logarithm map.
        """
        raise NotImplementedError

    @staticmethod
    def hat(v: np.ndarray) -> np.ndarray:
        """hat-operator for SE(3)
        Specifically, it takes in the 6-vector representation (= twist) and returns
        the corresponding matrix representation of Lie algebra element.

        Args:
            v: Twist vector of size ([*,] 6). As with common convention, first 3
               elements denote translation.

        Returns:
            mat: se(3) element of size ([*,] 4, 4)
        """
        mat = np.zeros((*v.shape[:-1], 4, 4))
        mat[..., :3, :3] = SO3q.hat(v[..., 3:])  # Rotation
        mat[..., :3, 3] = v[..., :3]  # Translation

        return mat

    @staticmethod
    def vee(mat: np.ndarray) -> np.ndarray:
        """vee-operator for SE(3), i.e. inverse of hat() operator.

        Args:
            mat: ([*, ] 4, 4) matrix containing the 4x4-matrix lie algebra
                 representation. Omega must have the following structure:
                     |  0 -f  e  a |
                     |  f  0 -d  b |
                     | -e  d  0  c |
                     |  0  0  0  0 | .

        Returns:
            v: twist vector of size ([*,] 6)

        """
        v = np.zeros((*mat.shape[:-2], 6))
        v[..., 3:] = SO3q.vee(mat[..., :3, :3])
        v[..., :3] = mat[..., :3, 3]
        return v

    """Comparison function"""
    def compare(self, other: 'SE3q') -> Dict:
        """Compares two SO3 instances, returning the rotation error in degrees"""
        error = self * other.inv()
        e = {'rot_deg': SO3q.rotation_angle(error.rot) * 180 / np.pi,
             'trans': np.linalg.norm(self.trans - other.trans, axis=-1)}
        return e

    """Conversion functions"""
    @property
    def rot(self) -> SO3q:
        return SO3q(self.data[..., :4])

    @property
    def trans(self) -> np.ndarray:
        return self.data[..., 4:]

    def vec(self) -> np.ndarray:
        """Returns the flattened representation"""
        return self.data

    def as_quat_trans(self):
        """Return the 7D representation (quaternion, translation)
        First 4 columns contain the quaternion, last 3 columns contain translation
        """
        return self.data

    def as_matrix(self) -> np.ndarray:
        return se3c.quattrans2mat(self.data)

    def is_valid(self) -> bool:
        """Check whether the data is valid, e.g. if the underlying SE(3)
        representation has a valid rotation"""
        return se3c.is_valid_quat_trans(self.data)

    @property
    def shape(self):
        return self.data.shape[:-1]
