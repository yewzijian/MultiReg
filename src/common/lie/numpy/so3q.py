"""SO3 using quaternion representation
Note that for edge cases, we only use up to first order approximation
"""

from typing import Dict, Optional

import numpy as np

from common.lie.numpy import so3_common as so3c
from common.lie.numpy.liegroupbase import LieGroupBase
from common.math_numpy.rand import uniform_2_sphere


class SO3q(LieGroupBase):

    DIM = 4
    DOF = 3
    N = 3  # Group transformation is 3x3 matrices
    name = 'SO3qNumpy'

    @staticmethod
    def identity(size: int = None) -> 'SO3q':
        if size is None:
            return SO3q(np.array([1.0, 0.0, 0.0, 0.0]))
        else:
            return SO3q(np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (size, 1)))

    @staticmethod
    def from_quaternion(q, normalize: bool = False) -> 'SO3q':
        assert q.shape[-1] == 4
        if normalize:
            q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        elif not np.allclose(np.linalg.norm(q, axis=-1), 1.):
            raise AssertionError('Quaternion must be unit length')
        return SO3q(q)

    @staticmethod
    def from_matrix(mat: np.ndarray, normalize: bool = False, check: bool = True) -> 'SO3q':
        if normalize:
            normalized = so3c.normalize_rotmat(mat)
            assert np.allclose(normalized, mat, atol=1e-3)
            return SO3q(so3c.rotmat2quat(normalized))
        else:
            if check:
                assert so3c.is_valid_rotmat(mat)
            return SO3q(so3c.rotmat2quat(mat))

    @staticmethod
    def _rand_quaternion(size: int = 1) -> np.ndarray:
        if size is None:
            u1 = np.random.rand()
            u2 = np.random.rand() * 2.0 * np.pi
            u3 = np.random.rand() * 2.0 * np.pi
        else:
            u1 = np.random.rand(size)
            u2 = np.random.rand(size) * 2.0 * np.pi
            u3 = np.random.rand(size) * 2.0 * np.pi

        a = np.sqrt(1 - u1)
        b = np.sqrt(u1)

        q = np.stack([a * np.sin(u2),
                      a * np.cos(u2),
                      b * np.sin(u3),
                      b * np.cos(u3)], axis=-1)
        return q

    @staticmethod
    def rand(size: Optional[int] = None, rot_mag=np.pi) -> 'SO3q':
        """Generates a random rotation

        Note that the rotation is only uniformly random if rot_mag is np.pi,
        which we will reference the Sophus implementation, which is based on
          http://planning.cs.uiuc.edu/node198.html
        Otherwise, we first sample a direction and angle uniformly. Note that
        this does not generate uniform rotations as it biases towards smaller
        angles.
        """
        if rot_mag == np.pi:
            q = SO3q._rand_quaternion(size)
            return SO3q(q)
        else:
            # First sample axis, then angle
            rand_dir = uniform_2_sphere(size)
            # Sample angle (we assume uniform distributed angles, which
            # actually does not generate uniform rotations)
            r = np.random.rand(size) if size is not None else np.random.rand()
            theta = r * (rot_mag / np.pi)
            omega = rand_dir * theta[:, None] if size is not None else rand_dir * theta
            return SO3q.exp(omega)

    @staticmethod
    def exp(omega: np.ndarray) -> 'SO3q':
        q, _ = SO3q.exp_and_theta(omega)
        return q

    @staticmethod
    def exp_and_theta(omega: np.ndarray) -> ('SO3q', np.ndarray):
        """Same as exp() but also returns theta (rotation angle in radians)

        This follows the implementation from ceres solver
        https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/rotation.h
        """
        orig_shape = omega.shape
        if omega.ndim == 1:
            omega = omega[None, :]

        theta = np.linalg.norm(omega, axis=-1)  # ([N,] 1)
        zero_theta = theta == 0.0
        quats = np.empty_like(omega, shape=(*omega.shape[:-1], 4))

        # For small rotations, use taylor 1st order approximation
        if np.sum(zero_theta) > 0:
            quats[zero_theta] = np.concatenate([
                np.array([[1.0]]),
                omega[zero_theta] * 0.5],
                axis=-1)

        # Standard conversion is numerical stable for non-zero rotations
        if np.sum(~zero_theta) > 0:
            mask = ~zero_theta
            theta_masked = theta[mask][:, None]  # (N, 1)
            half_theta_masked = 0.5 * theta_masked
            k = np.sin(half_theta_masked) / theta_masked
            quats[mask] = \
                np.concatenate([np.cos(half_theta_masked), omega[mask] * k],
                               axis=-1)

        quats = np.reshape(quats, (*orig_shape[:-1], 4))
        return SO3q(quats), theta

    def log(self) -> np.ndarray:
        """Converts quaternion to angle axis
        This follows the implementation from ceres solver
        https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/rotation.h
        """
        quat = self.data
        if quat.ndim == 1:
            quat = quat[None, :]  # (N, 4)

        omegas = np.empty_like(quat, shape=(*quat.shape[:-1], 3))
        sin_theta = np.linalg.norm(quat[..., 1:], axis=-1)  # (N, )
        zero_theta = sin_theta == 0.0  # (N, )

        if np.sum(~zero_theta) > 0:
            mask = ~zero_theta
            cos_theta_masked = quat[mask, 0]

            two_theta0 = 2.0 * np.arctan2(sin_theta[mask], cos_theta_masked)
            two_theta1 = 2.0 * np.arctan2(-sin_theta[mask], -cos_theta_masked)
            two_theta = np.where(cos_theta_masked >= 0, two_theta0, two_theta1)

            k = two_theta / sin_theta
            omegas[mask] = quat[mask, 1:] * k[:, None]

        if np.sum(zero_theta) > 0:
            # Taylor 1st order approximation
            omegas[zero_theta] = quat[zero_theta, 1:] * 2.0

        omegas = np.reshape(omegas, (*self.data.shape[:-1], 3))
        return omegas

    def inv(self) -> 'SO3q':
        """Quaternion inverse, which is equivalent to its conjugate"""
        return SO3q(so3c.quat_inv(self.data))

    def __mul__(self, other: 'SO3q') -> 'SO3q':
        """Quaternion multiplication.

        Computes qout = q1 * q2, where * is the Hamilton product between the two
        quaternions. Note that the Hamiltonian product is not commutative.
        """
        return SO3q(so3c.quat_mul(self.data, other.data))

    def transform(self, pts: np.ndarray) -> np.ndarray:
        assert len(self.shape) == pts.ndim - 2
        transformed = so3c.quat_rot(self.data, pts)
        return transformed

    @staticmethod
    def hat(v: np.ndarray) -> np.ndarray:
        """Maps a vector to a 3x3 skew symmetric matrix."""
        return so3c.hat(v)

    @staticmethod
    def vee(mat: np.ndarray) -> np.ndarray:
        """Inverse of hat operator, i.e. transforms skew-symmetric matrix to
        3-vector
        """
        return so3c.vee(mat)

    """Comparison functions"""
    def rotation_angle(self) -> np.ndarray:
        """Returns the rotation angle in radians"""
        sin_theta = np.minimum(np.linalg.norm(self.data[..., 1:], axis=-1), 1.0)  # (N, )
        return np.arcsin(sin_theta) * 2.0

    def compare(self, other: 'SO3q') -> Dict:
        """Compares two SO3 instances, returning the rotation error in degrees"""
        error = self * other.inv()
        e = {'rot_deg': SO3q.rotation_angle(error) * 180 / np.pi}
        return e

    """Conversion functions"""
    def vec(self) -> np.ndarray:
        """Returns the flattened representation"""
        return self.data

    def as_quaternion(self) -> np.ndarray:
        return self.data

    def as_matrix(self) -> np.ndarray:
        return so3c.quat2rotmat(self.data)

    def is_valid(self) -> bool:
        return so3c.is_valid_quaternion(self.data)

    @property
    def shape(self):
        return self.data.shape[:-1]

