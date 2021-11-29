"""Functions for random sampling"""
import numpy as np
from scipy.stats import special_ortho_group


def uniform_2_sphere(size: int = 1):
    """Uniform sampling on a 2-sphere

    Source: https://gist.github.com/andrewbolster/10274979

    Args:
        size: Number of vectors to sample

    Returns:
        Random Vector (np.ndarray) of size (size, 3) with norm 1.
        If size is None returned value will have size (3,)

    """
    if size is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, size)
        cos_theta = np.random.uniform(-1.0, 1.0, size)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)
