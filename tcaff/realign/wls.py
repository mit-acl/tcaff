import numpy as np
from numpy.linalg import det

from tcaff.utils.transform import transform

def wls(pts1, pts2, weights=None):
    """Aruns method for 3D registration

    Args:
        pts1 (numpy.array, shape(n,2 or 3)): initial set of points
        pts2 (numpy.array, shape(n,2 or 3)): second set of points that should be aligned to first set
        weights (numpy.array, shape(n), optional): weights applied to associations. Defaults to None.

    Returns:
        numpy.array, shape(3,3 or 4,4): rigid body transformation to align pts2 with pts1
    """
    if weights is None:
        weights = np.ones((pts1.shape[0],1))
    weights = weights.reshape((-1,1))
    mean1 = (np.sum(pts1 * weights, axis=0) / np.sum(weights)).reshape(-1)
    mean2 = (np.sum(pts2 * weights, axis=0) / np.sum(weights)).reshape(-1)
    pts1_mean_reduced = pts1 - mean1
    pts2_mean_reduced = pts2 - mean2
    assert pts1_mean_reduced.shape == pts2_mean_reduced.shape
    H = pts1_mean_reduced.T @ (pts2_mean_reduced * weights)
    U, s, Vh = np.linalg.svd(H)
    R = U @ Vh
    if np.allclose(det(R), -1.0):
        Vh_prime = Vh.copy()
        Vh_prime[-1,:] *= -1.0
        R = U @ Vh_prime
    t = mean1.reshape((-1,1)) - R @ mean2.reshape((-1,1))
    T = np.concatenate([np.concatenate([R, t], axis=1), np.hstack([np.zeros((1, R.shape[0])), [[1]]])], axis=0)
    return T

def wls_residual(pts1, pts2, weights, T):
    pts2_transformed = transform(T, pts2, stacked_axis=0)
    # (pts1 - pts2_transformed).T @ np.diag(weights) @ (pts1 - pts2_transformed)
    return np.sum(np.linalg.norm(pts1 - pts2_transformed, axis=1)**2 * weights)