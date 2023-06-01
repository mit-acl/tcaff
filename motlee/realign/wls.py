import numpy as np

from motlee.utils.transform import transform

def wls(pts1, pts2, weights=None):
    if weights is None:
        weights = np.ones((pts1.shape[0],1))
    weights = weights.reshape((-1,1))
    mean1 = (np.sum(pts1 * weights, axis=0) / np.sum(weights)).reshape(-1)
    mean2 = (np.sum(pts2 * weights, axis=0) / np.sum(weights)).reshape(-1)
    det1_mean_reduced = pts1 - mean1
    det2_mean_reduced = pts2 - mean2
    assert det1_mean_reduced.shape == det2_mean_reduced.shape
    H = det1_mean_reduced.T @ (det2_mean_reduced * weights)
    U, s, V = np.linalg.svd(H)
    R = U @ V.T
    t = mean1.reshape((-1,1)) - R @ mean2.reshape((-1,1))
    T = np.concatenate([np.concatenate([R, t], axis=1), np.hstack([np.zeros((1, R.shape[0])), [[1]]])], axis=0)
    return T

def wls_residual(pts1, pts2, weights, T):
    pts2_transformed = transform(T, pts2, stacked_axis=0)
    # (pts1 - pts2_transformed).T @ np.diag(weights) @ (pts1 - pts2_transformed)
    return np.sum(np.linalg.norm(pts1 - pts2_transformed, axis=1)**2 * weights)