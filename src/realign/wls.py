import numpy as np

def wls(pts1, pts2, weights):
    mean1 = (np.sum(pts1 * weights, axis=0) / np.sum(weights)).reshape(-1)
    mean2 = (np.sum(pts2 * weights, axis=0) / np.sum(weights)).reshape(-1)
    det1_mean_reduced = pts1 - mean1
    det2_mean_reduced = pts2 - mean2
    assert det1_mean_reduced.shape == det2_mean_reduced.shape
    H = det1_mean_reduced.T @ (det2_mean_reduced * weights)
    U, s, V = np.linalg.svd(H)
    R = U @ V.T
    t = mean1.reshape((3,1)) - R @ mean2.reshape((3,1))
    T = np.concatenate([np.concatenate([R, t], axis=1), np.array([[0,0,0,1]])], axis=0)
    return T