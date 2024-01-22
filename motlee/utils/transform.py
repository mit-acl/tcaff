import numpy as np
from scipy.spatial.transform import Rotation as Rot

def transform_vec(T, vec):
    unshaped_vec = vec.reshape(-1)
    resized_vec = np.concatenate(
        [unshaped_vec, np.zeros((T.shape[0] - 1 - unshaped_vec.shape[0]))]).reshape(-1)
    resized_vec = np.concatenate(
        [resized_vec, np.ones((T.shape[0] - resized_vec.shape[0]))]).reshape((-1, 1))
    transformed = T @ resized_vec
    return transformed.reshape(-1)[:unshaped_vec.shape[0]].reshape(vec.shape) 

# TODO: rename stacked_axis and default to 0
# TODO: handle tensors
def transform(T, vecs, stacked_axis=1):
    if len(vecs.reshape(-1)) == 2 or len(vecs.reshape(-1)) == 3:
        return transform_vec(T, vecs)
    vecs_horz_stacked = vecs if stacked_axis==1 else vecs.T
    zero_padded_vecs = np.vstack(
        [vecs_horz_stacked, np.zeros((T.shape[0] - 1 - vecs_horz_stacked.shape[0], vecs_horz_stacked.shape[1]))]
    )
    one_padded_vecs = np.vstack(
        [zero_padded_vecs, np.ones((1, vecs_horz_stacked.shape[1]))]
    )
    transformed = T @ one_padded_vecs
    transformed = transformed[:vecs_horz_stacked.shape[0],:] 
    return transformed if stacked_axis == 1 else transformed.T

# TODO: use einsum here
def transform_covariance(T, covariance):
    n = covariance.shape[0]
    R = T[:n,:n]
    transformed_covariance = R @ covariance @ R.T
    return transformed_covariance

def transform_covariances(T, covariances):
    transformed_covariances = []
    for covariance in covariances:
        transformed_covariances.append(transform_covariance(T, covariance))
    return np.array(transformed_covariances)

# gives scalar value to magnitude of translation
def T_mag(T, deg2m):
    R = Rot.from_matrix(T[0:3, 0:3])
    t = T[0:2, 3]
    rot_mag = R.as_euler('xyz', degrees=True)[2] / deg2m
    t_mag = np.linalg.norm(t)
    return np.abs(rot_mag) + np.abs(t_mag)

def transform_2_xypsi(T):
    dim = T.shape[1] - 1
    if dim == 2:
        T = T2d_2_T3d(T)
    x = T[0,3]
    y = T[1,3]
    psi = Rot.from_matrix(T[:3,:3]).as_euler('xyz', degrees=False)[2]
    return x, y, psi

def xypsi_2_transform(x, y, psi):
    T = np.eye(4)
    T[:2,:2] = Rot.from_euler('xyz', [0, 0, psi]).as_matrix()[:2,:2]
    T[0,3] = x
    T[1,3] = y
    return T

def pos_quat_to_transform(pos, quat):
    T = np.eye(4)
    T[:3,:3] = Rot.from_quat(quat).as_matrix()
    T[:3,3] = pos.reshape(-1)
    return T

def T3d_2_T2d(T3d):
    T2d = np.delete(np.delete(T3d, 2, axis=0), 2, axis=1)
    return T2d

def T2d_2_T3d(T2d):
    T3d = np.eye(4)
    T3d[:2,:2] = T2d[:2,:2]
    T3d[:2,3] = T2d[:2,2]
    return T3d