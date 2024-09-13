import numpy as np
from numpy.linalg import det, eig


def is_elongated(covs_img, max_axis_ratio=np.inf):
    """
    Finds elongated covariances from input

    Args:
        covs_img ((n,2,2) np.array): n 2-dimensional covariance matrix inputs
        max_axis_ratio (float, optional): maximum ratio allowed between major and minor covariance matrix axes. Defaults to np.inf.

    Returns:
        (n,) np.array: boolean array indicating whether covariances are elongated
    """
    covs_arr = np.array(covs_img)
    eigvals, eigvecs = eig(covs_arr)
    return np.bitwise_or(eigvals[:,0] > max_axis_ratio*eigvals[:,1], eigvals[:,1] > max_axis_ratio*eigvals[:,0])

def is_out_of_bounds_pts(pts_3d, bounds_3d=np.array([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])):
    """
    Determines whether input points are out of 3D bounding box

    Args:
        pts_3d ((n,3) np.array): 3D points
        bounds_3d ((3,2) np.array, optional): minimum (first column) and maximum (second column) allowable pt values. Defaults to np.array([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]).

    Returns:
        (n,) np.array: boolean array indicating whether points are out of bounds
    """
    pts_arr = np.array(pts_3d)
    if len(pts_arr.shape) == 3 and pts_arr.shape[2] == 1:
        pts_arr = pts_arr.reshape(pts_arr.shape[:2])
    assert pts_arr.shape[1] == 3, "axis 1 shape is not 3"
    out_of_bounds = np.bitwise_or(pts_arr - bounds_3d[:,0] < 0, bounds_3d[:,1] - pts_arr < 0)
    # print(pts_arr.shape)
    # print(pts_arr)
    # print(out_of_bounds)
    # print(np.bitwise_or.reduce(out_of_bounds, axis=1))
    return np.bitwise_or.reduce(out_of_bounds, axis=1)

def is_out_of_bounds_area(covs_img, bounds_area=np.array([0., np.inf]), sigma=2):
    """
    Determines whether input covariance areas are out of bounds

    Args:
        covs_img ((n,2,2) np.array): n 2-dimensional covariance matrix inputs
        bounds_area ((2,) np.array, optional): min and maximum areas. Defaults to np.array([0., np.inf]).
        sigma (float, optional): standard deviation to use for computing area. Defaults to 2.

    Returns:
        (n,) np.array: boolean array indicating whether areas are out of bounds
    """
    covs_arr = np.array(covs_img)
    dets = det(covs_arr)
    areas = np.pi * np.sqrt(dets) * sigma**2
    return np.bitwise_or(areas < bounds_area[0], areas > bounds_area[1])
    
    
# def sigma_volume_from_det(cov, n_sigmas):
#     dim = cov.shape[0]
#     assert dim == 2 or dim == 3
#     coef = 1 if dim == 2 else 4/3
#     return coef*np.pi * np.sqrt(det(cov)) * (n_sigmas**dim)