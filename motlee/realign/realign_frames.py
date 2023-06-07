import open3d as o3d
import numpy as np
from numpy.linalg import inv, norm
from scipy.optimize import linear_sum_assignment
if o3d.__DEVICE_API__ == 'cuda':
    import open3d.cuda.pybind.t.pipelines.registration as treg
else:
    import open3d.cpu.pybind.t.pipelines.registration as treg

from motlee.realign.wls import wls, wls_residual
from motlee.utils.transform import transform, T2d_2_T3d

import clipperpy

# STARTING A LIST OF MAGIC NUMBERS HERE
NUM_CONES_REQ = 8
DATA_ASSOCIATION_METHOD = 'icp'
# DATA_ASSOCIATION_METHOD = 'clipper'
ONLY_STRONG_CORRESPONDENCES = True
ICP_MAX_DIST = 1.0
CLIPPER_SIG = 0.3
CLIPPER_EPS = 0.4

def detections2pointcloud(detections, org_by_tracks):
    dets_cp = []
    if org_by_tracks:
        pass
    else:
        for frame in detections:
            for detection in frame:
                if detection is not None:
                    dets_cp.append(np.concatenate([detection, [[0]]], axis=0).reshape(-1))
        dets_np = np.array(dets_cp)
    return o3d.t.geometry.PointCloud(dets_np)

def run_icp(detections1, detections2, initial_guess=np.eye(4), max_dist=.5):
    trans_init = initial_guess
    # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
    estimation = treg.TransformationEstimationPointToPoint()
    # Search distance for Nearest Neighbour Search [Hybrid-Search is used].
    max_correspondence_distance = max_dist
    # Initial alignment or source to target transform.
    init_source_to_target = trans_init
    # Convergence-Criteria for Vanilla ICP
    criteria = treg.ICPConvergenceCriteria(relative_fitness=0.0000001,
                                        relative_rmse=0.0000001,
                                        max_iteration=30)
    # Down-sampling voxel-size. If voxel_size < 0, original scale is used.
    voxel_size = -1
    reg_point_to_point = treg.icp(detections2, detections1, max_correspondence_distance,
                            init_source_to_target, estimation, criteria,
                            voxel_size)
    return reg_point_to_point

def clipper_data_association(detections1, detections2, sigma=0.30, epsilon=0.4):
    """
    Parameters
    ----------
    detections1 : (n,3) np.array
    detections2 : (m,3) np.array

    Return
    ------
    Ain : (p,2) np.array (int) - inlier set. First column contains indices from detections1, 
        second contains corresponding indices from detections2
    """
    iparams = clipperpy.invariants.EuclideanDistanceParams()
    iparams.sigma = sigma
    iparams.epsilon = epsilon
    invariant = clipperpy.invariants.EuclideanDistance(iparams)

    params = clipperpy.Params()
    clipper = clipperpy.CLIPPER(invariant, params)

    n = len(detections1)
    m = len(detections2)
    A = clipperpy.utils.create_all_to_all(n, m)

    clipper.score_pairwise_consistency(detections1.T, detections2.T, A)
    clipper.solve()
    Ain = clipper.get_selected_associations()
    
    return Ain

def clipper_inlier_associations(Ain, pts1, pts2, weights1=None, weights2=None):
    """
    Returns two lists of ordered pts that have been associated with each other

    Args:
        Ain (np.array(p,2, int)): inlier set from CLIPPER
        pts1 (np.array(n,2|3): first set of points
        pts2 (np.array(m,2|3)): second set of points
        weights1 (np.array(n,)): weights
        weights2 (np.array(m,)): weights

    Returns:
        pts1_corres (np.array(p,2|3)): reordered points from pts1 corresponding to pts2_corres
        pts2_corres (np.array(p,2|3)): reordered points from pts2 corresponding to pts1_corres
    """
    assert (weights1 is None and weights2 is None) or \
        (weights1 is not None and weights2 is not None)
    if Ain.shape[0] == 0:
        return np.array(), np.array()
    print(pts1)
    print(Ain)
    pts1_corres = np.zeros((Ain.shape[0], pts1.shape[1]))
    pts2_corres = np.zeros((Ain.shape[0], pts2.shape[1]))
    weights1_corres = np.zeros((Ain.shape[0]))
    weights2_corres = np.zeros((Ain.shape[0]))
    
    for i in range(Ain.shape[0]):
        pts1_corres[i,:] = pts1[Ain[i,0]]
        pts2_corres[i,:] = pts2[Ain[i,1]]
        if weights1 is not None and weights2 is not None:
            weights1_corres[i] = weights1[Ain[i,0]]
            weights2_corres[i] = weights2[Ain[i,1]]
    return pts1_corres, pts2_corres, weights1_corres, weights2_corres

def icp_data_association(cones1_input, cones2_input, T_current, ages1=None, ages2=None):
    if ages1 is None and ages2 is None:
        cones1 = []; ages1 = []
        cones2 = []; ages2 = []
        for cone in cones1_input:
            cones1.append(cone.state[:2, :].reshape(-1).tolist() + [0])
            ages1.append([cone.ell])
        for cone in cones2_input:
            cones2.append(cone.state[:2, :].reshape(-1).tolist() + [0])
            ages2.append([cone.ell])
        if len(cones1) < NUM_CONES_REQ or len(cones2) < NUM_CONES_REQ: 
            return None, None, T_current
        cones1 = np.array(cones1)
        ages1 = np.array(ages1)
        cones2 = np.array(cones2)
        ages2 = np.array(ages2)
    else:
        cones1 = np.array(cones1_input); cones2 = np.array(cones2_input)
    cone1_ptcld = o3d.t.geometry.PointCloud(cones1)
    cone2_ptcld = o3d.t.geometry.PointCloud(cones2)
    correspondence_set2 = run_icp(cone1_ptcld, cone2_ptcld, T_current, max_dist=ICP_MAX_DIST).correspondences_.numpy()
        
    cones1_reordered = np.zeros(cones2.shape)
    ages1_reordered = np.zeros((cones2.shape[0], 1))
    for i in range(cones2.shape[0]):
        if correspondence_set2[i] == -1: continue
        try:
            cones1_reordered[i, :] = cones1[correspondence_set2[i], :]
        except:
            import ipdb; ipdb.set_trace()
        ages1_reordered[i, 0] = ages1[correspondence_set2[i], 0]
    no_correspond_idx = [i for i,x in enumerate(correspondence_set2.reshape(-1).tolist()) if x==-1]
    cones2 = np.delete(cones2, no_correspond_idx, axis=0)
    cones1_reordered = np.delete(cones1_reordered, no_correspond_idx, axis=0)
    ages2 = np.delete(ages2, no_correspond_idx, axis=0)
    ages1_reordered = np.delete(ages1_reordered, no_correspond_idx, axis=0)
    
    # if len(cones2_new) < NUM_CONES_REQ:
    if len(cones2) < NUM_CONES_REQ:
        return None, None, T_current
        
    # weights = 1/(.01 + ages2_new * ages1_new)
    weights = 1/(.01 + ages2 * ages1_reordered)
    
    # return cones1_new, cones2_new, weights
    return cones1_reordered, cones2, weights

def realign_cones(cones1_input, cones2_input, T_current):
    if len(cones1_input) == 0 or len(cones2_input) == 0:
        return T_current, None, None
    if DATA_ASSOCIATION_METHOD == 'icp':
        cones1_reordered, cones2, weights1 = icp_data_association(cones1_input, cones2_input, T_current)
        cones2_reordered, cones1, weights2 = icp_data_association(cones2_input, cones1_input, inv(T_current))
        if cones1_reordered is None or cones2_reordered is None or cones1 is None or cones2 is None:
            return T_current, None, None
        c1_out = np.concatenate([cones1_reordered, cones1], axis=0)
        c2_out = np.concatenate([cones2, cones2_reordered], axis=0)
        weights_all = np.concatenate([weights1, weights2], axis=0)

        if ONLY_STRONG_CORRESPONDENCES:
            to_delete = []
            for i in range(c1_out.shape[0]):
                no_other_pair = True
                for j in range(c2_out.shape[0]):
                    if i == j: continue
                    if np.allclose(c1_out[i,:], c1_out[j,:]) and np.allclose(c2_out[i,:], c2_out[j,:]):
                        no_other_pair = False
                if no_other_pair:
                    to_delete.append(i)

            c1_out = np.delete(c1_out, to_delete, axis=0)
            c2_out = np.delete(c2_out, to_delete, axis=0)
            weights_all = np.delete(weights_all, to_delete, axis=0)
        
    elif DATA_ASSOCIATION_METHOD == 'clipper':
        cones1 = np.array([c.state.reshape(-1)[:2] for c in cones1_input])
        cones2 = np.array([c.state.reshape(-1)[:2] for c in cones2_input])
        ages1 = np.array([c.ell for c in cones1_input])
        ages2 = np.array([c.ell for c in cones2_input])
        Ain = clipper_data_association(
            cones1, cones2, sigma=CLIPPER_SIG, epsilon=CLIPPER_EPS)
        print(cones1)
        print(Ain)
        c1_out, c2_out, ages1_out, ages2_out = \
            clipper_inlier_associations(Ain, cones1, cones2, ages1, ages2)
        weights_all = 1/(.01 + ages1_out * ages2_out)   

    num_cones = len(c1_out) / (1 if DATA_ASSOCIATION_METHOD=='clipper' else 2)
    if num_cones < NUM_CONES_REQ:
        return T_current, None, None

    T_new = wls(c1_out, c2_out, weights_all)
    if T_new.shape[0] == 3 and T_new.shape[1] == 3:
        T_new = T2d_2_T3d(T_new)
    residual = wls_residual(c1_out, c2_out, weights_all, T_new)
    return T_new, residual, num_cones