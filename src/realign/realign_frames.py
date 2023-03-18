import open3d as o3d
import numpy as np
from numpy.linalg import inv, norm
from scipy.optimize import linear_sum_assignment
if o3d.__DEVICE_API__ == 'cuda':
    import open3d.cuda.pybind.t.pipelines.registration as treg
else:
    import open3d.cpu.pybind.t.pipelines.registration as treg

from realign.wls import wls
from utils.transform import transform

import clipperpy

# STARTING A LIST OF MAGIC NUMBERS HERE
NUM_CONES_REQ = 10
USE_CLIPPER = False
ONLY_STRONG_CORRESPONDENCES = True
ICP_MAX_DIST = 1.0

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

def realign_frames(detections1, detections2, initial_guess=np.eye(4), max_dist=.5):
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

def realign_frames_clipper(detections1, detections2, sigma=0.30, epsilon=0.90):
    """
    Parameters
    ----------
    detections1 : (n,3) np.array
    detections2 : (m,3) np.array

    Return
    ------
    corres_set : (m,) np.array -- if no corres, entry is -1 else corresponding index of det1
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

    corres_set = np.zeros((m,)).astype(int)
    for i in range(m):
        if i in Ain[:,1]:
            idx, = np.where(Ain[:,1] == i)
            corres_set[i] = Ain[idx,0]
        else:
            corres_set[i] = -1

    return corres_set

def realign_cones(cones1, cones2, T_current):
    if True:
        return recently_weighted_realign(cones1, cones2, T_current)
    cone1_states = []
    cone2_states = []
    for cone in cones1:
        cone1_states.append(cone.state[:2, :].reshape(-1).tolist() + [0])
    for cone in cones2:
        cone2_states.append(cone.state[:2, :].reshape(-1).tolist() + [0])
    if len(cone1_states) < NUM_CONES_REQ or len(cone2_states) < NUM_CONES_REQ: 
        return T_current
    cone1_ptcld = o3d.t.geometry.PointCloud(np.array(cone1_states))
    cone2_ptcld = o3d.t.geometry.PointCloud(np.array(cone2_states))
    return realign_frames(cone1_ptcld, cone2_ptcld, T_current, max_dist=1.0).transformation.numpy()

def get_cones_and_weights(cones1_input, cones2_input, T_current, ages1=None, ages2=None):
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
    if USE_CLIPPER:
        correspondence_set2 = realign_frames_clipper(cones1, cones2)
    else:
        cone1_ptcld = o3d.t.geometry.PointCloud(cones1)
        cone2_ptcld = o3d.t.geometry.PointCloud(cones2)
        correspondence_set2 = realign_frames(cone1_ptcld, cone2_ptcld, T_current, max_dist=ICP_MAX_DIST).correspondences_.numpy()
    
    
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
    
    if len(cones2) < NUM_CONES_REQ:
        return None, None, T_current
        
    weights = 1/(.01 + ages2 * ages1_reordered)
    
    return cones1_reordered, cones2, weights

def recently_weighted_realign(cones1_input, cones2_input, T_current):
    cones1_reordered, cones2, weights1 = get_cones_and_weights(cones1_input, cones2_input, T_current)
    cones2_reordered, cones1, weights2 = get_cones_and_weights(cones2_input, cones1_input, inv(T_current))
    if cones1_reordered is None or cones2_reordered is None or cones1 is None or cones2 is None:
        return T_current
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

    if len(c1_out)/2 < NUM_CONES_REQ:
        return T_current

    return wls(c1_out, c2_out, weights_all)
    
    
    
    
    
    
    
    
    # correspondence set in terms of 2, want in terms of 1
    correspondence_set = np.zeros((cones1.shape[0], 1), dtype=np.int32)
    for i in range(cones1.shape[0]):
        where = np.argwhere(correspondence_set2 == i)
        if where.size == 0:
            correspondence_set[i, 0] = int(-1)
        elif where.size == 2:
            correspondence_set[i, 0] = int(where[0,0])
        else:
            correspondence_set[i, 0] = int(where[0,0])
            # print(where)
            # assert False
            
    cones2_reordered = np.zeros(cones1.shape)
    ages2_reordered = np.zeros((cones1.shape[0], 1))
    for i in range(cones1.shape[0]):
        if correspondence_set[i] == -1: continue
        try:
            cones2_reordered[i, :] = cones2[correspondence_set[i], :]
        except:
            import ipdb; ipdb.set_trace()
        ages2_reordered[i, 0] = ages2[correspondence_set[i], 0]
    no_correspond_idx = [i for i,x in enumerate(correspondence_set.reshape(-1).tolist()) if x==-1]
    cones1 = np.delete(cones1, no_correspond_idx, axis=0)
    cones2_reordered = np.delete(cones2_reordered, no_correspond_idx, axis=0)
    ages1 = np.delete(ages1, no_correspond_idx, axis=0)
    ages2_reordered = np.delete(ages2_reordered, no_correspond_idx, axis=0)
    
    if len(cones1) < NUM_CONES_REQ:
        return T_current
        
    weights = 1/(.01 + ages1 * ages2_reordered)
    return wls(cones1, cones2_reordered, weights)

def my_realign_cones(cones1_in, cones2_in, ages1, ages2, T_current):
    if cones1_in.shape[0] < NUM_CONES_REQ or cones2_in.shape[0] < NUM_CONES_REQ:
        return T_current, 0, 0
    try:
        cones1_reordered, cones2, weights1 = get_cones_and_weights(np.copy(cones1_in), np.copy(cones2_in), T_current, np.copy(ages1), np.copy(ages2))
    except:
        import ipdb; ipdb.set_trace()
    try:
        cones2_reordered, cones1, weights2 = get_cones_and_weights(np.copy(cones2_in), np.copy(cones1_in), inv(T_current), np.copy(ages2), np.copy(ages1))
    except:
        import ipdb; ipdb.set_trace()
    if cones1_reordered is None or cones2_reordered is None or cones1 is None or cones2 is None:
        return T_current, 0, 0
    c1_out = np.concatenate([cones1_reordered, cones1], axis=0)
    c2_out = np.concatenate([cones2, cones2_reordered], axis=0)
    weights_all = np.concatenate([weights1, weights2], axis=0)

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

    if len(c1_out)/2 < NUM_CONES_REQ:
        return T_current, 0, 0

    T_new = wls(c1_out, c2_out, weights_all)
    
    c2_transformed = np.empty(c2_out.shape)
    for i in range(c2_out.shape[0]):
        c2_transformed[i, :] = transform(T_new, c2_out[i, :])
    
    least_sq_sum = 0
    for i in range(c2_out.shape[0]):
        least_sq_sum += weights_all.item(i) * np.linalg.norm(c1_out[i, :] - c2_transformed[i, :])**2        
        
    return T_new, least_sq_sum / 2, c2_out.shape[0] / 2

    cone1_states = cones1
    cone2_states = cones2
    # TODO: change how I get ell
    # cone1_states = []
    # cone2_states = []
    # for cone in cones1:
    #     cone1_states.append(cone.state[:2, :].reshape(-1).tolist() + [0])
    # for cone in cones2:
    #     cone2_states.append(cone.state[:2, :].reshape(-1).tolist() + [0])
    # if len(cone1_states) < NUM_CONES_REQ or len(cone2_states) < NUM_CONES_REQ: 
    #     return T_current
    cone1_ptcld = o3d.t.geometry.PointCloud(np.array(cone1_states))
    cone2_ptcld = o3d.t.geometry.PointCloud(np.array(cone2_states))
    correspondence_set2 = realign_frames(cone1_ptcld, cone2_ptcld, T_current, max_dist=1.0)
    # correspondence set in terms of 2, want in terms of 1
    correspondence_set = np.zeros((cones1.shape[0], 1), dtype=np.int32)
    for i in range(cones1.shape[0]):
        where = np.argwhere(correspondence_set2 == i)
        if where.size == 0:
            correspondence_set[i, 0] = int(-1)
        elif where.size == 2:
            correspondence_set[i, 0] = int(where[0,0])
        else:
            correspondence_set[i, 0] = int(where[0,0])
            # print(where)
            # assert False
            
    cones2_reordered = np.zeros(cones1.shape)
    ages2_reordered = np.zeros((cones1.shape[0], 1))
    for i in range(cones1.shape[0]):
        if correspondence_set[i] == -1: continue
        try:
            cones2_reordered[i, :] = cones2[correspondence_set[i], :]
        except:
            import ipdb; ipdb.set_trace()
        ages2_reordered[i, 0] = ages2[correspondence_set[i], 0]
    no_correspond_idx = [i for i,x in enumerate(correspondence_set.reshape(-1).tolist()) if x==-1]
    cones1 = np.delete(cones1, no_correspond_idx, axis=0)
    cones2_reordered = np.delete(cones2_reordered, no_correspond_idx, axis=0)
    ages1 = np.delete(ages1, no_correspond_idx, axis=0)
    ages2_reordered = np.delete(ages2_reordered, no_correspond_idx, axis=0)
        
    weights = 1/(1 + ages1 * ages2_reordered)
    return wls(cones1, cones2_reordered, weights)