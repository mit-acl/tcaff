import open3d as o3d
import numpy as np
if o3d.__DEVICE_API__ == 'cuda':
    import open3d.cuda.pybind.t.pipelines.registration as treg
else:
    import open3d.cpu.pybind.t.pipelines.registration as treg

# STARTING A LIST OF MAGIC NUMBERS HERE
NUM_CONES_REQ = 7

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

def realign_frames(detections1, detections2, initial_guess=np.eye(4)):
    trans_init = initial_guess
    # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
    estimation = treg.TransformationEstimationPointToPoint()
    # Search distance for Nearest Neighbour Search [Hybrid-Search is used].
    max_correspondence_distance = .5

    # Initial alignment or source to target transform.
    init_source_to_target = trans_init

    # Convergence-Criteria for Vanilla ICP
    criteria = treg.ICPConvergenceCriteria(relative_fitness=0.0000001,
                                        relative_rmse=0.0000001,
                                        max_iteration=30)

    # Down-sampling voxel-size. If voxel_size < 0, original scale is used.
    voxel_size = -1

    time_trials=100
    for i in range(time_trials):
        reg_point_to_point = treg.icp(detections2, detections1, max_correspondence_distance,
                                    init_source_to_target, estimation, criteria,
                                    voxel_size)
    return reg_point_to_point.transformation.numpy()

def realign_cones(cones1, cones2, T_current):
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
    return realign_frames(cone1_ptcld, cone2_ptcld, T_current)