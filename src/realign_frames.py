import open3d as o3d
import numpy as np
if o3d.__DEVICE_API__ == 'cuda':
    import open3d.cuda.pybind.t.pipelines.registration as treg
else:
    import open3d.cpu.pybind.t.pipelines.registration as treg

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
    max_correspondence_distance = 3.0

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