import numpy as np
from scipy.spatial.transform import Rotation as Rot

from config import tracker_params as TRACK_PARAM

def transform(T, vec):
    unshaped_vec = vec.reshape(-1)
    resized_vec = np.concatenate(
        [unshaped_vec, np.ones((T.shape[0] - unshaped_vec.shape[0]))]).reshape((-1, 1))
    transformed = T @ resized_vec
    return transformed.reshape(-1)[:unshaped_vec.shape[0]].reshape(vec.shape)   

class FrameRealigner():
    
    def __init__(self, cam_id, connected_cams, params):
        self.cam_id = cam_id
        self.transforms = dict()
        self.new_transforms = dict()
        for cam in connected_cams:
            self.transforms[cam] = np.eye(4)
        self.detections_min_num = params.detections_min_num
        self.tol_growth_rate = params.tolerance_growth_rate
        self.T_mag_unity_tol = params.transform_mag_unity_tolerance
        self.deg2m = params.DEG_2_M
        self.tolerance_scale = 1
        self.realign_algorithm = params.realign_algorithm
        self.ALGORITHMS = params.RealignAlgorithm
    
    def realign(self, trackers):
        local_dets = self._get_camera_detections(self.cam_id, trackers)
        T_mags = []
        T_news = dict()
        for cam_id in self.transforms:
            if cam_id == self.cam_id: continue
            other_dets = self._get_camera_detections(cam_id, trackers)
            dets1, dets2 = self.detection_pairs_2_ordered_arrays(local_dets, other_dets)
            T_new = self.calc_T(dets1, dets2)
            
            self.new_transforms[cam_id] = T_new
            self.transforms[cam_id] = T_new @ self.transforms[cam_id]
            T_mags.append(self.T_mag(T_new))
        # TODO: Magic number here, we should have a better way to recognize a need for frame realignment
        # Intentionally left as 5 because I don't want to have this clause here, shouldn't arbitrarily trigger
        # realignment
        if len(trackers) > 5 and all(i == 0.0 for i in T_mags):
            self.tolerance_scale *= self.tol_growth_rate # should this be here??
            pass
        else:
            T_mags.append(.01) # prevent divide by zero warning
            scaling = self.tol_growth_rate ** (np.log2(max(T_mags) / self.T_mag_unity_tol))
            self.tolerance_scale = max(1, self.tolerance_scale * scaling) # does this cause crazy tau growth sometimes??
            # self.tolerance_scale = max(1, scaling)
        # print(T_mags + [self.tolerance_scale])
        
    def rectify_detections(self, trackers):
        for tracker in trackers:
            for cam_num in tracker.recent_detections:
                if cam_num not in self.new_transforms: continue
                dets = tracker.recent_detections[cam_num]
                T = self.transforms[cam_num]
                for i in range(dets.shape[0]):
                    dets[i, 0:3] = transform(T, dets[i, 0:3])
                    # dets[i, 3] = np.linalg.norm(tracker.historical_states[0:2, i] - dets[i, 0:2])
        self.new_transforms = dict()
                
    
    def _get_camera_detections(self, cam_num, trackers):
        dets = []
        for tracker in trackers:
            if cam_num in tracker.recent_detections:
                dets.append(tracker.recent_detections[cam_num])
            else:
                dets.append(np.zeros((TRACK_PARAM.n_recent_dets, 3)) * np.nan)
        return dets
    
    def detection_pairs_2_ordered_arrays(self, detection_list1, detection_list2):
        ordered_pairs = [np.array([]).reshape(0,4), np.array([]).reshape(0,4)]
        for track1, track2 in zip(detection_list1, detection_list2):
            if track1.shape != track2.shape:
                continue
            for i in range(track1.shape[0]):
                if np.isnan(track1[i,:]).any() or np.isnan(track2[i,:]).any():
                    continue
                else:
                    ordered_pairs[0] = np.concatenate([ordered_pairs[0], track1[i,:].reshape(-1,4)], axis=0)
                    ordered_pairs[1] = np.concatenate([ordered_pairs[1], track2[i,:].reshape(-1,4)], axis=0)
        return ordered_pairs
    
    def T_mag(self, T):
        R = Rot.from_matrix(T[0:3, 0:3])
        t = T[0:3, 3]
        rot_mag = R.as_euler('xyz', degrees=True)[2] / self.deg2m
        t_mag = np.linalg.norm(t)
        return np.abs(rot_mag) + np.abs(t_mag)
    
    def calc_T(self, det1, det2, add_weighting=True):    
        if det1.shape[0] < self.detections_min_num or \
            det2.shape[0] < self.detections_min_num:
            return np.eye(4)
        if False:
            mean_pts = (det1 + det2) / 2.0
            det_dist_from_mean = np.linalg.norm(det1 - mean_pts, axis=1)
            weight_vals = 1 / (.01 + det_dist_from_mean)
            weight_vals = weight_vals.reshape((-1,1))
            mean1 = (np.sum(det1 * weight_vals, axis=0) / np.sum(weight_vals)).reshape(-1)
            mean2 = (np.sum(det2 * weight_vals, axis=0) / np.sum(weight_vals)).reshape(-1)
            det1_mean_reduced = det1 - mean1
            det2_mean_reduced = det2 - mean2
            assert det1_mean_reduced.shape == det2_mean_reduced.shape
            H = det1_mean_reduced.T @ (det2_mean_reduced * weight_vals)
            U, s, V = np.linalg.svd(H)
            R = U @ V.T
            t = mean1.reshape((3,1)) - R @ mean2.reshape((3,1))
            T = np.concatenate([np.concatenate([R, t], axis=1), np.array([[0,0,0,1]])], axis=0)
        if add_weighting:
            weight_vals = 1 / (1e-6 + det1[:,3] * det2[:,3])
            weight_vals = weight_vals.reshape((-1,1))
            det1 = det1[:, 0:3]
            det2 = det2[:, 0:3]
            mean1 = (np.sum(det1 * weight_vals, axis=0) / np.sum(weight_vals)).reshape(-1)
            mean2 = (np.sum(det2 * weight_vals, axis=0) / np.sum(weight_vals)).reshape(-1)
            det1_mean_reduced = det1 - mean1
            det2_mean_reduced = det2 - mean2
            assert det1_mean_reduced.shape == det2_mean_reduced.shape
            H = det1_mean_reduced.T @ (det2_mean_reduced * weight_vals)
            U, s, V = np.linalg.svd(H)
            R = U @ V.T
            t = mean1.reshape((3,1)) - R @ mean2.reshape((3,1))
            T = np.concatenate([np.concatenate([R, t], axis=1), np.array([[0,0,0,1]])], axis=0)
        else:
            mean1 = np.mean(det1, axis=0).reshape(-1)
            mean2 = np.mean(det2, axis=0).reshape(-1)
            det1_mean_reduced = det1 - mean1
            det2_mean_reduced = det2 - mean2
            assert det1_mean_reduced.shape == det2_mean_reduced.shape
            H = det1_mean_reduced.T @ det2_mean_reduced
            U, s, V = np.linalg.svd(H)
            R = U @ V.T
            t = mean1.reshape((3,1)) - R @ mean2.reshape((3,1))
            T = np.concatenate([np.concatenate([R, t], axis=1), np.array([[0,0,0,1]])], axis=0)
        if np.isnan(T).any():
            T = np.eye(4)
        return T
    
    