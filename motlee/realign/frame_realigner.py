import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot

from motlee.config.track_params import TrackParams
from motlee.mot.measurement_info import MeasurementInfo
from motlee.utils.transform import transform, T_mag, transform_2_xypsi, xypsi_2_transform
from motlee.realign.wls import wls


RECURSIVE_LEAST_SQUARES = False
KF = False

class FrameRealigner():
    
    def __init__(self, cam_id, connected_cams, params):
        self.cam_id = cam_id
        self.transforms = dict()
        self.transform_covs = dict()
        self.transform_ders = dict()
        self.new_transforms = dict()
        self.realigns_since_change = dict()
        self.last_change_Tmag = dict()
        self.T_last = dict()
        self.realign_residual = dict()
        self.realign_num_cones = dict()
        for cam in connected_cams:
            self.transforms[cam] = np.eye(4)
            self.transform_covs[cam] = np.eye(3) if RECURSIVE_LEAST_SQUARES else np.eye(6)
            self.transform_ders[cam] = np.zeros((3,1))
            self.T_last[cam] = np.eye(4)
            self.realigns_since_change[cam] = 0
            self.last_change_Tmag[cam] = 0
            self.realign_residual[cam] = 0
            self.realign_num_cones[cam] = 0
        self.detections_min_num = params.detections_min_num
        self.tol_growth_rate = params.tolerance_growth_rate
        self.T_mag_unity_tol = params.transform_mag_unity_tolerance
        self.deg2m = params.DEG_2_M
        self.tolerance_scale = 1
        self.realign_algorithm = params.realign_algorithm
        self.ALGORITHMS = params.RealignAlgorithm
        if RECURSIVE_LEAST_SQUARES:
            self.Q0 = np.array([
                [1**2, 0.0, 0.0],
                [0.0, 1**2, 0.0],
                [0.0, 0.0, (3*np.pi/180)**2]
            ])
            self.R = 0.1*np.array([
                [1**2, 0.0, 0.0],
                [0.0, 1**2, 0.0],
                [0.0, 0.0, (3*np.pi/180)**2]
            ])
        elif KF:
            ts = .1
            self.A = np.array([
                [1., 0., 0., ts, 0., 0.],
                [0., 1., 0., 0., ts, 0.],
                [0., 0., 1., 0., 0., ts],
                [0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 1.]
            ], dtype=np.float64)
            self.H = np.eye(3, 6)
            self.Q0 = 10000*np.array([
                [(ts**4)/4, 0.,         0.,         (ts**3)/2,  0.,         0.],
                [0.,        (ts**4)/4,  0.,         0.,         (ts**3)/2,  0.],
                [0.,        0.,         (5*np.pi/180)**2*(ts**4)/4,  0.,         0.,         (5*np.pi/180)**2*(ts**3)/2],
                [(ts**3)/2, 0.,         0.,         ts**2,      0.,         0.],
                [0.,        (ts**3)/2,  0.,         0.,         ts**2,      0.],
                [0.,        0.,         (5*np.pi/180)**2*(ts**3)/2,  0.,      0.,            (5*np.pi/180)**2*ts**2],
            ])
            self.R = 0.01*np.array([
                [1**2, 0.0, 0.0],
                [0.0, 1**2, 0.0],
                [0.0, 0.0, (5*np.pi/180)**2]
            ])
            self.Q = np.copy(self.Q0)
        
    
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
        if self.realign_algorithm == self.ALGORITHMS.REALIGN_WLS:
            pass
        elif len(trackers) > 5 and all(i == 0.0 for i in T_mags):
            self.tolerance_scale *= self.tol_growth_rate # should this be here??
            pass
        else:
            # TODO: Figure out how exactly I am going to scale T
            # T_mags.append(.01) # prevent divide by zero warning
            # scaling = self.tol_growth_rate ** (np.log2(max(T_mags) / self.T_mag_unity_tol))
            scaling = self.tol_growth_rate * np.mean(T_mags) / self.T_mag_unity_tol
            # self.tolerance_scale = max(1, self.tolerance_scale * scaling) # does this cause crazy tau growth sometimes??
            self.tolerance_scale = max(1, scaling)
        print(T_mags + [self.tolerance_scale])
        
    def rectify_detections(self, trackers):
        for tracker in trackers:
            for cam_num in tracker.recent_detections:
                if cam_num not in self.new_transforms: continue
                dets = tracker.recent_detections[cam_num]
                T = self.transforms[cam_num]
                for i in range(dets.shape[0]):
                    dets[i, 0:3] = transform(T, dets[i, 0:3])
                    # dets[i, 3] = np.linalg.norm(tracker.historical_states[0:2, i] - dets[i, 0:2])
            
            ## Fixing state
            tracker._state = tracker.historical_states[:, tracker.lifespan].reshape((6,1))
            tracker.P = tracker.historical_covariance[:, tracker.lifespan*6:(tracker.lifespan+1)*6]
            for i in range(tracker.lifespan, tracker.dead_cnt, -1):
                tracker.predict()
                for cam_num in tracker.recent_detections:
                    det = tracker.recent_detections[cam_num][i, 0:3]
                    obs_msg = ObservationMsg(
                        tracker_id=(cam_num, -1), mapped_ids=[], 
                        xbar=tracker.xbar[tracker.id[0]], ell=tracker.ell
                    )
                    if not any(np.isnan(det)):
                        u = tracker.H.T @ inv(tracker.R) @ np.concatenate([det[0:2].reshape((2,1)), tracker.state[2:4,:].reshape((2,1))], axis=0)
                        U = tracker.H.T @ inv(tracker.R) @ tracker.H # TODO: right?
                        obs_msg.add_measurement(u, U)
                    tracker.observation_update(obs_msg)
                tracker.correction()
                # for cam_num in tracker.recent_detections:
                #     det = tracker.recent_detections[cam_num][i, 0:3]
                #     if not any(np.isnan(det)):
                #         tracker.recent_detections[cam_num][i, 3] = \
                #             np.linalg.norm(tracker.state[:, 0:2] - det[0:2])
                tracker.cycle(historical_only=True)
            for i in range(tracker.dead_cnt + 1):
                tracker.cycle(historical_only=True)
                
        self.new_transforms = dict()
        
    def update_transform(self, cam_id, transform, alignment_residual=None, alignment_num_cones=None):
        if not np.allclose(self.transforms[cam_id], transform):
            self.realigns_since_change[cam_id] = 0
            self.last_change_Tmag[cam_id] = T_mag(transform @ np.linalg.inv(self.transforms[cam_id]), self.deg2m)
            self.T_last[cam_id] = np.copy(self.transforms[cam_id])
        else:
            self.realigns_since_change[cam_id] += 1
        
        if RECURSIVE_LEAST_SQUARES:
            y_meas = np.array((transform_2_xypsi(transform))).reshape((3,1))
            xhatk = np.array((transform_2_xypsi(self.transforms[cam_id]))).reshape((3,1))

            Qkp1 = inv(inv(self.Q0) + inv(self.R))
            # Qkp1 = inv(inv(self.transform_covs[cam_id]) + inv(self.Q0))
            xhatkp1 = xhatk + Qkp1@inv(self.R)@(y_meas - xhatk)
            self.transform_covs[cam_id] = Qkp1
            self.transforms[cam_id] = xypsi_2_transform(*xhatkp1.reshape(-1).tolist())
        if KF:
            xk = np.vstack([np.array((transform_2_xypsi(self.transforms[cam_id]))).reshape((3,1)), self.transform_ders[cam_id]])
            xkp = self.A @ xk
            Qkp = self.A @ self.transform_covs[cam_id] @ self.A.T + self.Q0
            yk = np.array((transform_2_xypsi(transform))).reshape((3,1))
            Lk = self.Q @ self.H.T @ inv(self.H @ self.Q @ self.H.T + self.R)
            xkp1 = xkp + Lk @ (yk - self.H @ xkp)
            Qkp1 = (np.eye(6) - Lk@self.H) @ Qkp
            Qkp1 = (Qkp1.T + Qkp1) / 2
            self.transform_covs[cam_id] = Qkp1
            self.transforms[cam_id] = xypsi_2_transform(*xkp1.reshape(-1)[:3].tolist())
            self.transform_ders[cam_id] = xkp1[3:,:]
        else:
            self.transforms[cam_id] = transform
            self.realign_residual[cam_id] = alignment_residual
            self.realign_num_cones[cam_id] = alignment_num_cones

    def get_transform_p1(self, cam_id):
        if KF:
            xk = np.vstack([np.array((transform_2_xypsi(self.transforms[cam_id]))).reshape((3,1)), self.transform_ders[cam_id]])
            xkp = self.A @ xk
            return xypsi_2_transform(*xkp.reshape(-1)[:3].tolist())
        else:
            return self.transforms[cam_id]
        
    def transform_obs(self, obs):
        obs_cam = obs.tracker_id[0]
        if obs_cam not in self.transforms:
            return obs
        if obs_cam == self.cam_id:
            return obs
        
        # Extract z and correct
        # TODO: Assuming H is identity matrix (or 1s along diag)
        if obs.z is not None:
            z = obs.z
            p_meas = np.concatenate([z[0:2,:], [[0.], [1.]]], axis=0)
            p_meas_corrected = (self.transforms[obs_cam] @ p_meas)[0:2,:]
            obs.z = np.concatenate([p_meas_corrected, z[2:]], axis=0)
            
        # Extract xbar and correct
        pos = np.concatenate([obs.xbar[0:2,:], [[0], [1]]], axis=0)
        vel = np.concatenate([obs.xbar[4:6,:], [[0]]], axis=0)
        pos_corrected = self.transforms[obs_cam] @ pos
        vel_corrected = self.transforms[obs_cam][0:3, 0:3] @ vel
        obs.xbar[0:2,:] = pos_corrected[0:2,:]
        obs.xbar[4:6,:] = vel_corrected[0:2,:]
        return obs
                
    
    def _get_camera_detections(self, cam_num, trackers):
        dets = []
        for tracker in trackers:
            if cam_num in tracker.recent_detections:
                dets.append(tracker.recent_detections[cam_num])
            else:
                dets.append(np.zeros((TrackParams().n_recent_dets, 3)) * np.nan)
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
            T = wls(det1, det2, weight_vals)
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
    
    