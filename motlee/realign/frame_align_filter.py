import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot

from motlee.utils.transform import transform, T_mag, transform_2_xypsi, xypsi_2_transform, \
    T2d_2_T3d
from motlee.realign.wls import wls, wls_residual

class FrameAlignFilter():
    
    def __init__(self, 
                 cam_id, 
                 connected_cams, 
                 ts=1,
                 filter_frame_align=True,
                 deg2m=8.1712,
    ):
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
        
        self.deg2m = deg2m
        self.tolerance_scale = 1
        ts = ts
        self.filter_frame_align = filter_frame_align
        if self.filter_frame_align:
            self.A = np.array([
                [1., 0., 0., ts, 0., 0.],
                [0., 1., 0., 0., ts, 0.],
                [0., 0., 1., 0., 0., ts],
                [0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 1.]
            ], dtype=np.float64)
            self.H = np.eye(3, 6)
            self.Q0 = 2*np.array([
                [(ts**4)/4, 0.,         0.,         (ts**3)/2,  0.,         0.],
                [0.,        (ts**4)/4,  0.,         0.,         (ts**3)/2,  0.],
                [0.,        0.,         (5*np.pi/180)**2*(ts**4)/4,  0.,         0.,         (5*np.pi/180)**2*(ts**3)/2],
                [(ts**3)/2, 0.,         0.,         ts**2,      0.,         0.],
                [0.,        (ts**3)/2,  0.,         0.,         ts**2,      0.],
                [0.,        0.,         (5*np.pi/180)**2*(ts**3)/2,  0.,      0.,            (5*np.pi/180)**2*ts**2],
            ])
            #0.5
            self.R = 0.1*np.array([
                [1**2, 0.0, 0.0],
                [0.0, 1**2, 0.0],
                [0.0, 0.0, (5*np.pi/180)**2]
            ])
            self.Q = np.copy(self.Q0)
            
        for cam in connected_cams:
            self.transforms[cam] = np.eye(4)
            self.transform_covs[cam] = np.copy(self.Q) if filter_frame_align else np.eye(6)
            self.transform_ders[cam] = np.zeros((3,1))
            self.T_last[cam] = np.eye(4)
            self.realigns_since_change[cam] = 0
            self.last_change_Tmag[cam] = 0
            self.realign_residual[cam] = 0
            self.realign_num_cones[cam] = 0
        
    def update_transform(self, cam_id, frame_align_sol):
        if frame_align_sol.success:
            self.realigns_since_change[cam_id] = 0
            self.last_change_Tmag[cam_id] = T_mag(frame_align_sol.transform @ np.linalg.inv(self.transforms[cam_id]), self.deg2m)
            self.T_last[cam_id] = np.copy(self.transforms[cam_id])
        else:
            self.realigns_since_change[cam_id] += 1

        if self.filter_frame_align:
            xk = np.vstack([np.array((transform_2_xypsi(self.transforms[cam_id]))).reshape((3,1)), self.transform_ders[cam_id]])
            xkp = self.A @ xk
            Qkp = self.A @ self.transform_covs[cam_id] @ self.A.T + self.Q0
            if frame_align_sol.success:
                yk = np.array((transform_2_xypsi(frame_align_sol.transform))).reshape((3,1))
                Lk = Qkp @ self.H.T @ inv(self.H @ Qkp @ self.H.T + self.R)
                xkp1 = xkp + Lk @ (yk - self.H @ xkp)
                Qkp1 = (np.eye(6) - Lk@self.H) @ Qkp
            else:
                xkp1 = xkp
                Qkp1 = Qkp
            Qkp1 = (Qkp1.T + Qkp1) / 2
            self.transform_covs[cam_id] = Qkp1
            self.transforms[cam_id] = xypsi_2_transform(*xkp1.reshape(-1)[:3].tolist())
            self.transform_ders[cam_id] = xkp1[3:,:]
        elif frame_align_sol.success:
            self.transforms[cam_id] = frame_align_sol.transform
            self.realign_residual[cam_id] = frame_align_sol.transform_residual
            self.realign_num_cones[cam_id] = frame_align_sol.num_objs_associated

    def get_transform_p1(self, cam_id):
        if self.filter_frame_align:
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
    
    def T_mag(self, T):
        R = Rot.from_matrix(T[0:3, 0:3])
        t = T[0:3, 3]
        rot_mag = R.as_euler('xyz', degrees=True)[2] / self.deg2m
        t_mag = np.linalg.norm(t)
        return np.abs(rot_mag) + np.abs(t_mag)