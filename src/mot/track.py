import numpy as np
from numpy.linalg import inv
from copy import deepcopy

from .measurement_info import MeasurementInfo

class Track():

    def __init__(self, camera_id, track_id, track_params, init_measurements, init_state):
        self.A = np.copy(track_params.A)
        self.H = np.copy(track_params.H)
        self.Q = np.copy(track_params.Q)
        self.R = np.copy(track_params.R)
        self.P = np.copy(track_params.P)
        self.V = self.P[0:2,0:2] + self.R[0:2,0:2] # Pxy + Rxy
        self.shape_x = self.A.shape[0]
        self.shape_z = self.H.shape[0]

        self._id = (camera_id, track_id)
        self.mapped_ids = set([self._id])
        self._measurements = None
        self._state = init_state

        self.rec_det_max_len = track_params.n_dets
        self.recent_detections = dict()
        self.historical_states = np.zeros((self.shape_x, self.rec_det_max_len)) * np.nan
        self.historical_states[:,0] = self._state.reshape((-1))
        self.lifespan = 0    
        self.dead_cnt = -1    

        self.frames_seen = 1
        self._ell = 0
        self._seen = False
        self.color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))

        self.predict()

    @property
    def state(self):
        return self._state

    @property
    def id(self):
        return self._id

    @property
    def seen(self):
        return self._seen

    @property
    def measurements(self):
        return self._measurements
    
    @property
    def ell(self):
        return self._ell
    
    @property
    def xbar(self):
        return self._xbar[self._id[0]]

    def update(self, measurements):
        self.frames_seen += 1
        self._seen = True
        self._ell = 0
        self._measurements = measurements

    def predict(self):
        '''
        Sets xbar, z, R, and H
        '''
        xhat = self._state
        self._xbar = dict(); self.zs = dict(); self.us = dict(); self.Us = dict()
        self._xbar[self._id[0]] = self.A @ xhat
        if self._seen:
            self.zs[self._id[0]] = list()
            u = np.zeros((self.shape_x, 1))
            U = np.zeros((self.shape_x, self.shape_x))
            for z in self._measurements:
                # import ipdb; ipdb.set_trace()
                u += self.H.T @ inv(self.R) @ z
                U += self.H.T @ inv(self.R) @ self.H
                self.zs[self._id[0]].append(np.copy(z))
            self.us[self._id[0]] = u
            self.Us[self._id[0]] = U

    def add_measurement(self, meas_info):
        for mid in meas_info.mapped_ids:
            self.mapped_ids.add(mid) 
        if not meas_info.has_measurement_info:
            return
        self._xbar[meas_info.track_id[0]] = meas_info.xbar
        self._ell = min(self._ell, meas_info.ell)
        self.zs[meas_info.track_id[0]] = meas_info.zs
        self.us[meas_info.track_id[0]] = meas_info.u
        self.Us[meas_info.track_id[0]] = meas_info.U

    def get_measurement_info(self, R, T=np.eye(4)):
        '''
        Generates observation_msg containing tracker prediction and observation info.
        '''
        # TODO: magic numbers here (would be nice to have x and z dim not matter)
            
        # Extract xbar and correct
        xbar=np.copy(self._xbar[self._id[0]])
        pos = np.concatenate([xbar[0:2,:], [[0], [1]]], axis=0)
        vel = np.concatenate([xbar[2:4,:], [[0]]], axis=0)
        pos_corrected = T @ pos
        vel_corrected = T[0:3, 0:3] @ vel
        xbar[0:2,:] = pos_corrected[0:2,:]
        xbar[2:4,:] = vel_corrected[0:2,:]

        meas_info = MeasurementInfo(
                track_id=deepcopy(self._id), 
                mapped_ids=deepcopy(self.mapped_ids),
                xbar=xbar,
                ell=deepcopy(self._ell),
            )
        if self._seen:
            transformed_zs = []
            for z in self.zs[self._id[0]]:
                p_meas = np.concatenate([z[0:self.shape_z,:], [[0.], [1.]]], axis=0)
                p_meas_corrected = (T @ p_meas)[0:self.shape_z,:]
                transformed_zs.append(np.concatenate([p_meas_corrected, z[self.shape_z:]], axis=0))
            u = np.zeros(self.shape_z)
            U = np.zeros((self.shape_x, self.shape_x))
            for z in transformed_zs:
                u = self.H.T @ inv(R) @ z
                U = self.H.T @ inv(R) @ self.H
            meas_info.add_measurements(transformed_zs, u, U)
        return meas_info

    def correction(self):
        if len(self.zs) == 0:
            return
        y = np.zeros((self.shape_x, 1))
        Y = np.zeros((self.shape_x, self.shape_x))
        for cam in self.zs:
            u = self.us[cam]
            U = self.Us[cam]
            y += u
            Y += U

        M = inv(inv(self.P) + Y)
        gamma = 1/np.linalg.norm(M+1)
        xbar = self._xbar[self._id[0]]
        xbar_diffs = np.zeros((self.shape_x, 1))
        for track_id, xbar_j in self._xbar.items():
            if track_id == self._id[0]: continue
            xbar_diffs += xbar_j - xbar 

        xhat = xbar + (M @ (y - Y @ xbar)) \
            + (gamma * M  @ xbar_diffs)

        self._state = xhat
        self.P = self.A @ M @ self.A.T + self.Q
        self.V = self.P[0:2,0:2] + self.R[0:2,0:2]

    def cycle(self):
        
        self.historical_states = np.roll(self.historical_states, shift=1, axis=1)
        self.historical_states[:,0] = self._state.reshape((-1))

        self._seen = False
        self._ell += 1
        
        self.lifespan = min(self.lifespan + 1, self.rec_det_max_len - 1)
        self.dead_cnt += 1 if self.dead_cnt >= 0 else 0
        
        for cam_id in self.zs:
            if cam_id not in self.recent_detections:
                self.recent_detections[cam_id] = np.zeros((self.rec_det_max_len, 4)) * np.nan
        for cam_id in self.recent_detections:
            self.recent_detections[cam_id] = np.roll(self.recent_detections[cam_id], shift=1, axis=0)
            if cam_id not in self.zs:
                self.recent_detections[cam_id][0, :] = np.zeros(4) * np.nan
            else:
                z = self.zs[cam_id][0]
                # TODO: only grabbing first z right here
                self.recent_detections[cam_id][0, 0:3] = np.concatenate([z[0:2, :].reshape(-1), [0]])
                self.recent_detections[cam_id][0, 3] = np.linalg.norm(self._state[0:2,:] - z[0:2,:])
        
    def died(self):
        self.dead_cnt = 0
        self._xbar = dict(); self.zs = dict(); self.Rs = dict(); self.Hs = dict()
        
    def get_recent_detection_array(self):
        ids = []
        for camera_id in self.recent_detections:
            ids.append(camera_id)
        detection_array = None
        for i in sorted(ids):
            if detection_array is None:
                detection_array = np.copy(self.recent_detections[i])
            else:
                detection_array = np.concatenate([detection_array, self.recent_detections[i]], axis=1)
        return detection_array
    
    def __str__(self):
        return f'{self._id}: {self._state.reshape(-1)[:2]}'
