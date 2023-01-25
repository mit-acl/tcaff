import numpy as np
from numpy.linalg import inv
from copy import deepcopy

from .observation_msg import ObservationMsg
from config import tracker_params as PARAM

class Tracker():

    def __init__(self, camera_id, tracker_id, estimated_state, feature_vec): #, Ts=1):
        self.A = PARAM.A
        # TODO: Something about my Ts seems to be off, changing it significantly changed results
        self.H = PARAM.H
        self.Q = PARAM.Q
        self.R = PARAM.R
        self.P = PARAM.P
        self.V = self.P[0:2,0:2] + self.R[0:2,0:2] # Pxy + Rxy

        self._id = (camera_id, tracker_id)
        self.mapped_ids = set([self._id])
        self._measurement = None
        self._state = np.concatenate((estimated_state, np.zeros((2,1))), 0)
        self._a = [feature_vec] if not isinstance(feature_vec, list) else feature_vec
        self.include_appearance = False
        # self.recent_detections = []
        self.rec_det_max_len = PARAM.n_recent_dets
        self.recent_detections = dict()

        self.frames_seen = 1
        self._ell = 0
        self._seen = False
        self.seen_by_this_camera = False
        # self.color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        self.to_str = ''
        if camera_id == 0:
            self.color = (np.random.randint(100,255), 0, 0)
        elif camera_id == 1:
            self.color = (0, np.random.randint(100,255), 0)
        elif camera_id == 2:
            self.color = (0, 0, np.random.randint(100,255))
        elif camera_id == 3:
            self.color = (np.random.randint(100,255), 0, np.random.randint(100,255))

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
    def measurement(self):
        return self._measurement
    
    @property
    def ell(self):
        return self._ell
    
    @property
    def feature_vecs(self):
        return self._a

    def update(self, measurement, feature_vec=None):
        self.frames_seen += 1
        self._seen = True
        self._ell = 0
        self._measurement = measurement
        if feature_vec is not None:
            self._a.append(feature_vec)
        # self._add_recent_detection(measurement[0:2,:].reshape(-1,1))

    def predict(self):
        xhat = self._state
        self.xbar = dict(); self.u = dict(); self.U = dict()
        self.xbar[self._id[0]] = self.A @ xhat
        if self._seen:
            z = self._measurement
            self.u[self._id[0]] = self.H.T @ inv(self.R) @ z
            self.U[self._id[0]] = self.H.T @ inv(self.R) @ self.H

    def observation_update(self, observation_msg):
        for mid in observation_msg.mapped_ids:
            self.mapped_ids.add(mid) 
        # TODO: don't think I'm handling this right
        if not observation_msg.has_measurement_info:
            return
        self.xbar[observation_msg.tracker_id[0]] = observation_msg.xbar
        self._ell = min(self._ell, observation_msg.ell)
        # if observation_msg.has_measurement_info:
        self.u[observation_msg.tracker_id[0]] = observation_msg.u
        self.U[observation_msg.tracker_id[0]] = observation_msg.U
        if observation_msg.has_appearance_info:
            for a in observation_msg.a:
                self._a.append(a)
                
    def add_appearance_gallery(self, appearance_vecs):
        for ai in appearance_vecs:
            for aj in self._a:
                if np.array_equal(ai, aj):
                    break
            self._a.append(ai)
    
    def observation_msg(self):
        '''
        Generates observation_msg containing tracker prediction and observation info.
        '''
        obs_msg = ObservationMsg(
                tracker_id=deepcopy(self._id), 
                mapped_ids=deepcopy(self.mapped_ids),
                xbar=np.copy(self.xbar[self._id[0]]),
                ell=deepcopy(self._ell),
            )
        if self._seen:
            u = np.copy(self.u[self._id[0]])
            U = np.copy(self.U[self._id[0]])
            obs_msg.add_measurement(u, U)
        if self.include_appearance:
            self.include_appearance = False
            obs_msg.add_appearance(deepcopy(self._a))
        return obs_msg

    def correction(self):
        if len(self.u) == 0:
            return
        y = np.zeros((6,1))
        S = np.zeros((6,6))
        for _, u in self.u.items():
            y += u
        for _, U in self.U.items():
            S += U

        M = inv(inv(self.P) + S)
        gamma = 1/np.linalg.norm(M+1)
        xbar = self.xbar[self._id[0]]
        xbar_diffs = np.zeros((6,1))
        for tracker_id, xbar_j in self.xbar.items():
            if tracker_id == self._id[0]: continue
            xbar_diffs += xbar_j - xbar 

        xhat = xbar + (M @ (y - S @ xbar)) \
            + (gamma * M  @ xbar_diffs)

        self._state = xhat
        self.P = self.A @ M @ self.A.T + self.Q
        self.V = self.P[0:2,0:2] + self.R[0:2,0:2]

    def cycle(self):
        if self._seen:
            self.to_str = f'{self._id}, state: {np.array2string(self._state.T,precision=2)},' + \
                f' measurement: {np.array2string(self._measurement.T,precision=2)}'
        else:
            self.to_str = f'{self._id}, state: {np.array2string(self._state.T,precision=2)},' + \
                f' not seen.'
            # self._add_recent_detection(None)
        for cam_id in self.u:
            if cam_id not in self.recent_detections:
                self.recent_detections[cam_id] = np.zeros((self.rec_det_max_len, 4)) * np.nan
        for cam_id in self.recent_detections:
            self.recent_detections[cam_id] = np.roll(self.recent_detections[cam_id], shift=1, axis=0)
            if cam_id not in self.u:
                self.recent_detections[cam_id][0, :] = np.zeros(4) * np.nan
            else:
                # TODO: Assuming H is identity matrix (or 1s along diag)
                z = PARAM.R @ self.u[cam_id][0:4,:]
                self.recent_detections[cam_id][0, 0:3] = np.concatenate([z[0:2, :].reshape(-1), [0]])
                self.recent_detections[cam_id][0, 3] = np.linalg.norm(self._state[0:2,:] - z[0:2,:])
        self._seen = False
        self._ell += 1
        
    def include_appearance_in_obs(self):
        self.include_appearance = True
        
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
        
    def _add_recent_detection(self, detection):
        self.recent_detections.insert(0, detection)
        self.recent_detections = \
            self.recent_detections[:min(len(self.recent_detections), self.rec_det_max_len)]

    def __str__(self):
        return self.to_str
