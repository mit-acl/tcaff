import numpy as np
from numpy.linalg import inv
from observation_msg import ObservationMsg
from copy import deepcopy
class Tracker():

    def __init__(self, camera_id, tracker_id, estimated_state, feature_vec, Ts=1):
        self.A = np.eye(6); self.A[0,4] = 1/Ts; self.A[1,5] = 1/Ts
        self.H = np.eye(4,6)
        # self.Q = np.eye(6)
        # self.Q = np.array([[
        #     [Ts**4/4, Ts**3/2],
        #     [Ts**3/2, Ts**2]
        # ]])
        self.Q = np.array([
            [Ts**4/4, 0, 0, 0, Ts**3/2, 0],
            [0, Ts**4/4, 0, 0, 0, Ts**3/2],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0],
            [Ts**3/2,0,0,0,Ts**2,0],
            [0,Ts**3/2,0,0,0,Ts**2]
        ])
        self.R = np.eye(4)*3
        self.P = np.eye(6)
        self.V = self.P[0:2,0:2] + self.R[0:2,0:2] # Pxy + Rxy

        self._id = (camera_id, tracker_id)
        self.mapped_ids = set([self._id])
        self._measurement = None
        self._state = np.concatenate((estimated_state, np.zeros((2,1))), 0)
        self._a = [feature_vec] if not isinstance(feature_vec, list) else feature_vec
        self.include_appearance = False

        self.frames_seen = 1
        self._ell = 0
        self._seen = False
        self.seen_by_this_camera = False
        self.Ts = Ts
        self.color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        self.to_str = ''

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
        if not observation_msg.has_measurement_info:
            return
        self.xbar[observation_msg.tracker_id[0]] = observation_msg.xbar
        self._ell = min(self._ell, observation_msg.ell)
        # if observation_msg.has_measurement_info:
        #     self.u[observation_msg.tracker_id[0]] = observation_msg.u
        #     self.U[observation_msg.tracker_id[0]] = observation_msg.U
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
        self._seen = False
        self._ell += 1
        
    def include_appearance_in_obs(self):
        self.include_appearance = True

    def __str__(self):
        return self.to_str
