import numpy as np
from numpy.linalg import inv
from observation_msg import ObservationMsg

class Tracker():

    def __init__(self, camera_id, tracker_id, measurement, Ts=1):
        self.A = np.eye(6); self.A[0,4] = 1/Ts; self.A[1,5] = 1/Ts
        self.H = np.eye(4,6)
        self.Q = np.eye(6)
        self.R = np.eye(4)
        self.P = np.eye(6)
        self.V = self.P[0:2,0:2] + self.R[0:2,0:2] # Pxy + Rxy

        self._id = (camera_id, tracker_id)
        self._measurement = measurement
        self._state = np.concatenate((self.measurement, np.zeros((2,1))), 0)
        # self.initialized = False
        self.frames_seen = 1
        self.frames_since_being_seen = 0
        self._seen = True
        self.Ts = Ts
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
    def measurement(self):
        return self._measurement

    def update(self, measurement):
        self.frames_seen += 1
        self._seen = True
        self._measurement = measurement

    def predict(self):
        xhat = self._state
        z = self._measurement
        self.xbar = dict(); self.u = dict(); self.U = dict()
        self.xbar[self._id[0]] = self.A @ xhat
        self.u[self._id[0]] = self.H.T @ inv(self.R) @ z
        self.U[self._id[0]] = self.H.T @ inv(self.R) @ self.H

    def observation_update(self, observation_msg):
        self.xbar[observation_msg.tracker_id[0]] = observation_msg.xbar
        self.u[observation_msg.tracker_id[0]] = observation_msg.u
        self.U[observation_msg.tracker_id[0]] = observation_msg.U
    
    def observation_msg(self):
        return ObservationMsg(
            tracker_id = self._id, 
            xbar=self.xbar[self._id[0]],
            u=self.u[self._id[0]],
            U=self.U[self._id[0]],
            ell=0
        )

    def correction(self):
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

        self._seen = False

    def __str__(self):
        return f'{self._id}, state: {np.array2string(self._state.T,precision=2)},' + \
            f' measurement: {np.array2string(self._measurement.T,precision=2)}'
