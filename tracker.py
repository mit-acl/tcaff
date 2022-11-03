import numpy as np
from numpy.linalg import inv
from observation_msg import ObservationMsg

# class TrackerCandidate():

#     def __init__(self, new_id, measurement):


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

    @property
    def state(self):
        return self._state
        # if self._state:
        #     return self._state
        # else:
        #     return np.concatenate((self.measurement, np.zeros((2,1))), 0)

    @property
    def id(self):
        return self._id

    @property
    def seen(self):
        return self._seen

    @property
    def measurement(self):
        return self._measurement

    # def measurement_update(self, measurement):
    #     # if not self.initialized:
    #     #     self.state = np.concatenate((pos, box_size, vel), axis=0)
    #     #     self.predict()
    #     #     self.correction()
    #     self.measurements.append(measurement)
    #     self._seen = True

    def update(self, measurement):
        self.frames_seen += 1
        self._measurement = measurement
        
    def cycle(self):
        if not self._seen:
            self.frames_since_being_seen += 1
        self._seen = False

    # def initialize(self):
    #     # TODO: is this right? Initial state guess comes from last two measurements?
    #     pos = self.measurements[-1][0:2,:]
    #     vel = (self.measurements[-1][0:2,:] - self.measurements[-2][0:2,:]) / self.Ts
    #     box_size = np.mean((self.measurements[-1][2:4,:], self.measurements[-2][2:4,:]), axis=0)
    #     self.state = np.concatenate((pos, box_size, vel), axis=0)
    #     self.measurements = [self.measurements[-1]]

    def predict(self):
        xhat = self._state
        z = self._measurement
        self.xbar = dict(); self.u = dict(); self.U = dict()
        self.xbar[self._id[0]] = self.A @ xhat
        self.u[self._id[0]] = self.H.T @ inv(self.R) @ z
        self.U[self._id[0]] = self.H.T @ inv(self.R) @ self.H

    def observation_update(self, observation_msg):
        self.xhat[observation_msg.tracker_id[0]] = observation_msg.xhat
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


# class ManagedTracker():

#     def __init__(self, local_ids):
#         self.local_ids = {}
#         self._state = None # first ekf run
#         for cam_num, tracker_id in enumerate(local_ids):
#             self.local_ids[cam_num] = tracker_id
        
#         self.measurements = []
#         self.color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
#         self.frames_since_being_seen = 0
#         self.seen = True
#         self.frames_seen = 1

#     @property
#     def state(self):
#         return self._state

#     def id(self, camera_number):
#         return self.local_ids[camera_number]

#     def next_cycle(self):
#         if self.seen:
#             self.frames_since_being_seen = 0
#         else:
#             self.frames_since_being_seen += 1
#         self.seen = False
#         self.measurements = []

#     def update(self, measurement):
#         # TODO: somewhere the EKF will need to happen
#         self.measurements.append(measurement)
#         self.seen = True
#         self.frames_seen += 1