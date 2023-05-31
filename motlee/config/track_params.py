import numpy as np

class TrackParams():
    '''
    Default track parameters
    '''
    def __init__(self, ts=.1):
    # def __init__(self, ts=.1):
        self.ts = ts
        self.A = np.array([
            [1, 0, ts, 0],
            [0, 1, 0, ts],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        self.H = np.eye(2, 4)
        self.Q = 50*np.array([
            [(ts**4)/4, 0, (ts**3)/2, 0],
            [0, (ts**4)/4, 0, (ts**3)/2],
            [(ts**3)/2, 0, ts**2, 0],
            [0, (ts**3)/2, 0, ts**2]
        ])#*1000
        self.R = .75*np.eye(2)
        self.P = np.eye(4)#*1000
        # self.P = np.copy(self.Q)

        self.n_dets = 100 # how many recent detections to keep
        
class ConeParams():
    '''
    Default cone parameters
    '''
    def __init__(self, ts=.1):
        self.ts = ts
        self.A = np.array([
            [1, 0],
            [0, 1],
        ], dtype=np.float64)
        self.H = np.eye(2)
        self.Q = np.eye(2)
        self.R = .75*np.eye(2)
        self.P = np.eye(2)#*1000
        # self.P = np.copy(self.Q)

        self.n_dets = 1 # how many recent detections to keep