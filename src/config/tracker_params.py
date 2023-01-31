import numpy as np

Ts = 1
A = np.eye(6); A[0,4] = Ts; A[1,5] = Ts
# TODO: Something about my Ts seems to be off, changing it significantly changed results
H = np.eye(4,6)
Q = np.array([
    [(Ts**4)/4, 0, 0, 0, (Ts**3)/2, 0],
    [0, (Ts**4)/4, 0, 0, 0, (Ts**3)/2],
    [0,0,1,0,0,0],
    [0,0,0,1,0,0],
    [(Ts**3)/2,0,0,0,Ts**2,0],
    [0,(Ts**3)/2,0,0,0,Ts**2]
])
R = np.eye(4)*2
P = np.eye(6)

n_recent_dets = 100 # how many recent detections to keep