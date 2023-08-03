import numpy as np

class MotionModel():
    
    def __init__(self, A, H, Q, R, P0):
        """
        Typical discrete-time motion model parameters
        x[k+1] = Ax[k] + w[k]
        z[k] = Hx[k] + v[k]
        w ~ N(0, Q)
        v ~ N(0, R)

        Args:
            A ((n,n) np.array): transition matrix
            H ((p,n) np.array): measurement matrix
            Q ((n,n) np.array): process noise covariance
            R ((p,p) np.array): measurement noise covariance
            P0 ((n,n) np.array): initial estimate covariance
        """
        self.A = A.copy()
        self.H = H.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        self.P0 = P0.copy()