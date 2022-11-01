import enum
import numpy as np

class LocalDataAssociation():

    def __init__(self, H=np.eye(4,6), Tau=.5, alpha=2000):
        self.R = np.zeros((4,4))
        self.P = np.zeros((6,6))
        self.V = np.eye(2)
        # TODO: Change this
        Tau = 40
        self.Tau = Tau
        self.alpha = alpha
        self.H = H
        self.Xs = []
        self.new_Zs = []
        self.unassociated_Zs = []
        self.colors = []

    def update_measurements(self, Zs):
        geometry_scores = np.zeros((len(self.Xs), len(Zs)))
        for i, X in enumerate(self.Xs):
            Hx_xy = (self.H @ X)[0:2,:]
            for j, Z in enumerate(Zs):
                z_xy = Z[0:2,:]
                # Mahalanobis distance
                d = np.sqrt((z_xy - Hx_xy).T @ np.linalg.inv(self.V) @ (z_xy - Hx_xy))
                
                # Geometry similarity value
                s_d = 1/self.alpha*d if d < self.Tau else 1
                geometry_scores[i,j] = s_d

        # TODO:
        # Add in support for feature matching too

        xz_pairs = []
        self.new_Zs = []
        for i, X in enumerate(self.Xs):
            # TODO: Remove below
            if not Zs:
                break
            xz_pairs.append(np.argmin(geometry_scores[i,:]))
            self.new_Zs.append(Zs[xz_pairs[i]])

        # TODO:
        # assuming here that no detections ever leave, will need to fix

        # TODO:
        # tracker manager for how deleting and starting trackers actually works

        self.unassociated_Zs = []
        for j, Z in enumerate(Zs):
            if j not in xz_pairs:
                self.unassociated_Zs.append(Z)


    def update_uncertainty(self, P, R):
        P_xy = P[0:2,0:2]
        R_xy = R[0:2,0:2]
        self.V = P_xy + R_xy


    def get_trackers(self):
        # TODO: change what we're returning here
        self.Xs = []
        for Z in self.new_Zs:
            self.Xs.append(np.concatenate((Z, np.zeros((2,1))), 0))
        if self.unassociated_Zs:
            for Z in self.unassociated_Zs:
                self.Xs.append(np.concatenate((Z, np.zeros((2,1))), 0))
                self.colors.append(
                    (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)))
        return self.Xs, self.colors