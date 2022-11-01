import enum
import numpy as np

class Target():

    def __init__(self, id, state):
        self._id = id
        self._state = state
        self.color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        self.frames_since_being_seen = 0
        self.frames_seen = 1
        print(id)

    @property
    def X(self):
        return self._state

    @property
    def id(self):
        return self._id

    def update(self, measurement):
        # TODO: somewhere the EKF will need to happen
        self._state = np.concatenate((measurement, np.zeros((2,1))), 0)
        self.frames_since_being_seen = 0
        self.frames_seen += 1
    
    def no_association(self):
        self.frames_since_being_seen += 1

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
        self.targets = []
        self.new_targets = []
        self.next_available_id = 0
        self.n_meas_to_init_target = 2

    def update_measurements(self, Zs):
        geometry_scores = dict()
        for target in self.targets + self.new_targets:
            geometry_scores[target.id] = np.zeros((len(Zs)))
            Hx_xy = (self.H @ target.X)[0:2,:]
            for j, Z in enumerate(Zs):
                z_xy = Z[0:2,:]
                # Mahalanobis distance
                d = np.sqrt((z_xy - Hx_xy).T @ np.linalg.inv(self.V) @ (z_xy - Hx_xy)).item(0)
                
                # Geometry similarity value
                s_d = 1/self.alpha*d if d < self.Tau else 1
                print(s_d)
                geometry_scores[target.id][j] = s_d

        # TODO:
        # Add in support for feature matching too

        target_pairs = dict()
        Zs_associated = []
        for Z in Zs:
            Zs_associated.append(False)
        if Zs:
            for target in self.targets + self.new_targets:
                Z_best_idx = np.argmin(geometry_scores[target.id])
                Z_best = Zs[Z_best_idx]
                if geometry_scores[target.id][Z_best_idx] < 1: # TODO: should I really compare to Tau here
                    target.update(Z_best)
                    Zs_associated[Z_best_idx] = True
                else:
                    target.no_association()

        for target in self.new_targets:
            if target.frames_seen >= self.n_meas_to_init_target:
                self.targets.append(target)
                self.new_targets.remove(target)

        # TODO:
        # assuming here that no detections ever leave, will need to fix

        # TODO:
        # tracker manager for how deleting and starting trackers actually works

        self.unassociated_Zs = []
        for j, Z in enumerate(Zs):
            if not Zs_associated[j]:
                self.unassociated_Zs.append(Z)

        for Z in self.unassociated_Zs:
            new_target = Target(self.next_available_id, np.concatenate((Z, np.zeros((2,1))), 0))
            self.new_targets.append(new_target)
            self.next_available_id += 1


    def update_uncertainty(self, P, R):
        P_xy = P[0:2,0:2]
        R_xy = R[0:2,0:2]
        self.V = P_xy + R_xy


    def get_trackers(self):
        # TODO: change what we're returning here
        Xs = []
        colors = []
        for target in self.targets:
            Xs.append(target.X)
            colors.append(target.color)
        return Xs, colors