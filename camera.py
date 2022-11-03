import numpy as np
from tracker import Tracker

class Camera():

    def __init__(self, camera_id, Tau=.5, alpha=2000):
        # TODO: Change this
        Tau = 40
        self.Tau = Tau
        self.alpha = alpha
        self.trackers = []
        self.new_trackers = []
        self.next_available_id = 0
        self.n_meas_to_init_tracker = 2
        self.camera_id = camera_id
        self.tracker_mapping = dict()

    def local_data_association(self, Zs):
        geometry_scores = dict()
        for tracker in self.trackers + self.new_trackers:
            geometry_scores[tracker.id] = np.zeros((len(Zs)))
            Hx_xy = (tracker.H @ tracker.state)[0:2,:]
            for j, Z in enumerate(Zs):
                z_xy = Z[0:2,:]
                # Mahalanobis distance
                d = np.sqrt((z_xy - Hx_xy).T @ np.linalg.inv(tracker.V) @ (z_xy - Hx_xy)).item(0)
                
                # Geometry similarity value
                s_d = 1/self.alpha*d if d < self.Tau else 1
                geometry_scores[tracker.id][j] = s_d

        # TODO:
        # Add in support for feature matching too

        tracker_pairs = dict()
        Zs_associated = []
        for Z in Zs:
            Zs_associated.append(False)
        if Zs:
            for tracker in self.trackers + self.new_trackers:
                Z_best_idx = np.argmin(geometry_scores[tracker.id])
                Z_best = Zs[Z_best_idx]
                if geometry_scores[tracker.id][Z_best_idx] < 1: # TODO: should I really compare to Tau here
                    tracker.update(Z_best)
                    Zs_associated[Z_best_idx] = True

        for tracker in self.new_trackers:
            if tracker.frames_seen >= self.n_meas_to_init_tracker:
                self.trackers.append(tracker)
                self.new_trackers.remove(tracker)
            elif not tracker.seen:
                self.new_trackers.remove(tracker)

        # TODO:
        # assuming here that no detections ever leave, will need to fix

        # TODO:
        # tracker manager for how deleting and starting trackers actually works

        self.unassociated_Zs = []
        for j, Z in enumerate(Zs):
            if not Zs_associated[j]:
                new_tracker = Tracker(self.camera_id, self.next_available_id, Z)
                self.new_trackers.append(new_tracker)
                self.next_available_id += 1

        for tracker in self.trackers + self.new_trackers:
            tracker.predict()

    def dkf(self):  
        for tracker in self.trackers + self.new_trackers:
            tracker.correction()

    def get_observations(self):
        observations = []
        for tracker in self.trackers:
            observations.append(tracker.observation_msg())
        return observations

    def add_observations(self, observations):
        for obs in observations:
            if obs.tracker_id[0] == self.camera_id: continue
            # TODO: Change this as soon as the tracker manager is implemented.
            if obs.tracker_id not in self.tracker_mapping: continue
            target_tracker_id = self.tracker_mapping[obs.tracker_id]
            [tracker_idx for tracker_idx, tracker in enumerate(self.trackers) if tracker.id == target_tracker_id]
            self.trackers[tracker_idx].observation_update(obs)
            
    def get_trackers(self):
        # TODO: change what we're returning here
        Xs = []
        colors = []
        for tracker in self.trackers:
            Xs.append(tracker.state)
            colors.append(tracker.color)
        return Xs, colors