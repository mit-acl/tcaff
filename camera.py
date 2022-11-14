import numpy as np
from tracker import Tracker

class Camera():

    def __init__(self, camera_id, Tau=.5, alpha=2000):
        # TODO: Tune for Tau
        Tau = 50
        self.Tau = Tau
        self.alpha = alpha
        self.trackers = []
        self.new_trackers = []
        self.next_available_id = 0
        self.n_meas_to_init_tracker = 2
        self.camera_id = camera_id
        self.tracker_mapping = dict()
        self.unassociated_obs = []

    def local_data_association(self, Zs):
        for tracker in self.trackers + self.new_trackers:
            tracker.cycle()
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
        Z_best_set = set()
        for Z in Zs:
            Zs_associated.append(False)
        if Zs:
            for i, tracker in enumerate(self.trackers + self.new_trackers):
                Z_best_idx = np.argmin(geometry_scores[tracker.id])
                # TODO
                # I don't think this is the right fix... Shouldn't be needed when we are using appearance information
                if i < len(self.trackers):
                    Z_best_set.add(Z_best_idx)
                elif Z_best_idx in Z_best_set: # prune new trackers that have measurements associated with a current tracker
                    self.new_trackers.remove(tracker)
                    continue
                Z_best = Zs[Z_best_idx]
                if geometry_scores[tracker.id][Z_best_idx] < 1: # TODO: should I really compare to Tau here
                    tracker.update(Z_best)
                    Zs_associated[Z_best_idx] = True

        for tracker in self.new_trackers:
            if tracker.frames_seen >= self.n_meas_to_init_tracker:
                self.trackers.append(tracker)
                self.tracker_mapping[tracker.id] = tracker.id
                self.new_trackers.remove(tracker)
            elif not tracker.seen:
                self.new_trackers.remove(tracker)

        # TODO:
        # assuming here that no detections ever leave, will need to fix

        # TODO:
        # tracker manager for how deleting and starting trackers actually works

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

    def tracker_manager(self):
        tracked_states = []
        for tracker in self.trackers + self.new_trackers:
            m = tracker.observation_msg()
            m.xbar = tracker.state
            tracked_states.append(m)
        geometry_scores = np.zeros((len(self.unassociated_obs), len(self.unassociated_obs + tracked_states)))
        for i in range(len(self.unassociated_obs)):
            xi_xy = self.unassociated_obs[i].xbar[0:2,:]
            for j in range(len(self.unassociated_obs)):
                if i == j:
                    continue
                xj_xy = self.unassociated_obs[j].xbar[0:2,:]
                # Mahalanobis distance
                d = np.sqrt((xi_xy - xj_xy).T @ (xi_xy - xj_xy)).item(0)
                
                # Geometry similarity value
                s_d = 1/self.alpha*d if d < self.Tau else 1
                geometry_scores[i,j] = s_d
            for j in range(len(tracked_states)):
                j_gs = j+len(self.unassociated_obs) # j for indexing into geometry_scores
                # if self.unassociated_obs[i].tracker_id[0] == tracked_states[j].tracker_id[0]: # same camera
                #     geometry_scores[i,j] = 1
                #     continue
                xj_xy = tracked_states[j].xbar[0:2,:]
                # Mahalanobis distance
                d = np.sqrt((xi_xy - xj_xy).T @ (xi_xy - xj_xy)).item(0)
                
                # Geometry similarity value
                s_d = 1/self.alpha*d if d < self.Tau else 1
                geometry_scores[i,j_gs] = s_d

        # TODO:
        # Add in support for feature matching too

        tracker_groups = self._get_tracker_groups(geometry_scores)
        
        self._create_trackers(tracker_groups, tracked_states)
        unassociated_obs = self.unassociated_obs
        len_unassociated_obs = self.add_observations(unassociated_obs)
        assert len_unassociated_obs == 0


    def get_observations(self):
        observations = []
        for tracker in self.trackers:
            if tracker.seen:
                observations.append(tracker.observation_msg())
        return observations

    def add_observations(self, observations):
        self.unassociated_obs = []
        for obs in observations:
            if obs.tracker_id in self.tracker_mapping: 
                target_tracker_id = self.tracker_mapping[obs.tracker_id]
                for tracker in self.trackers:
                    if tracker.id == target_tracker_id:
                        tracker.observation_update(obs)
                        break                
            else: 
                self.unassociated_obs.append(obs)
        return len(self.unassociated_obs)
            
    def get_trackers(self):
        # TODO: change what we're returning here
        Xs = []
        colors = []
        for tracker in self.trackers:
            if tracker.id[0] == self.camera_id: # only return local trackers
                Xs.append(tracker.state)
                colors.append(tracker.color)
        return Xs, colors

    def _get_tracker_groups(self, geometry_scores):
        tracker_groups = []
        new_tracker_groups = []
        for i in range(geometry_scores.shape[0]):
            group = set()
            for j in range(geometry_scores.shape[1]):
                if geometry_scores[i,j] < 1:
                    group.add(j)
            tracker_groups.append(group)

        while tracker_groups: 
            g1 = tracker_groups.pop(0)
            added = []
            for member in g1:
                for g2 in new_tracker_groups:
                    if member in g2:
                        new_tracker_groups.remove(g2)
                        g2 = g2.union(g1)
                        new_tracker_groups.append(g2)
                        added.append(g2)
            if not added:
                new_tracker_groups.append(g1)
            elif len(added) == 1:
                continue
            else:
                for g in added[1:]:
                    new_tracker_groups.remove(added[0])
                    if g in new_tracker_groups:
                        new_tracker_groups.remove(g)
                    added[0] = added[0].union(g)
                    new_tracker_groups.append(added[0])
        return new_tracker_groups
    
    def _create_trackers(self, tracker_groups, tracked_states):
        for group in tracker_groups:
            local_tracker_id = None
            mean_xbar = np.zeros((6,1))
            group_size = 0
            for index in group:
                if index >= len(self.unassociated_obs):
                    if index >= len(self.unassociated_obs) + len(self.trackers):
                        t = self.new_trackers[index-(len(self.unassociated_obs)+len(self.trackers))]
                        self.trackers.append(t)
                        self.tracker_mapping[t.id] = t.id
                        self.new_trackers.remove(t)
                    assert local_tracker_id == None or local_tracker_id == tracked_states[index-len(self.unassociated_obs)].tracker_id, \
                         f'1: {local_tracker_id}, 2: {tracked_states[index-len(self.unassociated_obs)].tracker_id}'
                    local_tracker_id = tracked_states[index-len(self.unassociated_obs)].tracker_id
                    continue
                elif self.unassociated_obs[index].tracker_id[0] == self.camera_id:
                    assert local_tracker_id == None or local_tracker_id == self.unassociated_obs[index].tracker_id, \
                        f'1: {local_tracker_id}, 2: {self.unassociated_obs[index].tracker_id}'
                    local_tracker_id = self.unassociated_obs[index].tracker_id
                if index < len(self.unassociated_obs):
                    mean_xbar += self.unassociated_obs[index].xbar
                    group_size += 1
            if local_tracker_id == None:
                mean_xbar /= group_size
                local_tracker_id = (self.camera_id, self.next_available_id)
                new_tracker = Tracker(self.camera_id, self.next_available_id, mean_xbar[0:4,:])
                self.trackers.append(new_tracker)
                self.next_available_id += 1

            for index in group:
                if index >= len(self.unassociated_obs):
                    continue
                self.tracker_mapping[self.unassociated_obs[index].tracker_id] = local_tracker_id
            
    def __str__(self):
        return_str = ''
        return_str += f'Camera {self.camera_id}\n'
        if self.trackers:
            return_str += 'Trackers:\n'
            for t in self.trackers:
                return_str += f'\t{t}\n'
        if self.new_trackers:
            return_str += 'New Trackers:\n'
            for t in self.new_trackers:
                return_str += f'\t{t}\n'
        if self.tracker_mapping:
            return_str += 'Mappings:\n'
            for global_t, local_t in self.tracker_mapping.items():
                return_str += f'\t{global_t} --> {local_t}\n'
        # return_str += ''
        return return_str