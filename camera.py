import numpy as np
from numpy.linalg import norm as norm
from copy import copy
from tracker import Tracker

class Camera():

    def __init__(self, camera_id, Tau_LDA=.5, alpha=2000, kappa=4):
        # TODO: Tune for Tau
        Tau_LDA = 30
        Tau_GDA = 30
        self.Tau_LDA = Tau_LDA
        self.Tau_GDA = Tau_GDA
        self.alpha = alpha
        self.kappa = kappa
        self.trackers = []
        self.new_trackers = []
        self.next_available_id = 0
        self.n_meas_to_init_tracker = 2
        self.camera_id = camera_id
        self.tracker_mapping = dict()
        self.unassociated_obs = []

    def local_data_association(self, Zs, feature_vecs):
        geometry_scores = dict()
        appearance_scores = dict()
        for tracker in self.trackers + self.new_trackers:
            geometry_scores[tracker.id] = np.zeros((len(Zs)))
            appearance_scores[tracker.id] = np.zeros((len(Zs)))
            Hx_xy = (tracker.H @ tracker.state)[0:2,:]
            for j, (Z, aj) in enumerate(zip(Zs, feature_vecs)):
                z_xy = Z[0:2,:]
                # Mahalanobis distance
                d = np.sqrt((z_xy - Hx_xy).T @ np.linalg.inv(tracker.V) @ (z_xy - Hx_xy)).item(0)
                
                # Geometry similarity value
                s_d = 1/self.alpha*d if d < self.Tau_LDA else 1
                geometry_scores[tracker.id][j] = s_d
                
                # Feature similarity value (if geometry is consistent)
                if s_d < 1:
                    s_a = float('inf')
                    for ai in tracker.feature_vecs:
                        s_a_new = 1 - (aj.T @ ai) / (norm(aj)*norm(ai))
                        s_a = min(s_a, s_a_new)
                else:
                    s_a = 1
                appearance_scores[tracker.id][j] = s_a

        tracker_pairs = dict()
        Zs_associated = []
        Z_best_set = set()
        product_scores = dict()
        
        for tracker_id in geometry_scores:
            product_scores[tracker_id] = geometry_scores[tracker_id] * appearance_scores[tracker_id] # element by element multiplication
        for Z in Zs:
            Zs_associated.append(False)
        if Zs:
            for i, tracker in enumerate(self.trackers + self.new_trackers):
                Z_best_idx = np.argmin(product_scores[tracker.id])
                is_current_tracker = i < len(self.trackers)
                # TODO
                # I don't think this is the right fix... Shouldn't be needed when we are using appearance information
                if is_current_tracker:
                    Z_best_set.add(Z_best_idx)
                elif Z_best_idx in Z_best_set: # prune new trackers that have measurements associated with a current tracker
                    self.new_trackers.remove(tracker)
                    continue
                Z_best = Zs[Z_best_idx]
                if product_scores[tracker.id][Z_best_idx] < 1: # TODO: should I really compare to Tau here
                    if is_current_tracker and tracker.seen_by_this_camera:
                        tracker.update(Z_best)
                    elif is_current_tracker and not tracker.seen_by_this_camera:
                        tracker.update(Z_best, feature_vecs[Z_best_idx])
                        tracker.seen_by_this_camera = True
                        tracker.include_appearance_in_obs()
                    else:
                        tracker.update(Z_best, feature_vecs[Z_best_idx])
                    Zs_associated[Z_best_idx] = True

        for tracker in self.new_trackers:
            if tracker.frames_seen >= self.n_meas_to_init_tracker:
                tracker.include_appearance_in_obs()
                self.trackers.append(tracker)
                self.tracker_mapping[tracker.id] = tracker.id
                self.new_trackers.remove(tracker)
            elif not tracker.seen:
                self.new_trackers.remove(tracker)

        for j, (Z, aj) in enumerate(zip(Zs, feature_vecs)):
            if not Zs_associated[j]:
                new_tracker = Tracker(self.camera_id, self.next_available_id, Z, aj)
                new_tracker.seen_by_this_camera = True
                self.new_trackers.append(new_tracker)
                self.next_available_id += 1

        for tracker in self.trackers + self.new_trackers:
            tracker.predict()

    def dkf(self):  
        for tracker in self.trackers + self.new_trackers:
            tracker.correction()

    def tracker_manager(self):
        # no initialization necessary if no unassociated observations
        if not self.unassociated_obs:
            self.manage_deletions()
            return
        tracked_states = []
        for tracker in self.trackers + self.new_trackers:
            tracker.include_appearance_in_obs()
            m = tracker.observation_msg()
            m.xbar = tracker.state
            tracked_states.append(m)
        geometry_scores = np.zeros((len(self.unassociated_obs), len(self.unassociated_obs + tracked_states)))
        appearance_scores = np.zeros((len(self.unassociated_obs), len(self.unassociated_obs + tracked_states)))
        # TODO: This iteration stuff happens in three places now. Clean up and put in some function
        for i in range(len(self.unassociated_obs)):
            xi_xy = self.unassociated_obs[i].xbar[0:2,:]
            for j in range(len(self.unassociated_obs)):
                if i == j:
                    continue
                xj_xy = self.unassociated_obs[j].xbar[0:2,:]
                # distance
                d = np.sqrt((xi_xy - xj_xy).T @ (xi_xy - xj_xy)).item(0)
                
                # Geometry similarity value
                s_d = 1/self.alpha*d if d < self.Tau_GDA else 1
                geometry_scores[i,j] = s_d
                
                # Appearance similarity value
                if s_d < 1:
                    s_a = float('inf')
                    for ai in self.unassociated_obs[i].a:
                        for aj in self.unassociated_obs[j].a:
                            s_a_new = 1 - (aj.T @ ai) / (norm(aj)*norm(ai))
                            s_a = min(s_a, s_a_new)
                else:
                    s_a = 1
                appearance_scores[i,j] = s_a
                
            for j in range(len(tracked_states)):
                j_gs = j+len(self.unassociated_obs) # j for indexing into geometry_scores
                xj_xy = tracked_states[j].xbar[0:2,:]
                # distance
                d = np.sqrt((xi_xy - xj_xy).T @ (xi_xy - xj_xy)).item(0)
                
                # Geometry similarity value
                s_d = 1/self.alpha*d if d < self.Tau_GDA else 1
                geometry_scores[i,j_gs] = s_d
                
                # Appearance similarity value
                if s_d < 1:
                    s_a = float('inf')
                    for ai in self.unassociated_obs[i].a:
                        for aj in tracked_states[j].a:
                            s_a_new = 1 - (aj.T @ ai) / (norm(aj)*norm(ai))
                            s_a = min(s_a, s_a_new)
                else:
                    s_a = 1
                appearance_scores[i,j_gs] = s_a

        product_scores = geometry_scores * appearance_scores # element-wise multiplication
        tracker_groups = self._get_tracker_groups(product_scores)
        
        self._create_trackers(tracker_groups, tracked_states)
        unassociated_obs = self.unassociated_obs
        len_unassociated_obs = self.add_observations(unassociated_obs)
        assert len_unassociated_obs == 0
        
        self.manage_deletions()
                
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
                assert obs.has_appearance_info, f'From camera: {self.camera_id}, No appearance info: {obs}'
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

    def _get_tracker_groups(self, similarity_scores):
        # TODO: Maximum clique kind of thing going on here...
        # Handle better

        tracker_groups = []
        new_tracker_groups = []
        for i in range(similarity_scores.shape[0]):
            group = set()
            for j in range(similarity_scores.shape[0]):
                if similarity_scores[i,j] < .1: # TODO: Magic number here?
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
        
        # associate with local indexes
        if similarity_scores.shape[0] != similarity_scores.shape[1]:
            for group in new_tracker_groups:
                for i in group:
                    local_idx = np.argmin(similarity_scores[i,similarity_scores.shape[0]:similarity_scores.shape[1]])
                    local_idx += similarity_scores.shape[0]
                    if similarity_scores[i,local_idx] < 1:
                        group.add(local_idx)
                        break
        
        return new_tracker_groups
    
    def _create_trackers(self, tracker_groups, tracked_states):
        for group in tracker_groups:
            local_tracker_id = None
            mean_xbar = np.zeros((6,1))
            group_size = 0
            appearance_vecs = []
            
            # Assign local tracker id
            for index in group:
                if index >= len(self.unassociated_obs):
                    # If associated with a currently "new" tracker
                    if index >= len(self.unassociated_obs) + len(self.trackers):
                        t = self.new_trackers[index-(len(self.unassociated_obs)+len(self.trackers))]
                        self.trackers.append(t)
                        self.tracker_mapping[t.id] = t.id
                        self.new_trackers.remove(t)
                    # assert local_tracker_id == None or local_tracker_id == tracked_states[index-len(self.unassociated_obs)].tracker_id, \
                    #      f'1: {local_tracker_id}, 2: {tracked_states[index-len(self.unassociated_obs)].tracker_id}'
                    local_tracker_id = tracked_states[index-len(self.unassociated_obs)].tracker_id
                    continue
                elif self.unassociated_obs[index].tracker_id[0] == self.camera_id:
                    # assert local_tracker_id == None or local_tracker_id == self.unassociated_obs[index].tracker_id, \
                    #     f'1: {local_tracker_id}, 2: {self.unassociated_obs[index].tracker_id}'
                    local_tracker_id = self.unassociated_obs[index].tracker_id
                
                # if this is an unassociated observation (not a currently tracked tracker)
                if index < len(self.unassociated_obs):
                    mean_xbar += self.unassociated_obs[index].xbar
                    group_size += 1
                    appearance_vecs += self.unassociated_obs[index].a
            if local_tracker_id == None:
                mean_xbar /= group_size
                local_tracker_id = (self.camera_id, self.next_available_id)
                new_tracker = Tracker(self.camera_id, self.next_available_id, mean_xbar[0:4,:], appearance_vecs)
                # new_tracker.include_appearance_in_obs()
                self.trackers.append(new_tracker)
                self.next_available_id += 1
            else:
                for tracker in self.trackers:
                    if tracker.id == local_tracker_id:
                        tracker.add_appearance_gallery(appearance_vecs)
                        break

            for index in group:
                if index >= len(self.unassociated_obs):
                    continue
                self.tracker_mapping[self.unassociated_obs[index].tracker_id] = local_tracker_id

    def manage_deletions(self):
        for tracker in self.trackers + self.new_trackers:
            tracker.cycle()
            if tracker.ell > self.kappa:
                self.trackers.remove(tracker)
            
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