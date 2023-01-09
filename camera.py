import numpy as np
from numpy.linalg import norm as norm
from scipy.optimize import linear_sum_assignment

from tracker import Tracker

class Camera():

    def __init__(self, camera_id, Tau_LDA=1, alpha=2000, kappa=4):
        # TODO: Tune for Tau
        Tau_GDA = Tau_LDA
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
        self.inconsistencies = 0

    def local_data_association(self, Zs, feature_vecs):
        all_trackers = self.trackers + self.new_trackers
        geometry_scores = np.zeros((len(all_trackers), len(Zs)))
        appearance_scores = np.zeros((len(all_trackers), len(Zs)))
        large_num = 1000
        for i, tracker in enumerate(all_trackers):
            Hx_xy = (tracker.H @ tracker.state)[0:2,:]
            for j, (Z, aj) in enumerate(zip(Zs, feature_vecs)):
                z_xy = Z[0:2,:]
                # Mahalanobis distance
                d = np.sqrt((z_xy - Hx_xy).T @ np.linalg.inv(tracker.V) @ (z_xy - Hx_xy)).item(0)
                
                # Geometry similarity value
                s_d = 1/self.alpha*d if d < self.Tau_LDA else large_num
                geometry_scores[i,j] = s_d
                
                # Feature similarity value (if geometry is consistent)
                if s_d < 1:
                    s_a = float('inf')
                    for ai in tracker.feature_vecs:
                        s_a_new = 1 - (aj.T @ ai) / (norm(aj)*norm(ai))
                        s_a = min(s_a, s_a_new)
                else:
                    s_a = 1
                appearance_scores[i,j] = s_a

        unassociated = []
        product_scores = geometry_scores * appearance_scores # element-wise multiplication
        # augment cost to add option for no associations
        hungarian_cost = np.concatenate([
            np.concatenate([product_scores, np.ones(product_scores.shape)], axis=1),
            np.ones((product_scores.shape[0], 2*product_scores.shape[1]))], axis=0)
        row_ind, col_ind = linear_sum_assignment(hungarian_cost)
        for t_idx, z_idx in zip(row_ind, col_ind):
            # tracker and measurement associated together
            if t_idx < len(all_trackers) and z_idx < len(Zs):
                assert product_scores[t_idx,z_idx] < 1
                is_current_tracker = t_idx < len(self.trackers)
                if is_current_tracker and all_trackers[t_idx].seen_by_this_camera:
                    all_trackers[t_idx].update(Zs[z_idx])
                elif is_current_tracker and not all_trackers[t_idx].seen_by_this_camera:
                    all_trackers[t_idx].update(Zs[z_idx], feature_vecs[z_idx])
                    all_trackers[t_idx].seen_by_this_camera = True
                    all_trackers[t_idx].include_appearance_in_obs()
                else:
                    all_trackers[t_idx].update(Zs[z_idx], feature_vecs[z_idx])            
            # unassociated measurement
            elif z_idx < len(Zs):
                unassociated.append(z_idx)
            # unassociated tracker or augmented part of matrix
            else:
                continue
            
        # if there are no trackers, hungarian matrix will be empty, handle separately
        if len(all_trackers) == 0:
            for z_idx in range(len(Zs)):
                unassociated.append(z_idx)

        for tracker in self.new_trackers:
            if tracker.frames_seen >= self.n_meas_to_init_tracker:
                tracker.include_appearance_in_obs()
                self.trackers.append(tracker)
                self.tracker_mapping[tracker.id] = tracker.id
                self.new_trackers.remove(tracker)
            elif not tracker.seen:
                self.new_trackers.remove(tracker)

        for z_idx in unassociated:
            new_tracker = Tracker(self.camera_id, self.next_available_id, Zs[z_idx], feature_vecs[z_idx])
            new_tracker.update(Zs[z_idx])
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
        return
                
    def get_observations(self):
        observations = []
        for tracker in self.trackers:
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
                matched = False
                for mid in obs.mapped_ids:
                    if mid in self.tracker_mapping:
                        matched = True
                        target_tracker_id = self.tracker_mapping[mid]
                        self.tracker_mapping[obs.tracker_id] = target_tracker_id
                        for tracker in self.trackers:
                            if tracker.id == target_tracker_id:
                                tracker.observation_update(obs)
                                break
                        break
                if not matched:                        
                    assert obs.has_appearance_info, f'From camera: {self.camera_id}, No appearance info: {obs}'
                    self.unassociated_obs.append(obs)
        return len(self.unassociated_obs)
            
    def get_trackers(self, format='state_color'):
        # TODO: change what we're returning here
        if format == 'state_color':
            Xs = []
            colors = []
            for tracker in self.trackers:
                assert tracker.id[0] == self.camera_id
                Xs.append(tracker.state)
                colors.append(tracker.color)
            return Xs, colors
        elif format == 'dict':
            tracker_dict = dict()
            for tracker in self.trackers:
                tracker_dict[tracker.id] = np.array(tracker.state[0:2,:].reshape(-1))
            return tracker_dict

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
        self.inconsistencies = 0
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
                    if local_tracker_id != None and local_tracker_id != tracked_states[index-len(self.unassociated_obs)].tracker_id:
                        # print('yeah it happened')
                        self.inconsistencies += 1
                    local_tracker_id = tracked_states[index-len(self.unassociated_obs)].tracker_id
                    continue
                elif self.unassociated_obs[index].tracker_id[0] == self.camera_id:
                    # assert local_tracker_id == None or local_tracker_id == self.unassociated_obs[index].tracker_id, \
                    #     f'1: {local_tracker_id}, 2: {self.unassociated_obs[index].tracker_id}'
                    if local_tracker_id != None and local_tracker_id != self.unassociated_obs[index].tracker_id:
                        # print('yeah it happened')
                        self.inconsistencies += 1
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
        
        return

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