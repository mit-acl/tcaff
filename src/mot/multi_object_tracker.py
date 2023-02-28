import numpy as np
from numpy.linalg import norm as norm
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as Rot
from copy import deepcopy

from .tracker import Tracker
from config import tracker_params as TRACK_PARAM
from realign.frame_realigner import FrameRealigner

NUM_CAMS = 4
class MultiObjectTracker():

    def __init__(self, camera_id, connected_cams, params):
        self.connected_cams = connected_cams
        self.realigner = FrameRealigner(
            cam_id=camera_id,
            connected_cams=connected_cams,
            params=params 
        )
        # TODO: Tune for Tau
        self.Tau_init = params.Tau_LDA
        self.Tau_LDA = params.Tau_LDA
        self.Tau_GDA = params.Tau_GDA
        self.Tau_grown = 1
        self.Tau_cone = 0.5
        
        self.alpha = params.alpha
        self.kappa = params.kappa
        self.trackers = []
        self.new_trackers = []
        self.old_trackers = []
        self.cones = []
        self.new_cones = []
        self.next_available_id = 0
        self.n_meas_to_init_tracker = params.n_meas_to_init_tracker
        
        self.camera_id = camera_id
        self.tracker_mapping = dict()
        self.unassociated_obs = []
        self.inconsistencies = 0
        self.groups_by_id = []
        self.recent_detection_list = []
        for i in range(50):
            self.recent_detection_list.append(None)
        

    def local_data_association(self, Zs, feature_vecs):
        frame_detections = []
        for z in Zs:
            frame_detections.append(z[0:2,:].reshape(-1,1))
        self.recent_detection_list.insert(0, frame_detections)
        self.recent_detection_list = self.recent_detection_list[:-1]
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
            
    def cone_update(self, Zs):
        for i in range(len(Zs)):
            Zs[i] = np.concatenate([Zs[i][:2,:].reshape(-1), [.1, .15]]).reshape((4, 1))
        all_cones = self.cones + self.new_cones
        geometry_scores = np.zeros((len(all_cones), len(Zs)))
        large_num = 1000
        for i, tracker in enumerate(all_cones):
            Hx_xy = (tracker.H @ tracker.state)[0:2,:]
            for j, Z in enumerate(Zs):
                z_xy = Z[0:2,:]
                # Mahalanobis distance
                d = np.sqrt((z_xy - Hx_xy).T @ np.linalg.inv(tracker.V) @ (z_xy - Hx_xy)).item(0)
                
                # Geometry similarity value
                s_d = 1/self.alpha*d if d < self.Tau_cone else large_num
                geometry_scores[i,j] = s_d

        unassociated = []
        product_scores = geometry_scores 
        # augment cost to add option for no associations
        hungarian_cost = np.concatenate([
            np.concatenate([product_scores, np.ones(product_scores.shape)], axis=1),
            np.ones((product_scores.shape[0], 2*product_scores.shape[1]))], axis=0)
        row_ind, col_ind = linear_sum_assignment(hungarian_cost)
        for t_idx, z_idx in zip(row_ind, col_ind):
            # tracker and measurement associated together
            if t_idx < len(all_cones) and z_idx < len(Zs):
                assert product_scores[t_idx,z_idx] < 1
                is_current_tracker = t_idx < len(self.cones)
                all_cones[t_idx].update(Zs[z_idx])        
            # unassociated measurement
            elif z_idx < len(Zs):
                unassociated.append(z_idx)
            # unassociated tracker or augmented part of matrix
            else:
                continue
            
        # if there are no trackers, hungarian matrix will be empty, handle separately
        if len(all_cones) == 0:
            for z_idx in range(len(Zs)):
                unassociated.append(z_idx)

        for cone in self.new_cones:
            if cone.frames_seen >= self.n_meas_to_init_tracker:
                self.cones.append(cone)
                self.new_cones.remove(cone)
            elif not cone.seen:
                self.new_cones.remove(cone)

        for z_idx in unassociated:
            new_cone = Tracker(self.camera_id, self.next_available_id, Zs[z_idx], feature_vec=[], keep_history=False)
            new_cone.A = np.eye(6); new_cone.A[4:,4:] = np.zeros((2,2))
            new_cone.Q = np.eye(6)
            new_cone.update(Zs[z_idx])
            new_cone.seen_by_this_camera = True
            self.new_cones.append(new_cone)
            self.next_available_id += 1

        for cone in self.cones + self.new_cones:
            cone.predict()
            cone.correction()
            cone.cycle()
            if cone.ell > 600:
                self.cones.remove(cone)

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
        
        self.groups_by_id = self._create_trackers(tracker_groups, tracked_states)
        unassociated_obs = self.unassociated_obs
        len_unassociated_obs = self.add_observations(unassociated_obs, transform=False)
            # transform has already occured at this point
        assert len_unassociated_obs == 0
        
        self.manage_deletions()
        return
                
    def get_observations(self):
        observations = []
        for tracker in self.trackers:
            tracker_obs = tracker.observation_msg()
            for cam in self.connected_cams:
                obs = deepcopy(tracker_obs)
                obs.add_destination(cam)
                # if (cam % NUM_CAMS == 0) != (self.camera_id % NUM_CAMS == 0):
                # try:
                #     if self.realigner.realigns_since_change[cam] > 2 or self.realigner.last_change_Tmag[cam] > 1:
                #         obs.R *= 1e2
                # except:
                #     import ipdb; ipdb.set_trace()
                observations.append(obs)
        return observations

    def add_observations(self, observations, transform=True):
        # TODO: Probably not the best memory usage here...
        observations = deepcopy(observations)
        self.unassociated_obs = []
        # Process each observation
        for obs in observations:
            assert obs.destination == self.camera_id
            if transform:
                obs = self.realigner.transform_obs(obs)
            # Check if we recognize the tracker id
            if obs.tracker_id in self.tracker_mapping: 
                target_tracker_id = self.tracker_mapping[obs.tracker_id]
                for tracker in self.trackers:
                    if tracker.id == target_tracker_id:
                        tracker.observation_update(obs)
                        break
            else:
                matched = False
                # Check if the incoming message has already been paired to one of our trackers
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
                # Add to unassociated_obs for tracker initialization if this is new
                if not matched:                        
                    # assert obs.has_appearance_info, f'From camera: {self.camera_id}, No appearance info: {obs}'
                    # TODO: I can only do this because I am not using apperance vectors right now!!!
                    if not obs.has_appearance_info:
                        obs.add_appearance([np.ones((2, 1))])
                    if np.trace(obs.R) < 100:
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
        
    def get_recent_detections(self, from_trackers=False):
        if from_trackers:
            recent_detections = []
            for tracker in self.trackers:
                recent_detections.append(tracker.recent_detections)
            return recent_detections
        else: # raw detections
            return self.recent_detection_list
        
    def frame_realign(self):
        self.realigner.realign(self.trackers + self.old_trackers)
        self.realigner.rectify_detections(self.trackers + self.old_trackers)
        self.Tau_LDA = self.realigner.tolerance_scale * self.Tau_init

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
        
        # # associate with local indexes
        STOP_INCONSISTENCIES = True
        # TODO: This is a simple hack. It improves MOT performance, but can also lead to hackish results (gruops with common elements)
        if STOP_INCONSISTENCIES:
            if similarity_scores.shape[0] != similarity_scores.shape[1]:
                for group in new_tracker_groups:
                    for i in group:
                        local_idx = np.argmin(similarity_scores[i,similarity_scores.shape[0]:similarity_scores.shape[1]])
                        local_idx += similarity_scores.shape[0]
                        if similarity_scores[i,local_idx] < 1:
                            group.add(local_idx)
                            break
            # need_merging = []
            # for group in new_tracker_groups:
            #     for other_group in new_tracker_groups:
            #         if group == other_group: continue
            #         for el in group:
            #             if el in other_group:
            #                 need_merging.append((group, other_group))
            #                 break
            # for pair in need_merging:
            #     if pair[0] in new_tracker_groups and pair[1] in new_tracker_groups:
            #         new_tracker_groups.remove(pair[0]); new_tracker_groups.remove(pair[1])
            #         new_tracker_groups.append(pair[0].union(pair[1]))    
        else:
            if similarity_scores.shape[0] != similarity_scores.shape[1]:
                for group in new_tracker_groups:
                    to_add = set()
                    for i in group:
                        local_idx = np.argmin(similarity_scores[i,similarity_scores.shape[0]:similarity_scores.shape[1]])
                        local_idx += similarity_scores.shape[0]
                        if similarity_scores[i,local_idx] < .1:
                            to_add.add(local_idx)
                            # break
                    group = group.union(to_add)
        
        return new_tracker_groups
    
    def _create_trackers(self, tracker_groups, tracked_states):
        groups_by_id = list()
        for group in tracker_groups:
            group_by_id = list()
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
                    group_by_id.append(tracked_states[index-len(self.unassociated_obs)].tracker_id)
                    continue
                elif self.unassociated_obs[index].tracker_id[0] == self.camera_id:
                    # assert local_tracker_id == None or local_tracker_id == self.unassociated_obs[index].tracker_id, \
                    #     f'1: {local_tracker_id}, 2: {self.unassociated_obs[index].tracker_id}'
                    if local_tracker_id != None and local_tracker_id != self.unassociated_obs[index].tracker_id:
                        # print('yeah it happened')
                        self.inconsistencies += 1
                    local_tracker_id = self.unassociated_obs[index].tracker_id
                group_by_id.append(self.unassociated_obs[index].tracker_id)
                
                # if this is an unassociated observation (not a currently tracked tracker)
                if index < len(self.unassociated_obs):
                    mean_xbar += self.unassociated_obs[index].xbar
                    group_size += 1
                    appearance_vecs += self.unassociated_obs[index].a
            if local_tracker_id == None:
                mean_xbar /= group_size
                local_tracker_id = (self.camera_id, self.next_available_id)
                new_tracker = Tracker(self.camera_id, self.next_available_id, mean_xbar[0:4,:], appearance_vecs)
                self.trackers.append(new_tracker)
                self.next_available_id += 1
                group_by_id.append(local_tracker_id)
            else:
                for tracker in self.trackers:
                    if tracker.id == local_tracker_id:
                        tracker.add_appearance_gallery(appearance_vecs)
                        break

            for index in group:
                if index >= len(self.unassociated_obs):
                    continue
                self.tracker_mapping[self.unassociated_obs[index].tracker_id] = local_tracker_id
            groups_by_id.append(group_by_id)
        
        return groups_by_id
        
    def manage_deletions(self):
        for tracker in self.old_trackers:
            tracker.cycle()
            if tracker.dead_cnt > TRACK_PARAM.n_recent_dets:
                self.old_trackers.remove(tracker)
        for tracker in self.trackers + self.new_trackers:
            tracker.cycle()
            if tracker.ell > self.kappa:
                self.trackers.remove(tracker)
                self.old_trackers.append(tracker)
                tracker.died()
            
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