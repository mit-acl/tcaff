import numpy as np
from numpy.linalg import norm as norm
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

from tracker import Tracker
import config.tracker_params as TRACK_PARAM

class Camera():

    def __init__(self, camera_id, Tau_LDA, Tau_GDA, alpha, kappa, T, n_meas_init=2):
        # TODO: Tune for Tau
        self.Tau_LDA = Tau_LDA
        self.Tau_GDA = Tau_GDA
        self.alpha = alpha
        self.kappa = kappa
        self.trackers = []
        self.new_trackers = []
        self.next_available_id = 0
        self.n_meas_to_init_tracker = n_meas_init
        self.camera_id = camera_id
        self.tracker_mapping = dict()
        self.unassociated_obs = []
        self.inconsistencies = 0
        self.groups_by_id = []
        self.T_local_global = T
        self.T_other = dict()
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
            observations.append(tracker.observation_msg())
        return observations

    def add_observations(self, observations, transform=True):
        # TODO: Probably not the best memory usage here...
        observations = deepcopy(observations)
        self.unassociated_obs = []
        # Process each observation
        for obs in observations:
            if transform:
                obs = self._transform_obs(obs)
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
        
    def get_recent_detections(self, from_trackers=False):
        if from_trackers:
            recent_detections = []
            for tracker in self.trackers:
                recent_detections.append(tracker.recent_detections)
            return recent_detections
        else: # raw detections
            return self.recent_detection_list
        
    def frame_realign(self):
        def detection_pairs_2_ordered_arrays(detection_list1, detection_list2):
            ordered_pairs = [np.array([]).reshape(0,3), np.array([]).reshape(0,3)]
            for track1, track2 in zip(detection_list1, detection_list2):
                if track1.shape != track2.shape:
                    continue
                for i in range(track1.shape[0]):
                    if np.isnan(track1[i,:]).any() or np.isnan(track2[i,:]).any():
                        continue
                    else:
                        ordered_pairs[0] = np.concatenate([ordered_pairs[0], track1[i,:].reshape(-1,3)], axis=0)
                        ordered_pairs[1] = np.concatenate([ordered_pairs[1], track2[i,:].reshape(-1,3)], axis=0)
            return ordered_pairs
        
        def calc_T(det1, det2, add_weighting=False):    
            if add_weighting:
                mean_pts = (det1 + det2) / 2.0
                det_dist_from_mean = np.linalg.norm(det1 - mean_pts, axis=1)
                weight_vals = 1 / (.01 + det_dist_from_mean)
                weight_vals = weight_vals.reshape((-1,1))
                det1 * weight_vals
                np.sum(det1 * weight_vals, axis=0)
                mean1 = (np.sum(det1 * weight_vals, axis=0) / np.sum(weight_vals)).reshape(-1)
                mean2 = (np.sum(det2 * weight_vals, axis=0) / np.sum(weight_vals)).reshape(-1)
                det1_mean_reduced = det1 - mean1
                det2_mean_reduced = det2 - mean2
                assert det1_mean_reduced.shape == det2_mean_reduced.shape
                H = det1_mean_reduced.T @ (det2_mean_reduced * weight_vals)
                U, s, V = np.linalg.svd(H)
                R = U @ V.T
                t = mean1.reshape((3,1)) - R @ mean2.reshape((3,1))
                T = np.concatenate([np.concatenate([R, t], axis=1), np.array([[0,0,0,1]])], axis=0)
            else:
                mean1 = np.mean(det1, axis=0).reshape(-1)
                mean2 = np.mean(det2, axis=0).reshape(-1)
                det1_mean_reduced = det1 - mean1
                det2_mean_reduced = det2 - mean2
                assert det1_mean_reduced.shape == det2_mean_reduced.shape
                H = det1_mean_reduced.T @ det2_mean_reduced
                U, s, V = np.linalg.svd(H)
                R = U @ V.T
                t = mean1.reshape((3,1)) - R @ mean2.reshape((3,1))
                T = np.concatenate([np.concatenate([R, t], axis=1), np.array([[0,0,0,1]])], axis=0)
            return T
        
        local_dets = []
        for tracker in self.trackers:
            if self.camera_id in tracker.recent_detections:
                local_dets.append(tracker.recent_detections[self.camera_id])
            else:
                local_dets.append(np.empty((TRACK_PARAM.n_recent_dets, 3)) * np.nan)
        for cam_id in self.T_other:
            if cam_id == self.camera_id: continue
            other_dets = []
            for tracker in self.trackers:
                if cam_id in tracker.recent_detections:
                    other_dets.append(tracker.recent_detections[cam_id])
                else:
                    other_dets.append(np.empty((TRACK_PARAM.n_recent_dets, 3)) * np.nan)
            dets1, dets2 = detection_pairs_2_ordered_arrays(local_dets, other_dets)
            T_new = calc_T(dets1, dets2)
            if np.isnan(T_new).any():
                T_new = np.eye(4)
            self.T_other[cam_id] = T_new @ self.T_other[cam_id]
            # print(self.T_other[cam_id])
            

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
                # new_tracker.include_appearance_in_obs()
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
    
    def _transform_obs(self, obs):
        # TODO: Right now I am assuming H and R are the same in each tracker!
        obs_cam = obs.tracker_id[0]
        if obs_cam not in self.T_other:
            return obs
        if obs_cam == self.camera_id:
            return obs
        
        # Extract z and correct
        # TODO: Assuming H is identity matrix (or 1s along diag)
        if obs.u is not None:
            z = TRACK_PARAM.R @ obs.u[0:4,:]
            p_meas = np.concatenate([z[0:2,:], [[0.], [1.]]], axis=0)
            p_meas_corrected = (self.T_other[obs_cam] @ p_meas)[0:2,:]
            obs.u = TRACK_PARAM.H.T @ np.linalg.inv(TRACK_PARAM.R) @ np.concatenate([p_meas_corrected, z[2:]], axis=0)
            
        # Extract xbar and correct
        pos = np.concatenate([obs.xbar[0:2,:], [[0], [1]]], axis=0)
        vel = np.concatenate([obs.xbar[4:6,:], [[0]]], axis=0)
        pos_corrected = self.T_other[obs_cam] @ pos
        vel_corrected = self.T_other[obs_cam][0:3, 0:3] @ vel
        obs.xbar[0:2,:] = pos_corrected[0:2,:]
        obs.xbar[4:6,:] = vel_corrected[0:2,:]
        return obs
        
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