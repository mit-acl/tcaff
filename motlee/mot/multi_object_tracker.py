import numpy as np
from numpy.linalg import norm as norm, inv
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as Rot
from scipy.linalg import logm
from copy import deepcopy

from motlee.mot.track import Track
from motlee.realign.frame_align_filter import FrameAlignFilter
from motlee.utils.transform import transform

NUM_CAMS = 4
COV_MAG = .01
USE_NLML = True

class MultiObjectTracker():

    def __init__(self, camera_id, connected_cams, params, track_params, filter_frame_align=False, frame_align_ts=None):
        self.connected_cams = connected_cams
        self.realigner = FrameAlignFilter(
            cam_id=camera_id,
            connected_cams=connected_cams,
            filter_frame_align=filter_frame_align,
            ts=frame_align_ts
        )
        self.Tau_init = params.Tau_LDA
        self.Tau_LDA = params.Tau_LDA
        self.Tau_GDA = params.Tau_GDA
        self.Tau_grown = 1
        
        self.alpha = params.alpha
        self.kappa = params.kappa
        self.tracks = []
        self.new_tracks = []
        self.old_tracks = []
        self.next_available_id = 0
        self.n_meas_to_init_track = params.n_meas_to_init_track
        self.merge_range_m = params.merge_range_m
        
        self.camera_id = camera_id
        self.track_mapping = dict()
        self.unassociated_obs = []
        self.inconsistencies = 0
        self.groups_by_id = []
        self.recent_detection_list = []
        self.cov = np.eye(4) * COV_MAG
        for i in range(50):
            self.recent_detection_list.append(None)
            
        self.pose = None
        self.MDs = []
        self.track_params = track_params
        

    def local_data_association(self, Zs, feature_vecs, Rs):
        self.MDs = []
        for track in self.tracks + self.new_tracks:
            track.predict()
        
        # merge any multiple detections within some range
        need_to_merge = True
        while need_to_merge:
            need_to_merge = False
            for i in range(len(Zs)):
                for j in range(len(Zs)):
                    if i == j: continue
                    if norm(Zs[i][:2,:] - Zs[j][:2,:]) < self.merge_range_m:
                        need_to_merge = True
                        break
                if need_to_merge:
                    break
            if need_to_merge:
                Zs[i] = (Zs[i] + Zs[j]) / 2
                del Zs[j]
        
        frame_detections = []
        for z in Zs:
            frame_detections.append(z[0:2,:].reshape(-1,1))
        all_tracks = self.tracks + self.new_tracks
        geometry_scores = np.zeros((len(all_tracks), len(Zs)))
        large_num = 1000
        for i, track in enumerate(all_tracks):
            Hx_xy = (track.H @ track.xbar)[0:2,:]
            for j, (Z, aj, R) in enumerate(zip(Zs, feature_vecs, Rs)):
                z_xy = Z[0:2,:]
                V = track.P[:2,:2] + R
                # Mahalanobis distance
                d = np.sqrt((z_xy - Hx_xy).T @ np.linalg.inv(V) @ (z_xy - Hx_xy)).item(0)
                self.MDs.append(d)
                
                # Geometry similarity value
                if not d < self.Tau_LDA:
                    s_d = large_num
                elif USE_NLML:
                    s_d = 1/self.alpha*(2*np.log(2*np.pi) + d**2 + np.log(np.linalg.det(V)))
                else:
                    s_d = 1/self.alpha*d
                if np.isnan(s_d):
                    s_d = large_num
                geometry_scores[i,j] = s_d

        unassociated = []
        # augment cost to add option for no associations
        hungarian_cost = np.concatenate([
            np.concatenate([geometry_scores, np.ones(geometry_scores.shape)], axis=1),
            np.ones((geometry_scores.shape[0], 2*geometry_scores.shape[1]))], axis=0)
        row_ind, col_ind = linear_sum_assignment(hungarian_cost)
        for t_idx, z_idx in zip(row_ind, col_ind):
            # track and measurement associated together
            if t_idx < len(all_tracks) and z_idx < len(Zs):
                assert geometry_scores[t_idx,z_idx] < 1
                all_tracks[t_idx].update([Zs[z_idx]], Rs[z_idx])       
            # unassociated measurement
            elif z_idx < len(Zs):
                unassociated.append(z_idx)
            # unassociated track or augmented part of matrix
            else:
                continue
            
        # if there are no tracks, hungarian matrix will be empty, handle separately
        if len(all_tracks) == 0:
            for z_idx in range(len(Zs)):
                unassociated.append(z_idx)

        for track in self.new_tracks:
            if track.frames_seen >= self.n_meas_to_init_track:
                self.tracks.append(track)
                self.track_mapping[track.id] = track.id
                self.new_tracks.remove(track)
            elif not track.seen:
                self.new_tracks.remove(track)

        for z_idx in unassociated:
            # zero velocity initial state
            if self.track_params.A.shape[0] == Zs[z_idx].shape[0]:
                init_state = Zs[z_idx]
            else:
                init_state = np.vstack([Zs[z_idx], np.zeros((self.track_params.A.shape[0] - Zs[z_idx].shape[0],1))])
            new_track = Track(self.camera_id, self.next_available_id, self.track_params, [Zs[z_idx]], init_state)
            R_shape = Rs[z_idx].shape[0]
            new_track.P[:R_shape,:R_shape] = Rs[z_idx]
            new_track.update([Zs[z_idx]], Rs[z_idx])
            self.new_tracks.append(new_track)
            self.next_available_id += 1

        for track in self.tracks + self.new_tracks:
            track.predict()

    def dkf(self):  
        for track in self.tracks + self.new_tracks:
            track.correction()

    def track_manager(self):
        # no initialization necessary if no unassociated observations
        if not self.unassociated_obs:
            self.manage_deletions()
            return
        tracked_states = []
        for track in self.tracks + self.new_tracks:
            m = track.get_measurement_info(track.R)
            m.xbar = track.state
            tracked_states.append(m)
        geometry_scores = np.zeros((len(self.unassociated_obs), len(self.unassociated_obs + tracked_states)))
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
                
            for j in range(len(tracked_states)):
                j_gs = j+len(self.unassociated_obs) # j for indexing into geometry_scores
                xj_xy = tracked_states[j].xbar[0:2,:]
                # distance
                d = np.sqrt((xi_xy - xj_xy).T @ (xi_xy - xj_xy)).item(0)
                
                # Geometry similarity value
                s_d = 1/self.alpha*d if d < self.Tau_GDA else 1
                geometry_scores[i,j_gs] = s_d

        track_groups = self._get_track_groups(geometry_scores)
        
        self.groups_by_id = self._create_tracks(track_groups, tracked_states)
        unassociated_obs = self.unassociated_obs
        len_unassociated_obs = self.add_observations(unassociated_obs)
            # transform has already occured at this point
        assert len_unassociated_obs == 0
        
        self.manage_deletions()
        return
                
    def get_observations(self):
        observations = []
        for track in self.tracks:   
            track_obs = track.get_measurement_info(track.R)
            for cam in self.connected_cams:
                T_fa = inv(self.realigner.transforms[cam] @ inv(self.realigner.T_last[cam]))

                R = np.copy(track.R)
                if track_obs.zs is not None:
                    # This needs to be fixed so that it is correct when being sent to other camera (rather than
                    # being written for being received)
                    # z_b = transform(inv(self.pose), track_obs.zs[0].reshape(-1)[:2])
                    z = track_obs.zs[0].reshape(-1)[:2]
                    Jac_T = np.array([
                        [1.0, 0.0, -z.item(1)], # should be expressed in body frame, not world frame
                        [0.0, 1.0, z.item(0)], # 
                    ])
                    # th_fa = Rot.from_matrix(T_fa[:3,:3]).as_euler('xyz', degrees=False)[2]
                    # Jac_T = np.array([
                    #     [1.0, 0.0, -obs.z.item(0)*np.sin(th_fa) - y],
                    #     [0.0, 1.0, obs.z.item(0)],
                    #     [0.0, 0.0, 0.0],
                    #     [0.0, 0.0, 0.0]
                    # ])
                    Jac_R = self.realigner.transforms[cam][:2,:2].T
                    Sigma_fa = np.zeros((3,3))
                    Sigma_fa[0, 0] = T_fa[0, 3]*(self.realigner.realigns_since_change[cam]+1) # for now, just the mag of x and y change
                    Sigma_fa[1, 1] = T_fa[1, 3]*(self.realigner.realigns_since_change[cam]+1) # for now, just the mag of x and y change
                    Sigma_fa[2, 2] = Rot.from_matrix(T_fa[:3,:3]).as_euler('xyz', degrees=False)[2] * (self.realigner.realigns_since_change[cam]+1) *.1 #* 8.1712
                    if self.realigner.filter_frame_align:
                        Sigma_fa = self.realigner.transform_covs[cam][:3, :3]
                    R = Jac_T @ (Sigma_fa) @ Jac_T.T + Jac_R @ R @ Jac_R.T
                    R = (R.T + R) / 2 # keep PD
                if cam in self.realigner.transforms:
                    T = inv(self.realigner.transforms[cam])
                else:
                    T = np.eye(4)
                obs = track.get_measurement_info(R, T=T) # TODO: figure out R
                obs.add_destination(cam)
                observations.append(obs)
        return observations

    def add_observations(self, observations):
        # TODO: Probably not the best memory usage here...
        observations = deepcopy(observations)
        self.unassociated_obs = []
        # Process each observation
        for obs in observations:
            assert obs.destination == self.camera_id
            # Check if we recognize the track id
            if obs.track_id in self.track_mapping: 
                target_track_id = self.track_mapping[obs.track_id]
                for track in self.tracks:
                    if track.id == target_track_id:
                        track.add_measurement(obs)
                        break
            else:
                matched = False
                # Check if the incoming message has already been paired to one of our tracks
                for mid in obs.mapped_ids:
                    if mid in self.track_mapping:
                        matched = True
                        target_track_id = self.track_mapping[mid]
                        self.track_mapping[obs.track_id] = target_track_id
                        for track in self.tracks:
                            if track.id == target_track_id:
                                track.add_measurement(obs)
                                break
                        break
                # Add to unassociated_obs for track initialization if this is new
                if not matched:                        
                    self.unassociated_obs.append(obs)
        return len(self.unassociated_obs)
            
    def get_tracks(self, format='state_color'):
        # TODO: change what we're returning here
        if format == 'state_color':
            Xs = []
            colors = []
            for track in self.tracks:
                assert track.id[0] == self.camera_id
                Xs.append(track.state)
                colors.append(track.color)
            return Xs, colors
        elif format == 'dict':
            track_dict = dict()
            for track in self.tracks:
                track_dict[track.id] = np.array(track.state[0:2,:].reshape(-1))
            return track_dict
        elif format == 'list':
            track_list = []
            for track in self.tracks:
                track_list.append(track.state[0:2,:].reshape(-1).tolist())
            return track_list
        else:
            print('you cannot do that')
            assert False
        
    def get_recent_detections(self, from_tracks=False):
        if from_tracks:
            recent_detections = []
            for track in self.tracks:
                recent_detections.append(track.recent_detections)
            return recent_detections
        else: # raw detections
            return self.recent_detection_list
        
    def frame_realign(self):
        self.realigner.realign(self.tracks + self.old_tracks)
        self.realigner.rectify_detections(self.tracks + self.old_tracks)
        self.Tau_LDA = self.realigner.tolerance_scale * self.Tau_init

    def _get_track_groups(self, similarity_scores):
        # TODO: Maximum clique kind of thing going on here...
        # Handle better

        track_groups = []
        new_track_groups = []
        for i in range(similarity_scores.shape[0]):
            group = set()
            for j in range(similarity_scores.shape[0]):
                if similarity_scores[i,j] < .1: # TODO: Magic number here?
                    group.add(j)
            track_groups.append(group)

        while track_groups: 
            g1 = track_groups.pop(0)
            added = []
            for member in g1:
                for g2 in new_track_groups:
                    if member in g2:
                        new_track_groups.remove(g2)
                        g2 = g2.union(g1)
                        new_track_groups.append(g2)
                        added.append(g2)
            if not added:
                new_track_groups.append(g1)
            elif len(added) == 1:
                continue
            else:
                for g in added[1:]:
                    new_track_groups.remove(added[0])
                    if g in new_track_groups:
                        new_track_groups.remove(g)
                    added[0] = added[0].union(g)
                    new_track_groups.append(added[0])
        
        # # associate with local indexes
        STOP_INCONSISTENCIES = True
        # TODO: This is a simple hack. It improves MOT performance, but can also lead to hackish results (gruops with common elements)
        if STOP_INCONSISTENCIES:
            if similarity_scores.shape[0] != similarity_scores.shape[1]:
                for group in new_track_groups:
                    for i in group:
                        local_idx = np.argmin(similarity_scores[i,similarity_scores.shape[0]:similarity_scores.shape[1]])
                        local_idx += similarity_scores.shape[0]
                        if similarity_scores[i,local_idx] < 1:
                            group.add(local_idx)
                            break
            # need_merging = []
            # for group in new_track_groups:
            #     for other_group in new_track_groups:
            #         if group == other_group: continue
            #         for el in group:
            #             if el in other_group:
            #                 need_merging.append((group, other_group))
            #                 break
            # for pair in need_merging:
            #     if pair[0] in new_track_groups and pair[1] in new_track_groups:
            #         new_track_groups.remove(pair[0]); new_track_groups.remove(pair[1])
            #         new_track_groups.append(pair[0].union(pair[1]))    
        else:
            if similarity_scores.shape[0] != similarity_scores.shape[1]:
                for group in new_track_groups:
                    to_add = set()
                    for i in group:
                        local_idx = np.argmin(similarity_scores[i,similarity_scores.shape[0]:similarity_scores.shape[1]])
                        local_idx += similarity_scores.shape[0]
                        if similarity_scores[i,local_idx] < .1:
                            to_add.add(local_idx)
                            # break
                    group = group.union(to_add)
        
        return new_track_groups
    
    def _create_tracks(self, track_groups, tracked_states):
        groups_by_id = list()
        for group in track_groups:
            group_by_id = list()
            local_track_id = None
            mean_xbar = np.zeros((4,1))
            group_size = 0
            
            # Assign local track id
            for index in group:
                if index >= len(self.unassociated_obs):
                    # If associated with a currently "new" track
                    if index >= len(self.unassociated_obs) + len(self.tracks):
                        t = self.new_tracks[index-(len(self.unassociated_obs)+len(self.tracks))]
                        self.tracks.append(t)
                        self.track_mapping[t.id] = t.id
                        self.new_tracks.remove(t)
                    # assert local_track_id == None or local_track_id == tracked_states[index-len(self.unassociated_obs)].track_id, \
                    #      f'1: {local_track_id}, 2: {tracked_states[index-len(self.unassociated_obs)].track_id}'
                    if local_track_id != None and local_track_id != tracked_states[index-len(self.unassociated_obs)].track_id:
                        # print('yeah it happened')
                        self.inconsistencies += 1
                    local_track_id = tracked_states[index-len(self.unassociated_obs)].track_id
                    group_by_id.append(tracked_states[index-len(self.unassociated_obs)].track_id)
                    continue
                elif self.unassociated_obs[index].track_id[0] == self.camera_id:
                    # assert local_track_id == None or local_track_id == self.unassociated_obs[index].track_id, \
                    #     f'1: {local_track_id}, 2: {self.unassociated_obs[index].track_id}'
                    if local_track_id != None and local_track_id != self.unassociated_obs[index].track_id:
                        # print('yeah it happened')
                        self.inconsistencies += 1
                    local_track_id = self.unassociated_obs[index].track_id
                group_by_id.append(self.unassociated_obs[index].track_id)
                
                # if this is an unassociated observation (not a currently tracked track)
                if index < len(self.unassociated_obs):
                    mean_xbar += self.unassociated_obs[index].xbar
                    group_size += 1
            if local_track_id == None:
                mean_xbar /= group_size
                local_track_id = (self.camera_id, self.next_available_id)
                new_track = Track(self.camera_id, self.next_available_id, self.track_params, [mean_xbar[0:4,:]], mean_xbar)                
                self.tracks.append(new_track)
                self.next_available_id += 1
                group_by_id.append(local_track_id)

            for index in group:
                if index >= len(self.unassociated_obs):
                    continue
                self.track_mapping[self.unassociated_obs[index].track_id] = local_track_id
            groups_by_id.append(group_by_id)
        
        return groups_by_id
        
    def manage_deletions(self):
        for track in self.old_tracks:
            track.cycle()
            if track.dead_cnt > self.track_params.n_dets:
                self.old_tracks.remove(track)
        for track in self.tracks + self.new_tracks:
            track.cycle()
            if track.ell > self.kappa:
                self.tracks.remove(track)
                self.old_tracks.append(track)
                track.died()
            
    def __str__(self):
        return_str = ''
        return_str += f'Camera {self.camera_id}\n'
        if self.tracks:
            return_str += 'Tracks:\n'
            for t in self.tracks:
                return_str += f'\t{t}\n'
        if self.new_tracks:
            return_str += 'New Tracks:\n'
            for t in self.new_tracks:
                return_str += f'\t{t}\n'
        if self.track_mapping:
            return_str += 'Mappings:\n'
            for global_t, local_t in self.track_mapping.items():
                return_str += f'\t{global_t} --> {local_t}\n'
        # return_str += ''
        return return_str