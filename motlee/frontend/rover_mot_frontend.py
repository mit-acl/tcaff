import numpy as np
from numpy.linalg import inv
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from copy import deepcopy
from time import time

from motlee.frontend.person_detector import PersonDetector
from motlee.frontend.detections import GroundTruth
from motlee.mot.multi_object_tracker import MultiObjectTracker
from motlee.metrics.metric_evaluator import MetricEvaluator
from motlee.metrics.inconsistency_counter import InconsistencyCounter
from motlee.utils.transform import transform, T_mag
from motlee.utils.debug_help import *
from motlee.utils.cam_utils import pt_is_seeable
from motlee.config.track_params import TrackParams, ConeParams

T_MAG_STATIC_OBJ_REALIGN_THRESH = 2.5
SKIP_RR01_IN_VIEW = False
PROP_T_FOR_REALIGN_INIT_GUESS = False
CONE_R = np.diag([.5**2, .5**2])

class RoverMotFrontend():
    
    def __init__(self, detection_dirs, ped_dir, rovers, use_noisy_odom, cam_types, vids, noise, mot_params, register_time, metric_d, frame_aligner, vicon_cones=False, viewer=False):
        self.viewer = viewer
        self.GT = GroundTruth(csv_dir=ped_dir)
        self.vids = vids
        self.mots = []
        self.cone_trackers = []
        self.mes = [[]]
        self.full_mes = []
        self.detector = PersonDetector(
            csv_dirs=detection_dirs, 
            sigma_t=noise[0], 
            sigma_r=noise[1], 
            cams=rovers, 
            cam_types=cam_types, 
            use_noisy_odom=use_noisy_odom,
            register_time=register_time,
            vicon_cones=vicon_cones)
        self.num_rovers = len(rovers)
        self.cam_types = cam_types

        cone_tracker_params = deepcopy(mot_params)
        cone_tracker_params.Tau_LDA = cone_tracker_params.cone_Tau
        cone_tracker_params.merge_range_m = cone_tracker_params.cone_merge_range_m
        for i in range(self.num_rovers):
            self.cone_trackers.append(MultiObjectTracker(i, connected_cams=[], params=cone_tracker_params, track_params=ConeParams()))
        for i in range(2*self.num_rovers):
            connected_cams = [*range(2*self.num_rovers)]; connected_cams.remove(i)
            self.mots.append(MultiObjectTracker(i, connected_cams=connected_cams, params=mot_params, track_params=TrackParams(), 
                                                filter_frame_align=mot_params.filter_frame_align, frame_align_ts=mot_params.ts_realign))
            self.mes[0].append(MetricEvaluator(max_d=metric_d))
            self.full_mes.append(MetricEvaluator(max_d=metric_d))
        self.rover_names = rovers
        
        self.inconsistencies = 0
        self.ic = InconsistencyCounter()

        self.TRIGGER_AUTO_CYCLE_TIME = .2 # .1: .698158, .2: .695
        self.last_frame_time = 0
        self.frame_aligner = frame_aligner
        
        self.topview_size = 600
        self.topview = np.ones((self.topview_size,self.topview_size,3))*255
        self.topview = self.topview.astype(np.uint8)
        cv.rectangle(self.topview, 
            (int(-7*self.topview_size/20 + self.topview_size/2), int(-7*self.topview_size/20 + self.topview_size/2)), 
            (int(7*self.topview_size/20 + self.topview_size/2), int(7*self.topview_size/20 + self.topview_size/2)), 
            color=(0, 0, 0), thickness=4)
        cv.rectangle(self.topview, (int(self.topview_size/2), int(-7*self.topview_size/20 + self.topview_size/2)),
            (int(3*self.topview_size/20 + self.topview_size/2), int(-7*self.topview_size/20 + self.topview_size/2)),
            color=(255, 255, 255), thickness=4)
        self.frame_time = self.detector.start_time
        self.deg2m = mot_params.DEG_2_M
        self.T_bel_true = []
        for i in range(2*self.num_rovers):
            self.T_bel_true.append(np.eye(4))
        self.debug = []
        self.timing = {
            'tracking': [], 
            'fa': [],
            'mapping': []
        }
                        
    def frame_time_func(self, framenum):
        return framenum / 30 + self.detector.start_time
                        
    def update(self, framenum, run_metrics=True, realign=False):
        self.frame_time = self.frame_time_func(framenum)
        
        # Frame Realignment
        if realign:
            # for mot in self.mots: mot.frame_realign()
            for mot1a, mot1b, cones1 in zip(self.mots[self.num_rovers:], self.mots[:self.num_rovers], self.cone_trackers):
                start_time = time()
                for mot2a, mot2b, cones2 in zip(self.mots[self.num_rovers:], self.mots[:self.num_rovers], self.cone_trackers):
                    if mot1a == mot2a: continue
                    T_current = mot1a.realigner.transforms[mot2a.camera_id]
                    objs1 = np.array([o.state.reshape(-1)[:2] for o in cones1.tracks])
                    objs2 = np.array([o.state.reshape(-1)[:2] for o in cones2.tracks])
                    ages1 = np.array([o.ell for o in cones1.tracks])
                    ages2 = np.array([o.ell for o in cones2.tracks])
                    
                    objs_dyn1, objs_dyn2 = None, None #mot1b.get_common_detections(mot2b.camera_id)
                    if np.any(np.isinf(objs1)) or np.any(np.isinf(objs2)):
                        import ipdb; ipdb.set_trace()
                    
                    if PROP_T_FOR_REALIGN_INIT_GUESS:
                        T_guess = mot1a.realigner.get_transform_p1(mot2a.camera_id)
                    else:
                        T_guess = T_current
                    if objs_dyn1 is not None and objs_dyn2 is not None:
                        sol = self.frame_aligner.align_objects(
                            static_objects=(objs1, objs2),
                            static_ages=(ages1, ages2),
                            # dynamic_objects=(objs_dyn1[:,:2], objs_dyn2[:,:2]),
                            # dynamic_weights=(objs_dyn1[:,3]*objs_dyn2[:,3]+1.)**(-1),
                            T_init_guess=T_guess
                        )
                    else:
                        sol = self.frame_aligner.align_objects(
                            static_objects=(objs1, objs2),
                            static_ages=(ages1, ages2),
                            T_init_guess=T_guess
                        )
                        
                    if sol.success and mot1a.realigner.T_mag(sol.transform @ np.linalg.inv(T_current)) > T_MAG_STATIC_OBJ_REALIGN_THRESH: 
                        #and self.get_filtered_T_mag(T_new @ np.linalg.inv(T_current), 'psi') < 8:
                        sol.success = False
                    
                    mot1a.realigner.update_transform(mot2a.camera_id, sol)
                    mot1a.realigner.update_transform(mot2b.camera_id, sol)
                    mot1b.realigner.update_transform(mot2a.camera_id, sol)
                    mot1b.realigner.update_transform(mot2b.camera_id, sol)
                self.timing['fa'].append(time() - start_time)
 
        for i, cone_tracker in enumerate(self.cone_trackers):
            cones = self.detector.get_cones(i, 'l515', framenum, self.frame_time)
            for i in range(len(cones)):
                cones[i] = cones[i][:2,:]
            start_time = time()
            cone_tracker.local_data_association(cones, np.zeros(len(cones)), [CONE_R for cone in cones])    
            for cone in cone_tracker.tracks + cone_tracker.new_tracks:
                cone.correction()
                cone.cycle()
                if cone.ell > 600:
                    cone_tracker.tracks.remove(cone)
            self.timing['mapping'].append(time() - start_time)

        if framenum % 1 == 0:
            new_d = dump_everything_in_the_whole_world(self.frame_time, framenum, self.rover_names, self.mots[self.num_rovers:], self.detector.get_ordered_detections(['l515']), self.make_gt_list())
            # new_d = dump_single_rover_mapping_tracks(self.frame_time, framenum, self.rover_names[0], self.mots[self.num_rovers:][0], self.cone_trackers[0], self.detector.get_ordered_detections(['l515'])[0])
            # new_d = dump_mapping_info(self.frame_time, framenum, self.rover_names, self.mots[self.num_rovers:], self.detector.get_ordered_detections(['l515']))
            self.debug.append(new_d)
            
        # Continues to next frame when robots have new detections
        # if not detector.times_different(self.frame_time, last_frame_time) and abs(self.frame_time - last_frame_time) < TRIGGER_AUTO_CYCLE_TIME:
        if abs(self.frame_time - self.last_frame_time) < self.TRIGGER_AUTO_CYCLE_TIME:
            # for vid in self.vids:
            #     vid.read()
            return
        
        self.last_frame_time = self.frame_time

        observations = []
        start_time = time()
        for i, (vid1, vid2, mot1, mot2) in enumerate(zip(self.vids[:self.num_rovers], self.vids[self.num_rovers:], self.mots[:self.num_rovers], self.mots[self.num_rovers:])):

            # ret1, frame1 = vid1.read()
            # ret2, frame2 = vid2.read()

            # if not ret1 or not ret2:
            #     break

            # for j, (mot, frame) in enumerate(zip([mot1, mot2], [frame1, frame2])):
            for j, mot in enumerate([mot1, mot2]):
                mot.pose = self.detector.detections[i]['t265'].T_WC(self.frame_time, T_BC=np.eye(4), true_pose=False)
                for k in range(self.num_rovers):
                    neighbor_pose = self.detector.detections[k]['t265'].T_WC(self.frame_time, T_BC=np.eye(4), true_pose=False)
                    mot.neighbor_poses[k] = np.copy(neighbor_pose)
                    mot.neighbor_poses[k+self.num_rovers] = np.copy(neighbor_pose)
                positions, boxes, feature_vecs, Rs = self.detector.get_person_boxes(i, self.cam_types[j], self.frame_time)
                Zs = []
                for pos, box in zip(positions, boxes):
                    x0, y0, x1, y1 = box
                    Zs.append(np.array([[pos[0], pos[1]]]).T)
                    # if self.viewer:
                    #     cv.rectangle(frame, (int(x0),int(y0)), (int(x1),int(y1)), (0,255,0), 4)

                mot.local_data_association(Zs, feature_vecs, Rs)
                observations += mot.get_observations()  

                # if self.viewer:
                #     height, width, channels = frame.shape
                #     resized = cv.resize(frame, (int(width/3), int(height/3)))
                #     cv.imshow(f"frame{i}", resized)

        for mot in self.mots:
            mot.add_observations([obs for obs in observations if obs.destination == mot.camera_id])
            mot.dkf()
            mot.track_manager()
            self.ic.add_groups(mot.camera_id, mot.groups_by_id)
            mot.groups_by_id = []
            
        self.timing['tracking'].append((time() - start_time) / self.num_rovers)
        self.inconsistencies += self.ic.count_inconsistencies()

        combined = self.topview.copy()

        if self.viewer:
            for i, (mot, det) in enumerate(zip(self.mots, self.detector.get_ordered_detections(self.cam_types))):
            # for i, (mot, det) in enumerate(zip(self.mots[self.num_rovers:], self.detector.get_ordered_detections(self.cam_types)[self.num_rovers:])):

                Xs, colors = mot.get_tracks()   

                for X, color in zip(Xs, colors):
                    # correct state for local error
                    X_corrected = transform(det.T_WC(self.frame_time) @ np.linalg.inv(det.T_WC(self.frame_time, true_pose=False)), X[0:2,:])
                    x, y = X_corrected.reshape(-1).tolist()
                    w = X.item(2)
                    x, y = x*self.topview_size/20 + self.topview_size/2, y*self.topview_size/20 + self.topview_size/2
                    cv.circle(combined, (int(x),int(y)), int(2*w/3), color, 4)
        
        ped_ids, peds = self.GT.ped_positions(self.frame_time)
        gt_dict = dict()
        for ped_id, ped_pos in zip(ped_ids, peds):
            if not self.in_view(ped_pos):
                continue
            gt_dict[ped_id] = ped_pos[0:2]
            if self.viewer:
                pt = (int(ped_pos[0]*self.topview_size/20 + self.topview_size/2), int(ped_pos[1]*self.topview_size/20 + self.topview_size/2))
                cv.drawMarker(combined, pt, self.get_track_color(int(ped_id)), cv.MARKER_STAR, thickness=2, markerSize=30)
        
        if run_metrics:
            for mes in self.mes: 
                for mot, me, dets in zip(self.mots, mes, self.detector.get_ordered_detections(self.cam_types)):
                    me.update(gt_dict, mot.get_tracks(format='dict'), 
                            T_true=dets.T_WC(self.frame_time), 
                            T_bel=dets.T_WC(self.frame_time, true_pose=False))
            for mot, me, dets in zip(self.mots, self.full_mes, self.detector.get_ordered_detections(self.cam_types)):
                me.update(gt_dict, mot.get_tracks(format='dict'), 
                        T_true=dets.T_WC(self.frame_time), 
                        T_bel=dets.T_WC(self.frame_time, true_pose=False))

        if self.viewer:
            cv.imshow('topview', combined)
            cv.waitKey(5)

    def get_track_color(self, i):
        c = np.array(plt.get_cmap('tab10').colors[i])
        c = (c * 255).astype(int)
        return tuple(v.item() for v in c[::-1])
    
    def in_view(self, ped_pos):
        ped_pos[2] /= 2 # middle of person instead of top
        cx = 425.5404052734375
        cy = 400.3540954589844
        fx = 285.72650146484375
        fy = 285.77301025390625
        K = np.array([[fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]])
        D = np.array([-0.006605582777410746, 0.04240882024168968, -0.04068116843700409, 0.007674722000956535])
        width = 800
        height = 848
        
        rover_range_0 = 1 if SKIP_RR01_IN_VIEW else 0
        for rover in self.detector.get_ordered_detections(['t265'])[rover_range_0:]:
            if pt_is_seeable(K, rover.T_WC(self.frame_time), width, height, ped_pos):
                return True
        return False
    
    def T_mag(self):
        T_mag_avg = 0
        num = len(self.detector.get_ordered_detections(self.cam_types))
        for i, rover in enumerate(self.detector.get_ordered_detections(self.cam_types)):
            T_mag_avg += T_mag(rover.T_WC(self.frame_time, true_pose=False) @ inv(rover.T_WC(self.frame_time)) @ inv(self.T_bel_true[i]), self.deg2m) / num
        return T_mag_avg
    
    def inform_true_pairwise_T(self, framenum=None):
        if framenum is None:
            frame_time = self.frame_time
        else:
            frame_time = self.frame_time_func(framenum)
        for i, mot1 in enumerate(self.mots[self.num_rovers:]):
            for j, mot2 in enumerate(self.mots[self.num_rovers:]):
                if mot1 == mot2: continue
                T = self.detector.get_T_obj2_obj1(i, 'l515', j, 'l515', frame_time)
                if True:
                    mot1.realigner.transforms[mot2.camera_id] = T
                    mot1.realigner.transforms[j] = T
                    self.mots[i].realigner.transforms[mot2.camera_id] = T
                    self.mots[i].realigner.transforms[j] = T
                    T_WC_func = self.detector.detections[i]['l515'].T_WC
                    self.T_bel_true[i] = \
                        T_WC_func(frame_time, true_pose=False) @ inv(T_WC_func(frame_time))
                    self.T_bel_true[i + self.num_rovers] = \
                        T_WC_func(frame_time, true_pose=False) @ inv(T_WC_func(frame_time))
                        
    def calc_T_diff(self, filter=None):
        T_avg = 0.0
        for i, mot1 in enumerate(self.mots[self.num_rovers:]):
            T_avg_i = 0
            for j, mot2 in enumerate(self.mots[self.num_rovers:]):
                if mot1 == mot2: continue
                T = self.detector.get_T_obj2_obj1(i, 'l515', j, 'l515', self.frame_time)
                T_est = mot1.realigner.transforms[mot2.camera_id]
                T_avg_i += self.get_filtered_T_mag(inv(T) @ T_est, filter=filter) / (self.num_rovers - 1)
            T_avg += T_avg_i / (self.num_rovers)
        return T_avg
    
    def get_residual_and_cones(self):
        residuals, num_cones = [], []
        for i, mot1 in enumerate(self.mots[self.num_rovers:]):
            for j, mot2 in enumerate(self.mots[self.num_rovers:]):
                if mot1 == mot2: continue
                residuals.append(mot1.realigner.realign_residual[mot2.camera_id])
                num_cones.append(mot1.realigner.realign_num_cones[mot2.camera_id])
        return residuals, num_cones        
    
    def get_all_T_diffs(self, filter=None):
        T_diffs = []
        for i, mot1 in enumerate(self.mots[self.num_rovers:]):
            for j, mot2 in enumerate(self.mots[self.num_rovers:]):
                if mot1 == mot2: continue
                T = self.detector.get_T_obj2_obj1(i, 'l515', j, 'l515', self.frame_time)
                T_est = mot1.realigner.transforms[mot2.camera_id]
                T_diffs.append(self.get_filtered_T_mag(inv(T) @ T_est, filter=filter))
        return T_diffs

    def get_filtered_T_mag(self, T, filter=None):
        if not filter:
            mag = T_mag(T, self.deg2m)
        elif filter == 'psi':
            unwrapped = Rot.from_matrix(T[:3, :3]).as_euler('xyz', degrees=False).item(2)
            wrapped = (unwrapped + np.pi) % (2*np.pi) - np.pi
            mag = np.abs(wrapped) * 180 / np.pi
        elif filter == 't':
            mag = np.linalg.norm(T[:2,3])
        return mag

    def make_gt_list(self):
        ped_ids, peds = self.GT.ped_positions(self.frame_time)
        gt_list = []
        for ped_id, ped_pos in zip(ped_ids, peds):
            if not self.in_view(ped_pos):
                continue
            gt_list.append(ped_pos[0:2].tolist())
        return gt_list