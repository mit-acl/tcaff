import numpy as np
from numpy.linalg import inv
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

from frontend.person_detector import PersonDetector
from frontend.detections import GroundTruth
from mot.multi_object_tracker import MultiObjectTracker
from metrics.metric_evaluator import MetricEvaluator
from metrics.inconsistency_counter import InconsistencyCounter
from realign.realign_frames import realign_cones, recently_weighted_realign
from utils.transform import transform, T_mag
from utils.debug_help import *
from utils.cam_utils import pt_is_seeable

T_MAG_STATIC_OBJ_REALIGN_THRESH = 5
SKIP_RR01_IN_VIEW = False
PROP_T_FOR_REALIGN_INIT_GUESS = False

class RoverMotFrontend():
    
    def __init__(self, detection_bags, ped_bag, rovers, rover_pose_topic, cam_types, vids, noise, mot_params, register_time, metric_d, vicon_cones=False, viewer=False):
        self.viewer = viewer
        self.GT = GroundTruth(bagfile=ped_bag)
        self.vids = vids
        self.mots = []
        self.mes = [[]]
        self.full_mes = []
        self.detector = PersonDetector(
            bagfiles=detection_bags, 
            sigma_t=noise[0], 
            sigma_r=noise[1], 
            cams=rovers, 
            cam_types=cam_types, 
            cam_pose_topic=rover_pose_topic, 
            register_time=register_time,
            vicon_cones=vicon_cones)
        self.num_rovers = len(rovers)
        self.cam_types = cam_types
        for i in range(2*self.num_rovers):
            connected_cams = [*range(2*self.num_rovers)]; connected_cams.remove(i)
            self.mots.append(MultiObjectTracker(i, connected_cams=connected_cams, params=mot_params))
            self.mes[0].append(MetricEvaluator(max_d=metric_d))
            self.full_mes.append(MetricEvaluator(max_d=metric_d))
        self.rover_names = rovers
        
        self.inconsistencies = 0
        self.ic = InconsistencyCounter()

        self.TRIGGER_AUTO_CYCLE_TIME = .2 # .1: .698158, .2: .695
        self.last_frame_time = 0
        
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
                        
    def frame_time_func(self, framenum):
        return framenum / 30 + self.detector.start_time
                        
    def update(self, framenum, run_metrics=True, realign=False):
        self.frame_time = self.frame_time_func(framenum)
        
        # Frame Realignment
        if realign:
            # for mot in self.mots: mot.frame_realign()
            for i, mot1 in enumerate(self.mots[self.num_rovers:]):
                for j, mot2 in enumerate(self.mots[self.num_rovers:]):
                    if mot1 == mot2: continue
                    T_current = mot1.realigner.transforms[mot2.camera_id]
                    if PROP_T_FOR_REALIGN_INIT_GUESS:
                        T_guess = mot1.realigner.get_transform_p1(mot2.camera_id)
                        T_new = realign_cones(mot1.cones, mot2.cones, T_guess)
                    else:
                        T_new = realign_cones(mot1.cones, mot2.cones, T_current)
                    if mot1.realigner.T_mag(T_new @ np.linalg.inv(T_current)) < T_MAG_STATIC_OBJ_REALIGN_THRESH: #and self.get_filtered_T_mag(T_new @ np.linalg.inv(T_current), 'psi') < 8:
                        mot1.realigner.update_transform(mot2.camera_id, T_new)
                        mot1.realigner.update_transform(j, T_new)
                        self.mots[i].realigner.update_transform(mot2.camera_id, T_new)
                        self.mots[i].realigner.update_transform(j, T_new)
            # for mot in self.mots:
            #     mot.cones = []
            #     mot.new_cones = []
            
        # if realign and self.frame_time > self.detector.start_time + 100:
        #     print('\tcones = [[], [], [], []]')
        #     print('\tTs = [[], [], [], []]')
        #     print('\tTs_true = [[], [], [], []]')
        #     print('\tTs_error = [[], [], [], []]')
        #     for i, mot in enumerate(self.mots[self.num_rovers:]):
        #         print(f'\tcones[{i}] = np.array([')
        #         for cone in mot.cones:                
        #             print(f'\t[{cone.state.item(0)}, {cone.state.item(1)}],')
        #         # for cone in self.detector.raw_cone_detections[i]:
        #         #     print(f'\t[{cone[0]}, {cone[1]}],')
        #         print('\t])')
        #         print(f'\tTs[{i}] = [')
        #         for mot2 in self.mots[self.num_rovers:]:
        #             if i == mot2.camera_id or mot.camera_id == mot2.camera_id: continue
        #             print(f'#{mot2.camera_id}')
        #             print(f'\tnp.array({mot.realigner.transforms[mot2.camera_id].tolist()}), ')
        #         print('\t]')
        #     self.inform_true_pairwise_T()
        #     for i, mot in enumerate(self.mots[self.num_rovers:]):
        #         print(f'\tTs_true[{i}] = [')
        #         for mot2 in self.mots[self.num_rovers:]:
        #             if i == mot2.camera_id or mot.camera_id == mot2.camera_id: continue
        #             print(f'#{mot2.camera_id}')
        #             print(f'\tnp.array({mot.realigner.transforms[mot2.camera_id].tolist()}), ')
        #         print('\t]')
        #         T = self.detector.get_T_error(i, 'l515', self.frame_time)
        #         print(f'\tTs_error[{i}] = [{T.tolist()}]')
            
        #     print('\treturn cones, Ts, Ts_true')
        #     exit(0)
            
        for i, mot in enumerate(self.mots[self.num_rovers:]):
            cones = self.detector.get_cones(i, 'l515', framenum, self.frame_time)
            # if framenum % 30 == 0:
            #     new_d = get_cone_debug_dict(self.frame_time, cones, self.detector.get_ordered_detections(['l515'])[0])
            #     self.debug.append(new_d)
            mot.cone_update(cones)    
        if framenum % 1 == 0:
            # new_d = get_realign_debug_dict(self.frame_time, self.mots[2], self.mots[3], 
            #                                self.detector.get_ordered_detections(['l515'])[0], 
            #                                self.detector.get_ordered_detections(['l515'])[1],
            #                                self.detector.get_T_obj2_obj1(0, 'l515', 1, 'l515', self.frame_time))
                # new_d = get_map_debug_dict(self.frame_time, mot, self.detector.get_ordered_detections(['l515'])[0])
                # self.debug.append(new_d) 
        
            # more advanced heh
            # new_l = []
            # for j in range(self.num_rovers):
            #     for i in range(self.num_rovers):
            #         if i == j: continue
            #         new_l.append(get_realign_debug_dict(self.frame_time, self.mots[self.num_rovers+j], self.mots[self.num_rovers+i], 
            #                                self.detector.get_ordered_detections(['l515'])[j], 
            #                                self.detector.get_ordered_detections(['l515'])[i],
            #                                self.detector.get_T_obj2_obj1(j, 'l515', i, 'l515', self.frame_time)))
            # self.debug.append(new_l)
            
            # even more advanced
            new_d = dump_everything_in_the_whole_world(self.frame_time, framenum, self.rover_names, self.mots[self.num_rovers:], self.detector.get_ordered_detections(['l515']), self.make_gt_list())
            # new_d = dump_mapping_info(self.frame_time, framenum, self.rover_names, self.mots[self.num_rovers:], self.detector.get_ordered_detections(['l515']))
            self.debug.append(new_d)
            
        # Continues to next frame when robots have new detections
        # if not detector.times_different(self.frame_time, last_frame_time) and abs(self.frame_time - last_frame_time) < TRIGGER_AUTO_CYCLE_TIME:
        if abs(self.frame_time - self.last_frame_time) < self.TRIGGER_AUTO_CYCLE_TIME:
            for vid in self.vids:
                vid.read()
            return
        
        self.last_frame_time = self.frame_time

        observations = []
        for i, (vid1, vid2, mot1, mot2) in enumerate(zip(self.vids[:self.num_rovers], self.vids[self.num_rovers:], self.mots[:self.num_rovers], self.mots[self.num_rovers:])):

            ret1, frame1 = vid1.read()
            ret2, frame2 = vid2.read()

            if not ret1 or not ret2:
                break

            for j, (mot, frame) in enumerate(zip([mot1, mot2], [frame1, frame2])):
                mot.pose = self.detector.detections[i]['t265'].T_WC(self.frame_time, T_BC=np.eye(4), true_pose=False)
                positions, boxes, feature_vecs = self.detector.get_person_boxes(frame, i, self.cam_types[j], self.frame_time)
                Zs = []
                for pos, box in zip(positions, boxes):
                    x0, y0, x1, y1 = box
                    Zs.append(np.array([[pos[0], pos[1]]]).T)
                    if self.viewer:
                        cv.rectangle(frame, (int(x0),int(y0)), (int(x1),int(y1)), (0,255,0), 4)

                mot.local_data_association(Zs, feature_vecs)
                observations += mot.get_observations()  

                if self.viewer:
                    height, width, channels = frame.shape
                    resized = cv.resize(frame, (int(width/3), int(height/3)))
                    cv.imshow(f"frame{i}", resized)

        for mot in self.mots:
            mot.add_observations([obs for obs in observations if obs.destination == mot.camera_id])
            mot.dkf()
            mot.track_manager()
            self.ic.add_groups(mot.camera_id, mot.groups_by_id)
            mot.groups_by_id = []
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