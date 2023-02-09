import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from frontend.person_detector import PersonDetector
from frontend.detections import GroundTruth
from mot.multi_object_tracker import MultiObjectTracker
from metrics.metric_evaluator import MetricEvaluator
from metrics.inconsistency_counter import InconsistencyCounter
import config.rover_mot_params as PARAMS

class RoverMotFrontend():
    
    def __init__(self, detection_bag, ped_bag, realign_algorithm, rovers, rover_pose_topic, cam_type, vids, noise, viewer=False):
        self.viewer = viewer
        self.GT = GroundTruth(bagfile=ped_bag)
        self.realign_algorithm = realign_algorithm
        self.vids = vids
        self.mots = []
        self.mes = []
        self.detector = PersonDetector(bagfile=detection_bag, sigma_t=noise[0], sigma_r=noise[1], cams=rovers, cam_type=cam_type, cam_pose_topic=rover_pose_topic)
        for i, rover in enumerate(rovers):
            connected_cams = [*range(len(rovers))]; connected_cams.remove(i)
            self.mots.append(MultiObjectTracker(i, connected_cams=connected_cams, params=PARAMS))
            self.mes.append(MetricEvaluator())
        
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
            
    def update(self, framenum, run_metrics=True, realign=False):
        self.frame_time = framenum / 30 + self.detector.start_time
        
        # Frame Realignment
        if realign:
            for mot in self.mots: mot.frame_realign()
            
        # Continues to next frame when robots have new detections
        # if not detector.times_different(self.frame_time, last_frame_time) and abs(self.frame_time - last_frame_time) < TRIGGER_AUTO_CYCLE_TIME:
        if abs(self.frame_time - self.last_frame_time) < self.TRIGGER_AUTO_CYCLE_TIME:
            for vid in self.vids:
                vid.read()
            return
        
        self.last_frame_time = self.frame_time

        observations = []
        for i, (vid, mot) in enumerate(zip(self.vids, self.mots)):

            ret, frame = vid.read()

            if not ret:
                break

            Zs = []
            positions, boxes, feature_vecs = self.detector.get_person_boxes(frame, i, self.frame_time)
            for pos, box in zip(positions, boxes):
                x0, y0, x1, y1 = box
                Zs.append(np.array([[pos[0], pos[1], 20, 50]]).T)
                if self.viewer:
                    cv.rectangle(frame, (int(x0),int(y0)), (int(x1),int(y1)), (0,255,0), 4)

            mot.local_data_association(Zs, feature_vecs)
            observations += mot.get_observations()  

            if self.viewer:
                height, width, channels = frame.shape
                resized = cv.resize(frame, (int(width/3), int(height/3)))
                cv.imshow(f"frame{i}", resized)

        for mot in self.mots:
            mot.add_observations(observations)
            mot.dkf()
            mot.tracker_manager()
            self.ic.add_groups(mot.camera_id, mot.groups_by_id)
            mot.groups_by_id = []
        self.inconsistencies += self.ic.count_inconsistencies()

        combined = self.topview.copy()

        if self.viewer:
            for i, mot in enumerate(self.mots):

                Xs, colors = mot.get_trackers()   

                for X, color in zip(Xs, colors):
                    x, y, w, h, _, _ = X.reshape(-1).tolist()
                    x, y = x*self.topview_size/20 + self.topview_size/2, y*self.topview_size/20 + self.topview_size/2
                    cv.circle(combined, (int(x),int(y)), int(2*w/3), color, 4)
        
        ped_ids, peds = self.GT.ped_positions(self.frame_time)
        gt_dict = dict()
        for ped_id, ped_pos in zip(ped_ids, peds):
            gt_dict[ped_id] = ped_pos[0:2]
            if self.viewer:
                pt = (int(ped_pos[0]*self.topview_size/20 + self.topview_size/2), int(ped_pos[1]*self.topview_size/20 + self.topview_size/2))
                cv.drawMarker(combined, pt, self.get_track_color(int(ped_id)), cv.MARKER_STAR, thickness=2, markerSize=30)
        
        if run_metrics:    
            for mot, me, dets in zip(self.mots, self.mes, self.detector.detections):
                me.update(gt_dict, mot.get_trackers(format='dict'), 
                          T_true=dets.T_WC(self.frame_time), 
                          T_bel=dets.T_WC(self.frame_time, true_pose=False))

        if self.viewer:
            cv.imshow('topview', combined)
            cv.waitKey(5)

    def get_track_color(self, i):
        c = np.array(plt.get_cmap('tab10').colors[i])
        c = (c * 255).astype(int)
        return tuple(v.item() for v in c[::-1])