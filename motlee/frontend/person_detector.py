# import torchreid feature extractor
# from torchreid.utils import FeatureExtractor
# if __name__ == '__main__':
#     from detections import get_rover_detections, get_cone_detections
# else:
#     from .detections import get_rover_detections, get_cone_detections
from motlee.frontend.detections import get_rover_detections, get_cone_detections
import numpy as np
from numpy.linalg import inv, norm
from scipy.spatial.transform import Rotation as Rot

class PersonDetector():

    def __init__(self, csv_dirs, sigma_r=0, sigma_t=0, 
                 cams=['RR01', 'RR04', 'RR05', 'RR06', 'RR08'], cam_types=['t265'], 
                 use_noisy_odom=False, register_time=16, vicon_cones=False):
        self.detections = []
        for cam in cams:
            self.detections.append(dict())
            for cam_type, csv_dir in zip(cam_types, csv_dirs):
                self.detections[-1][cam_type] = get_rover_detections(csv_dir=csv_dir, register_time=register_time,
                    sigma_r=sigma_r, sigma_t=sigma_t, 
                    rovers=[cam], cam_type=cam_type, use_noisy_odom=use_noisy_odom)[0]


        self.cone_detections = get_cone_detections(rovers=cams, vicon=vicon_cones)
        self.x_max, self.y_max = dict(), dict()
        self.x_max['l515'] = 1920
        self.y_max['l515'] = 1080
        self.x_max['t265'] = 800
        self.y_max['t265'] = 848
        self.start_time = self.detections[0][cam_types[0]].time(0)
        self.num_cams = len(cams)
        self.raw_cone_detections = [[], [], [], []]

    def get_person_boxes(self, cam_num, cam_type, frame_time):
        positions = []
        boxes = []
        features = []
        Rs = []
        T_WC = self.detections[cam_num][cam_type].T_WC(frame_time, T_BC=self.detections[cam_num][cam_type].T_BC, true_pose=False)
        for b, p in zip(self.detections[cam_num][cam_type].bbox(frame_time), self.detections[cam_num][cam_type].pos(frame_time)):
            theta = np.arctan2(p.item(1) - T_WC[1,3], p.item(0) - T_WC[0,3])
            rot = Rot.from_euler('xyz', [0., 0., theta]).as_matrix()[:2,:2]
            dist = norm(p.reshape(-1)[:2] - T_WC[:2,3])
            sigma_sq_depth = .5 + (dist > 5) * .1*(dist - 5)
            sigma_sq_breadth = .25
            R = np.diag([sigma_sq_depth, sigma_sq_breadth])
            # R *= .5
            R = rot @ R @ rot.T
            Rs.append(R)
            positions.append(p.reshape(-1).tolist())
            boxes.append(b)
            features.append(self._get_box_features(b, None, cam_type))
            
        return positions, boxes, features, Rs
    
    def get_cones(self, cam_num, cam_type, framenum, frame_time):
        T_WC = self.detections[cam_num][cam_type].T_WC(frame_time, T_BC=self.cone_detections[cam_num].T_BC, true_pose=True)
        T_WC_bel = self.detections[cam_num][cam_type].T_WC(frame_time, T_BC=self.cone_detections[cam_num].T_BC, true_pose=False)
        dets = self.cone_detections[cam_num].detection3d(frame_time, T_WC_bel)
        # for det in dets:
        #     self.raw_cone_detections[cam_num].append(det.reshape(-1).tolist())
        return dets
    
    def get_ordered_detections(self, cams):
        ordered_detections = []
        for cam in cams:
            for det in self.detections:
                ordered_detections.append(det[cam])
        return ordered_detections
    
    def _get_box_features(self, box, im, cam_type):
        return np.ones((2,1))
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, self.x_max[cam_type]), min(y1, self.y_max[cam_type])
        box_im = im[int(y0):int(y1), int(x0):int(x1)]
        feature_vec = self.extractor(box_im)
        return feature_vec.cpu().detach().numpy().reshape((-1,1)) # convert to numpy array
        # TODO: Should use tensor here?
        
    def times_different(self, t1, t2):
        for detections in self.detections:
            if detections.idx(t1) == detections.idx(t2):
                return False
        return True
    
    def get_T_error(self, cam_num, cam_type, frame_time):
        T_WC_true = self.detections[cam_num][cam_type].T_WC(
            frame_time, T_BC=self.detections[cam_num][cam_type].T_BC, true_pose=True)
        T_WC_bel = self.detections[cam_num][cam_type].T_WC(
            frame_time, T_BC=self.detections[cam_num][cam_type].T_BC, true_pose=False)
        return T_WC_true @ inv(T_WC_bel)
    
    def get_T_obj2_obj1(self, cam1_num, cam1_type, cam2_num, cam2_type, frame_time):
        T_WC1_true = self.detections[cam1_num][cam1_type].T_WC(
            frame_time, T_BC=self.detections[cam1_num][cam1_type].T_BC, true_pose=True)
        T_WC1_bel = self.detections[cam1_num][cam1_type].T_WC(
            frame_time, T_BC=self.detections[cam1_num][cam1_type].T_BC, true_pose=False)
        T_WC2_true = self.detections[cam2_num][cam2_type].T_WC(
            frame_time, T_BC=self.detections[cam2_num][cam2_type].T_BC, true_pose=True)
        T_WC2_bel = self.detections[cam2_num][cam2_type].T_WC(
            frame_time, T_BC=self.detections[cam2_num][cam2_type].T_BC, true_pose=False)
        
        return inv(T_WC1_true @ inv(T_WC1_bel)) @ T_WC2_true @ inv(T_WC2_bel)