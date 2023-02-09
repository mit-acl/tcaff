# import torchreid feature extractor
from torchreid.utils import FeatureExtractor
if __name__ == '__main__':
    from detections import get_rover_detections
else:
    from .detections import get_rover_detections
import numpy as np
from numpy.linalg import inv

class PersonDetector():

    def __init__(self, bagfile, device='cuda', sigma_r=0, sigma_t=0, 
                 cams=['RR01', 'RR04', 'RR05', 'RR06', 'RR08'], cam_type='t265', 
                 cam_pose_topic='/world'):
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0', # TODO: Is this a good enough re-id network?
            # model_path='a/b/c/model.pth.tar',
            device=device,
            verbose=False
        )
        # self.detections = get_epfl_frame_info(sigma_r=sigma_r, sigma_t=sigma_t)
        self.detections = get_rover_detections(bagfile=bagfile, sigma_r=sigma_r, sigma_t=sigma_t, 
            rovers=cams, cam_type=cam_type, rover_pose_topic=cam_pose_topic)
        self.x_max = 1920
        self.y_max = 1080
        self.start_time = self.detections[0].time(0)
        self.num_cams = len(cams)

    def get_person_boxes(self, im, cam_num, frame_time):
        positions = []
        boxes = []
        features = []
        for b, p in zip(self.detections[cam_num].bbox(frame_time), self.detections[cam_num].pos(frame_time)):
            positions.append(p.reshape(-1).tolist())
            boxes.append(b)
            features.append(self._get_box_features(b, im))
        return positions, boxes, features
    
    def _get_box_features(self, box, im):
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, self.x_max), min(y1, self.y_max)
        box_im = im[int(y0):int(y1), int(x0):int(x1)]
        feature_vec = self.extractor(box_im)
        return feature_vec.cpu().detach().numpy().reshape((-1,1)) # convert to numpy array
        # TODO: Should use tensor here?
        
    def times_different(self, t1, t2):
        for detections in self.detections:
            if detections.idx(t1) == detections.idx(t2):
                return False
        return True
    
    # def get_T_obj2_obj1(self, cam1, cam2, incl_noise=False):
    #     if incl_noise:
    #         R1, t1, R1_noise, t1_noise = self.get_cam_pose(cam1)
    #         R2, t2, R2_noise, t2_noise = self.get_cam_pose(cam2)
    #         T_b1_g_translation = np.eye(4)
    #         T_b1_l1 = np.eye(4)
    #         T_b2_g_translation = np.eye(4)
    #         T_b2_l2 = np.eye(4)
            
    #         T_b1_l1[0:3, 3] = t1_noise.reshape(-1)
    #         T_b1_l1[0:3, 0:3] = R1_noise
    #         T_b1_g_translation[0:3, 3] = t1.reshape(-1)
    #         T_b2_l2[0:3, 3] = t2_noise.reshape(-1)
    #         T_b2_l2[0:3, 0:3] = R2_noise
    #         T_b2_g_translation[0:3, 3] = t2.reshape(-1)
            
    #         return T_b1_g_translation @ T_b1_l1 @ inv(T_b1_g_translation) @ \
    #             T_b2_g_translation @ inv(T_b2_l2) @ inv(T_b2_g_translation)
    #     else:
    #         return np.eye(4)