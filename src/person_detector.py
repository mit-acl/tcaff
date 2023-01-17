# import torchreid feature extractor
from torchreid.utils import FeatureExtractor
from detections import get_epfl_frame_info, get_static_test_detections
import numpy as np

class PersonDetector():

    def __init__(self, run=1, device='cuda', threshold=.99, sigma_r=0, sigma_t=0, num_cams=4):
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0', # TODO: Is this a good enough re-id network?
            # model_path='a/b/c/model.pth.tar',
            device=device,
            verbose=False
        )
        # self.detections = get_epfl_frame_info(sigma_r=sigma_r, sigma_t=sigma_t)
        self.detections = get_static_test_detections(run=run, sigma_r=sigma_r, sigma_t=sigma_t, num_cams=num_cams)
        self.x_max = 1920
        self.y_max = 1080
        self.start_time = self.detections[0].time(0)
        self.num_cams = num_cams

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
    
    def get_cam_pose(self, cam_num):
        return self.detections[cam_num].R, self.detections[cam_num].T, self.detections[cam_num].R_offset, self.detections[cam_num].T_offset
    
    def get_cam_T(self, cam_num):
        return np.concatenate([np.concatenate([self.detections[cam_num].R, self.detections[cam_num].T], axis=1), [[0, 0, 0, 1]]], axis=0)

    # def get_T_cam1_cam2(self, cam1, cam2, incl_noise=False):
    #     if not incl_noise