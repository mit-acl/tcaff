# import torchreid feature extractor
from torchreid.utils import FeatureExtractor
from frame_info import get_epfl_frame_info

class PersonDetector():

    def __init__(self, device='cuda', threshold=.99, sigma_r=0, sigma_t=0):
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0', # TODO: Is this a good enough re-id network?
            # model_path='a/b/c/model.pth.tar',
            device=device
        )
        self.frames = get_epfl_frame_info(sigma_r=sigma_r, sigma_t=sigma_t)

    def get_person_boxes(self, im, cam_num, frame_num):
        positions = []
        boxes = []
        features = []
        for b, p in zip(self.frames[cam_num].bbox(frame_num), self.frames[cam_num].pos(frame_num)):
            positions.append(p.reshape(-1).tolist())
            boxes.append(b)
            features.append(self._get_box_features(b, im))
        return positions, boxes, features
    
    def _get_box_features(self, box, im):
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, 360), min(y1, 288)
        box_im = im[int(y0):int(y1), int(x0):int(x1)]
        feature_vec = self.extractor(box_im)
        return feature_vec.cpu().detach().numpy().reshape((-1,1)) # convert to numpy array
        # TODO: Should use tensor here?
        