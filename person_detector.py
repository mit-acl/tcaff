import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
# import cv2
# import time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# import torchreid feature extractor
from torchreid.utils import FeatureExtractor

class PersonDetector():

    def __init__(self, device='cuda', threshold=.99):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = device
        self.predictor = DefaultPredictor(self.cfg)
        classes = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes", None)
        self.person_class_idx = classes.index('person')
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0', # TODO: Is this a good enough re-id network?
            # model_path='a/b/c/model.pth.tar',
            device=device
        )

    def get_person_boxes(self, im):
        outputs = self.predictor(im)
        boxes = []
        features = []
        for b, c in zip(outputs['instances'].pred_boxes, outputs['instances'].pred_classes):
            if c == self.person_class_idx:
                boxes.append(b.tolist())
                features.append(self._get_box_features(b.tolist(), im))
        return boxes, features
    
    def _get_box_features(self, box, im):
        x0, y0, x1, y1 = box
        box_im = im[int(y0):int(y1), int(x0):int(x1)]
        feature_vec = self.extractor(box_im)
        return feature_vec.cpu().detach().numpy().reshape((-1,1)) # convert to numpy array
        # TODO: Should use tensor here?
        