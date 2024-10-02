import numpy as np
from yolov7_package import Yolov7Detector
from tcaff.utils.cam_utils import depth_mask_2_centroid

class PersonDetector3D():
    
    def __init__(self, K, depth_scale=.001, img_size=(256, 256), **kwargs):
        self.K = K
        self.depth_scale=depth_scale
        self.detector = Yolov7Detector(traced=False, img_size=img_size, **kwargs)
        
    def masks(self, img, scale1d=.5):
        """
        Creates images masks for detected people

        Args:
            img (cv image or numpy array): Query image
            scale1d (float, optional): Return masks that are scaled along each axis by this value. 
                This is useful for creating centered masks for depth detection. Defaults to .5.

        Returns:
            list: List of numpy arrays containing person masks
            list: List of pixel centroids
        """
        classes, boxes, scores = self.detector.detect(img)
        person_boxes = []
        for i, cl in enumerate(classes[0]):
            if self.detector.names[cl] == 'person':
                person_boxes.append(boxes[0][i])

        masks = [np.zeros(img.shape[:2]).astype(np.int8) for _ in person_boxes]
        centroids = []
        for box, mask in zip(person_boxes, masks):
            x0, y0, x1, y1 = np.array(box).astype(np.int64).reshape(-1).tolist()
            w, h = x1 - x0, y1 - y0
            xc, yc = x0 + w / 2, y0 + h / 2
            x0, x1 = int(xc - w*.5*scale1d), int(xc + w*.5*scale1d)
            y0, y1 = int(yc - h*.5*scale1d), int(yc + h*.5*scale1d)
            # x0 = max(x0, 0)
            # y0 = max(y0, 0)
            # x1 = min(x1, mask.shape[1])
            # y1 = min(y1, mask.shape[0])
            mask[y0:y1,x0:x1] = np.ones((y1-y0, x1-x0)).astype(np.int8)
            centroids.append((xc, yc))
        return masks, centroids
        
    def detect(self, img_color, img_depth):
        assert img_color.shape[:2] == img_depth.shape[:2], "Depth and color image shapes do not match"
        masks, centroids_pixels = self.masks(img_color)
        centroids_3d = []
        for mask, cent_pix in zip(masks, centroids_pixels):
            centroids_3d.append(depth_mask_2_centroid(img_depth, mask, cent_pix, self.K, self.depth_scale))
            
        return centroids_3d
        