#########################################
# Authors: Jouko Kinnari, Mason Peterson


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import skimage
from rosbags.highlevel import AnyReader
from yolov7_package import Yolov7Detector
import cv_bridge
import tqdm

from robot_utils.robot_data.img_data import ImgData

import sys
from FastSAM.fastsam import FastSAMPrompt
from FastSAM.fastsam import FastSAM
from fastsam3D.utils import compute_blob_mean_and_covariance, plotErrorEllipse
import fastsam3D.segment_qualification as seg_qual
from fastsam3D.depth_mask_2_centroid import depth_mask_2_centroid, mask_depth_2_width_height
from fastsam3D.fastsam3D import FastSAM3D

IMG_W = 1280
IMG_H = 720

class FastSAM3DBatch():
    
    def __init__(self, 
        fastsam3D : FastSAM3D,
        # T_BC : np.ndarray
    ):        
        # member variables
        self.fastsam3D = fastsam3D
        self.measurements = []
        self.bridge = cv_bridge.CvBridge()
        # self.T_BC = T_BC
        self.img_data = None
        self.depth_data = None
    
    def setup_data(
        self,
        bagfile,
        topic_caminfo,
        topic_img,
        topic_depth,
        img_compressed=True,
        depth_compressed=True,
        start_time=-np.inf,
        end_time=np.inf,
    ):
        self.img_data = ImgData(
            data_file=bagfile,
            file_type='bag',
            topic=topic_img,
            time_range=[start_time, end_time],
            compressed=img_compressed,
            compressed_encoding='bgr8'
        )
        self.img_data.extract_params(topic_caminfo)
        self.depth_data = ImgData(
            data_file=bagfile,
            file_type='bag',
            topic=topic_depth,
            time_range=[start_time, end_time],
            compressed=depth_compressed
        )
        
    def run(self,
        plot_dir=None, # TODO: fix plotting
        step=1,
        num_frames=np.inf,
    ):
        self.measurements = []
        for i in tqdm.tqdm(range(0, len(self.depth_data), step)):
            t = self.depth_data.times[i]
            if self.img_data.idx(t) is None:
                continue
            
            img = self.img_data.img(t)                

            if self.depth_data.compressed:
                # TODO: add the logic for figuring out if depth_data is compressed using rvl
                # rvl will only be needed in rare cases, not sure if it should be a dependency
                # or not
                from rvl import decompress_rvl
                depth = decompress_rvl(
                    np.array(self.depth_data.img_msgs[i].data[20:]).astype(np.int8), IMG_H*IMG_W).reshape((IMG_H, IMG_W))
            else:
                depth = self.depth_data.img(t)
            K = np.array(self.img_data.K).reshape((3,3))
            
            new_meas = self.fastsam3D.run(img, depth, K, plot=False)
            # TODO: fix plotting
             
            new_meas['time'] = self.depth_data.times[i]
            self.measurements.append(new_meas)
            
            # if plot_dir is not None:
            #     self._plot_3d_pos(centroids, means, plot_dir, i, fig, ax)
                
            if i / step > num_frames:
                break
        return self.measurements
    