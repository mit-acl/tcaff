import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.transforms import Affine2D
import numpy as np
from numpy.linalg import inv, norm
from scipy.spatial.transform import Rotation as Rot
import cv2 as cv
import sys
import argparse

sys.path.append('../src')
from motlee.utils.transform import transform

class RoverJSONParser():

    def __init__(self, datafile):
        self.data = []
        f = open(datafile)
        self.data = json.load(f)
        
    def idx(self, framenum):
        '''
        returns the data index for a desired framenum
        '''
        idx = None
        for j, df in enumerate(self.data):
            if df['framenum'] >= framenum:
                idx = j
                break
        return idx
    
    def get_two_pt_rover(self, framenum, dist=0.5, est=False):
        '''
        returns a body point and a second point indicating the direction of the rover
        '''
        idx = self.idx(framenum)
        T_WC = np.array(self.data[idx]['T_WC_bel' if est else 'T_WC']).reshape((4,4))
        body_pt = T_WC[:2,3]
        # use z vector (RDF frame) to get heading pt
        heading_pt = body_pt + dist * (T_WC[:2,2] / norm(T_WC[:2,2]))
        return body_pt, heading_pt

    def get_cones(self, framenum, est=False):
        return self._get_objects(framenum, 'cones', est=est)
    
    def get_tracks(self, framenum, est=False):
        return self._get_objects(framenum, 'tracks', est=est)
    
    def get_T_WC(self, framenum, est=False):
        idx = self.idx(framenum)
        T_WC = np.array(self.data[idx]['T_WC_bel' if est else 'T_WC']).reshape((4,4))
        return T_WC
        
    def _get_objects(self, framenum, object_name, est=False):
        idx = self.idx(framenum)
        objects = np.array(self.data[idx][object_name])
        if objects.shape[0] == 0:
            return objects
        if est:
            return objects
        T_WC = np.array(self.data[idx]['T_WC']).reshape((4,4))
        T_WC_bel = np.array(self.data[idx]['T_WC_bel']).reshape((4,4))
        objects = transform(T_WC @ inv(T_WC_bel), objects, stacked_axis=0)
        return objects