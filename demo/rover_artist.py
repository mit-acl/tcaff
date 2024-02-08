# Authors: Parker Lusk, Mason Peterson

import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.transforms import Affine2D
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot
import cv2 as cv
import sys
import argparse

from motlee.utils.transform import transform
from motlee.utils.cam_utils import is_viewable

class RoverArtist():
        

    def __init__(self, 
                 fig,
                 ax, 
                 rover_color='#DC143C', 
                 wheel_color='k', 
                 draw_frustum=True, 
                 scale=1., 
                 aspect=1,
                 fov=np.deg2rad(42)):
        
        self.ax = ax
        self.fig = fig
        
        s = scale # scale factor
        WIDTH = 0.381 * s
        LENGTH = 0.508 * s
        WHEELR = 0.12 * s #0.222 * s # radius of wheel
        WHEELD = 0.268 * s # distance btwn wheel centers
        WHEELT = (0.497 * s - WIDTH) / 2. # wheel thickness
        T_WB = Affine2D(np.eye(3))

        bl = - np.r_[LENGTH, WIDTH] / 2.
        body = plt.Rectangle(bl, LENGTH, WIDTH, color=rover_color, transform=T_WB)
        # print(bl)

        wheels = []
        wbl = - np.r_[LENGTH, WHEELD] / 2. - np.r_[0, WHEELT] * 1.7 + np.r_[0.01 * s, 0]
        wheels.append(plt.Rectangle(wbl, WHEELR, WHEELT, color=wheel_color, transform=T_WB))
        wbl = wbl + np.r_[WHEELD, 0]
        wheels.append(plt.Rectangle(wbl, WHEELR, WHEELT, color=wheel_color, transform=T_WB))
        wbl = wbl + np.r_[0, WIDTH] + np.r_[0, WHEELT] / 2
        wheels.append(plt.Rectangle(wbl, WHEELR, WHEELT, color=wheel_color, transform=T_WB))
        wbl = wbl - np.r_[WHEELD, 0]
        wheels.append(plt.Rectangle(wbl, WHEELR, WHEELT, color=wheel_color, transform=T_WB))

        pts = np.array([[0, - 0.8 * WIDTH / 2], [ 1.3 * WIDTH / 2, 0], [0, 0.8 * WIDTH / 2]])
        orientation = plt.Polygon(pts, closed=True, color=wheel_color, transform=T_WB)

        # camera FOV frustrum
        near = 0.1 * s
        far = .7 * s
        # aspect = 1.6428571428571428
        top = near * np.tan(fov / 2.)
        bottom = -top
        right = aspect * top
        left = -right
        rat = far/near

        pts = np.array([[near,left], [near,right], [far,rat*right], [far,rat*left]])
        pts += np.c_[LENGTH/2.,0]
        frustum = plt.Polygon(pts, closed=True, color='b', alpha=0.1, transform=T_WB)

        # label = plt.text(0.6 * -LENGTH/2, 0.2 * WIDTH/2, 0, fontsize='small', transform=T_WB,
        #                         # transform_rotates_text=True, 
        #                         rotation=-90, rotation_mode='anchor',
        #                         horizontalalignment='center', verticalalignment='center',
        #                         color='white')

        for wheel in wheels:
            ax.add_patch(wheel)
        ax.add_patch(body)
        ax.add_patch(orientation)
        if draw_frustum:
            ax.add_patch(frustum)
        # print(ax.add_patch)
        # ax.add_patch(plt.Rectangle((0.5, 0.5), 2, 2, color=wheel_color))

        self.artist = {'body': body, 'orientation': orientation, 'wheels': wheels, 'frustum': frustum if draw_frustum else None}
    
    def draw(self, T_WB):
        T_WB = Affine2D(T_WB) + self.ax.transData
        self.artist['body'].set_transform(T_WB)
        [w.set_transform(T_WB) for w in self.artist['wheels']]
        self.artist['orientation'].set_transform(T_WB)
        if self.artist['frustum'] is not None:
            self.artist['frustum'].set_transform(T_WB)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        return        