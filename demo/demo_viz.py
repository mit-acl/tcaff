# Last call:
# python3 dynamic_video.py --fps 4 --show-times -o /home/masonbp/results/motlee_nov_2023/videos/tcaff_035_new.mp4 --rate 2. 

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot
from typing import Dict, List

from motlee.utils.transform import transform
from motlee.utils.cam_utils import is_viewable

from robot_utils.robot_data import PoseData

class DemoViz():

    def __init__(
        self, 
        focus_robot: str, 
        robot_names: List[str],
        T_WR_est: Dict[str, PoseData],
        T_WR_gt: Dict[str, PoseData],
    ):
        self.robot_names = robot_names
        self.focus_robot = focus_robot
        self.T_WR_est = T_WR_est
        self.T_WR_gt = T_WR_gt
    
        self.last_T_fix = np.eye(4).reshape((16,))

        # setup plotting
        width = 5.0
        height = 5.0
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=400)
        
        self.xlim = np.array([-8., 8.])*1
        self.ylim = np.array([-8., 8.])*1
        self.marker_size = 3
        self.mew = 1
        self.rover_scale = 2.25
      
        self.ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

        self.fig.subplots_adjust(
            top=0.92,
            bottom=0.01,
            left=0.01,
            right=0.99,
            wspace=0.05, 
            hspace=0.05)
        
        self.aspect = 1.6428571428571428
        self.fov = np.deg2rad(42)
        self.cam2notcam = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])

        self.rover_artists = dict()
        rover_dict = {'est': {}, 'gt': {}}
        for robot in self.focus_robot:
            rover_dict['gt'][robot] = self.get_rover_artist(self.ax_ani, grayed_out=True, main_rover=(self.focus_robot == robot))
        for robot in self.focus_robot:
            rover_dict['est'][robot] = self.get_rover_artist(self.ax_ani, grayed_out=False, main_rover=(self.focus_robot == robot))
        self.rover_artists = rover_dict
        self.objs = []
        
    def update(
            self, 
            # T_WR: Dict[str, np.array], 
            # T_WR_gt: Dict[str, np.array], 
            t: float,
            T_oioj: Dict[str, np.array],
        ):
        # TODO: leaving off here. I think it would be nice if this knows nothing about the animation, but just 
        # updates the plot when it's called, and maybe it has a ts for updating the plot and returns True if it is updated...

        for j in range(len(self.objs)):
            obj = self.objs.pop(0).pop(0)
            obj.remove()
        
        # for gt in df['groundtruth']:
        #     num_viewed_by = 0
        #     for r in df['rovers']:
        #         T_WC = np.array(df['rovers'][r]['T_WC']).reshape((4,4))
        #         if is_viewable(np.array(gt), T_WC):
        #             num_viewed_by += 1
        #     # assert num_viewed_by > 0
        #     # if num_viewed_by == 0:
        #     #     color='black'
        #     # if num_viewed_by == 1:
        #     #     color='yellowgreen'
        #     # else:
        #     #     color='purple'
        #     color='yellowgreen'
        #     self.objs.append(self.ax_ani.plot(gt[0], gt[1], 'o', markersize=self.marker_size, color=color, mew=self.mew))

        robot = df['rovers'][self.focus_robot]
        T_WC = np.array(robot['T_WC']).reshape((4,4))
        T_WC_bel = np.array(robot['T_WC_bel']).reshape((4,4))
        T_wri = T_WR_gt[self.focus_robot] # T^{world}_{robot_i}
        T_oiri = T_WR[self.focus_robot] # T^{odom_i}_{robot_i}
        # for track in robot['tracks']:

        #     track_corrected = transform(T_WC @ inv(T_WC_bel), np.array(track + [0]))
        #     track_corrected[2] = 0
        #     self.objs.append(self.ax_ani.plot(track_corrected[0], track_corrected[1], 'x', markersize=self.marker_size, color='blue', mew=self.mew))

        for r in self.robot_names:
            T_wrj_gt = T_WR_gt[r]
            T_wrj_gt = self.T2pltT(T_wrj_gt, self.ax)
            self.draw_rover(self.rover_artists['gt'][r], T_wrj_gt)

        for r, T_oioj_i in T_oioj.items():
            T_ojrj = T_WR[r]
            Twrj = T_wri @ inv(T_oiri) @ T_oioj @ T_ojrj
            Twrj = self.T2pltT(Twrj, self.ax_ani)
            self.draw_rover(self.rover_artists['est'][r], Twrj)
            # self.ax_ani.set_aspect((self.xlim[1]-self.xlim[0]) / (self.ylim[1]-self.ylim[0]))
            self.ax.set_aspect(1)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
            
        return

    def get_rover_artist(self, ax, grayed_out, main_rover):
        if main_rover:
            rover_color = '#a45ee5'
            wheel_color = 'k'
        elif grayed_out:
            rover_color = '#dbb4bc'
            wheel_color = 'gray'
        else:
            rover_color = '#DC143C'
            wheel_color = 'k'
        include_frustum = grayed_out
        s = self.rover_scale # scale factor
        WIDTH = 0.381 * s
        LENGTH = 0.508 * s
        WHEELR = 0.12 * s #0.222 * s # radius of wheel
        WHEELD = 0.268 * s # distance btwn wheel centers
        WHEELT = (0.497 * s - WIDTH) / 2. # wheel thickness
        T_WB = Affine2D(np.eye(3))

        bl = - np.r_[LENGTH, WIDTH] / 2.
        body = plt.Rectangle(bl, LENGTH, WIDTH, color=rover_color, transform=T_WB)

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
        top = near * np.tan(self.fov / 2.)
        bottom = -top
        right = self.aspect * top
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
        if include_frustum:
            ax.add_patch(frustum)
        # ax.add_patch(plt.Rectangle((0.5, 0.5), 2, 2, color=wheel_color))

        artist = {'body': body, 'orientation': orientation, 'wheels': wheels, 'frustum': frustum if include_frustum else None}
        return artist
    
    def T2pltT(self, T, ax):
        T[:3, :3] = T[:3,:3] @ self.cam2notcam
        T = np.delete(T, [2], axis=0)
        T = np.delete(T, [2], axis=1)
        T = Affine2D(T) + ax.transData
        return T

    def pts_from(self, T_WC):
        x = [T_WC[0,3]] 
        y = [T_WC[1,3]]
        length = .5
        R = Rot.from_matrix(T_WC[:3,:3] @ self.cam2notcam )
        th = R.as_euler('xyz', degrees=False)[2]
        x.append(T_WC[0,3] + length * np.cos(th))
        y.append(T_WC[1,3] + length * np.sin(th))
        return x, y
    
    def draw_rover(self, artist, T_WB):
        artist['body'].set_transform(T_WB)
        [w.set_transform(T_WB) for w in artist['wheels']]
        artist['orientation'].set_transform(T_WB)
        if artist['frustum'] is not None:
            artist['frustum'].set_transform(T_WB)
        # artist['label'].set_transform(T_WB)

        return
        
    def get_track_color(self, i):
        c = np.array(plt.get_cmap('tab10').colors[i])
        c = (c * 255).astype(int)
        return tuple([255, 30, 30])
        return tuple(v.item() for v in c[::-1])
