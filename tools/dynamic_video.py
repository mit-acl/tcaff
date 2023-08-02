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

plt.style.use('/home/masonbp/computer/python/matplotlib/publication.mplstyle')

############# OPTIONS #################
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output',
                    type=str,
                    default=None,
                    help='file to save video to.')
parser.add_argument('-f', '--fps',
                    default=1,
                    type=int,
                    help='frames per second.')
parser.add_argument('-1', '--single-frame', 
                    default=None,
                    type=int,
                    help='only view one frame.')
parser.add_argument('--rate',
                    default=2,
                    help='speedup.')
parser.add_argument('-n', '--num-rovers',
                    default=3,
                    help='3 or 4')
parser.add_argument('-r', '--rover',
                    default='RR06',
                    type=str,
                    help='RR01, RR04, RR06, or RR08')
parser.add_argument('--num-frames',
                    type=int,
                    default=None,
                    help='Number of frames to include in video')
parser.add_argument('--fig1', action='store_true')
args = parser.parse_args()
fps = args.fps
num_rovers = args.num_rovers
ROVERS = ['RR04', 'RR06', 'RR08'] if num_rovers == 3 else ['RR01', 'RR04', 'RR06', 'RR08']
rover = args.rover

datafile1 = f'/home/masonbp/ford/data/mot_dynamic/dynamic_motlee_iros/results/iros/cam_project_test_nofix_{num_rovers}r.json'
datafile2 = f'/home/masonbp/ford/data/mot_dynamic/dynamic_motlee_iros/results/iros/cam_project_test_realign_{num_rovers}r.json'
video_out = args.output
video_in = f'/home/masonbp/ford/data/mot_dynamic/dynamic_motlee_iros/videos/l515/run1_{rover}.avi'

fx = 901.47021484375
fy = 901.8353881835938
cx = 649.6925048828125
cy = 365.004150390625
params = dict()
params['K'] = np.array([[fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]])

class Viewer():

    def __init__(self, datafiles, vidfile, rover, params=params):
        self.last_T_fix = np.eye(4).reshape((16,))

        # setup plotting
        plt.rcParams.update({'font.size': 5})
        # plt.rcParams.update({'wspace'})
        # width = 3.487
        width = 8
        self.fig, (self.axs) = plt.subplots(2, 2, figsize=(width, width*10/16))
        
        # self.fig.subplots_adjust(wspace=0, hspace=0)
        self.xlim = np.array([-32/3, 32/3])*1
        self.ylim = np.array([-6., 6.])*1
        self.marker_size = 6
        self.rover_scale = 2.25
        title_fontsize=13
        self.axs[0, 0].set_title("standard MOT in presence of localization error", fontname='Times New Roman', fontsize=title_fontsize, fontweight='bold')
        self.axs[0, 1].set_title("MOTLEE in presence of localization error", fontname='Times New Roman', fontsize=title_fontsize, fontweight='bold')
        
        for i in range(2):
            # self.axs[0, i].set_xlabel('m', labelpad=2)
            # self.axs[0, i].set_ylabel('m', labelpad=2)
            self.axs[0, i].set_xlim(self.xlim)
            self.axs[0, i].set_ylim(self.ylim)
            self.axs[0, i].set_xticks([-5, 0, 5], minor=False)
            self.axs[0, i].set_yticks([-5, 0, 5], minor=False)
            self.axs[0, i].grid(True)
            self.axs[1, i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            self.axs[1, i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            self.axs[0, i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            self.axs[0, i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            # self.axs[0, i].margins(x=0, y=0)
            # self.axs[1, i].margins(x=0, y=0)
        self.fig.set_dpi(300)


        self.data = []
        for df in datafiles:
            f = open(df)
            self.data.append(json.load(f))
        self.cap = cv.VideoCapture(vidfile)
        self.rover = rover
        self.K = params['K']
        self.aspect = 1.6428571428571428
        self.fov = np.deg2rad(42)
        self.cam2notcam = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])

        self.rover_artists = []
        for i in range(2):
            rover_dict = {'est': {}, 'gt': {}}
            for rover in ROVERS:
                rover_dict['gt'][rover] = self.get_rover_artist(self.axs[0, i], grayed_out=True, main_rover=(self.rover == rover))
            for rover in ROVERS:
                rover_dict['est'][rover] = self.get_rover_artist(self.axs[0, i], grayed_out=False, main_rover=(self.rover == rover))
            self.rover_artists.append(rover_dict)
        self.objs = [[], []]
        
    def view(self, framenum):
        '''
        Shows the desired frame number
        
        Parameters
        ----------
        framenum : int
        '''
        self.cap.set(cv.CAP_PROP_POS_FRAMES, framenum-1)
        res, frame_orig = self.cap.read()
        # self.fig.suptitle(f'Frame: {framenum}')
        print(framenum)

        # haha
        for i, datum in enumerate(self.data):
            # new plot setup
            # self.axs[0, i].clear()
            for j in range(len(self.objs[i])):
                obj = self.objs[i].pop(0).pop(0)
                obj.remove()
            self.axs[1, i].clear()
            frame = np.copy(frame_orig)

            df_idx = None
            for j, df in enumerate(datum):
                if df['framenum'] == framenum:
                    df_idx = j
                    break
            df = datum[df_idx]
            
            for gt in df['groundtruth']:
                num_viewed_by = 0
                for r in df['rovers']:
                    T_WC = np.array(df['rovers'][r]['T_WC']).reshape((4,4))
                    if is_viewable(np.array(gt), T_WC):
                        num_viewed_by += 1
                # assert num_viewed_by > 0
                # if num_viewed_by == 0:
                #     color='black'
                # if num_viewed_by == 1:
                #     color='yellowgreen'
                # else:
                #     color='purple'
                color='yellowgreen'
                self.objs[i].append(self.axs[0, i].plot(gt[0], gt[1], 'o', markersize=self.marker_size, color=color))

            rover = df['rovers'][self.rover]
            T_WC = np.array(rover['T_WC']).reshape((4,4))
            T_WC_bel = np.array(rover['T_WC_bel']).reshape((4,4))
            for track in rover['tracks']:

                track_corrected = transform(T_WC @ inv(T_WC_bel), np.array(track + [0]))
                track_corrected[2] = 0
                self.objs[i].append(self.axs[0, i].plot(track_corrected[0], track_corrected[1], 'x', markersize=self.marker_size, color='blue'))

                track_c = transform(inv(T_WC), track_corrected)
                if track_c.item(2) > 0:
                    uvs = (self.K @ track_c)
                    uvs /= uvs.item(2)
                    pixel_coords = uvs.reshape(-1)[:2]
                    pixel_coords /= 1
                    cv.drawMarker(frame, tuple(pixel_coords[:].astype(int).tolist()), self.get_track_color(0), cv.MARKER_TILTED_CROSS, thickness=10, markerSize=90)

            for r in df['rovers']:
                T_WR = np.array(df['rovers'][r]['T_WC']).reshape((4,4))
                T_WR = self.T2pltT(T_WR, self.axs[0, i])
                self.draw_rover(self.rover_artists[i]['gt'][r], T_WR)

            for r, T_fix in rover['T_fix'].items():
                T_fix = np.array([T_fix]).reshape((4,4))
                r_name = ROVERS[int(r) % len(ROVERS)]
                T_j_hat = np.array([df['rovers'][r_name]['T_WC_bel']]).reshape((4,4))
                T_j = T_WC @ inv(T_WC_bel) @ T_fix @ T_j_hat
                T_j = self.T2pltT(T_j, self.axs[0, i])
                self.draw_rover(self.rover_artists[i]['est'][r_name], T_j)
                # self.axs[0, i].set_aspect((self.xlim[1]-self.xlim[0]) / (self.ylim[1]-self.ylim[0]))
                self.axs[0, i].set_aspect(1)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            self.axs[1, i].imshow(frame[...,::-1])
            self.fig.subplots_adjust(
                top=0.95,
                bottom=0.01,
                left=0.01,
                right=0.99,
                wspace=0.05, 
                hspace=0.05)
            
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
        # print(ax.add_patch)
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


viewer = Viewer([datafile1, datafile2], video_in, rover)
start_frame = 125*30 #3750 #120*30
viewer.fig.subplots_adjust(wspace=0, hspace=0)
if not args.num_frames:
    num_frames = (180-125) * fps
else:
    num_frames = args.num_frames

if not args.single_frame:
    # saving to m4 using ffmpeg writer
    ani = FuncAnimation(viewer.fig, lambda i: viewer.view(i*30/fps + start_frame), frames=num_frames, interval=1, repeat=False) 
    print('saving video...')
    writervideo = FFMpegWriter(fps=int(fps*args.rate))
    ani.save(video_out, writer=writervideo)
    plt.close()
else:
    viewer.view(args.single_frame)
    if args.fig1:
        annoation_fontsize=12
        arrow_kw = {'head_width': .3, 'fc': 'k'}
        viewer.axs[0, 0].text(-9, 2, "   false \npositive", fontsize=annoation_fontsize)
        viewer.axs[0, 0].arrow(-7, 1.5, .4, -1.7, **arrow_kw)
        viewer.axs[0, 0].text(-5.5, -4.8, "perceived robot pose, due to drift", fontsize=annoation_fontsize)#, backgroundcolor='white')
        viewer.axs[0, 0].arrow(-5.7, -4.4, -.7, .45, **arrow_kw)
        viewer.axs[0, 0].text(-1.5, 4.4, "ground truth pose", fontsize=annoation_fontsize)#, backgroundcolor='white')
        viewer.axs[0, 0].arrow(1.3, 4.2, -1.1, -.65, **arrow_kw)
        viewer.axs[0, 0].arrow(1.7, 4.2, -1.4, -4.3, **arrow_kw)
        viewer.axs[0, 0].text(7, .8, "track", fontsize=annoation_fontsize)#, backgroundcolor='white')
        viewer.axs[0, 0].arrow(6.8, .8, -1.4, -1.0, **arrow_kw)
    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output)