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

from rover_json_parser import RoverJSONParser

############# OPTIONS #################
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output',
                    type=str,
                    default='/home/masonbp/ford/motlee/vids/single_agent_maps_tracks.mp4',
                    help='file to save video to.')
parser.add_argument('-f', '--fps',
                    default=1,
                    type=int,
                    help='frames per second.')
parser.add_argument('-r', '--root',
                    default='/home/masonbp/ford/data/mot_dynamic/dynamic_motlee_iros/results/single_agent_tracks_maps/',
                    type=str,
                    help='Root for data files.')
parser.add_argument('-1', '--single-frame', 
                    default=None,
                    type=int,
                    help='only view one frame.')
parser.add_argument('--rate',
                    default=2,
                    help='speedup.')
parser.add_argument('--num-frames',
                    type=int,
                    default=None,
                    help='Number of frames to include in video')

args = parser.parse_args()
fps = args.fps
video_out = args.output

class SingleAgentMapsTracks():

    def __init__(self, root):
        self.parsers = {}
        for rover in ['RR01', 'RR04', 'RR06', 'RR08']:
            self.parsers[rover] = RoverJSONParser(f'{args.root}/{rover}.json')
        self.rovers = [['RR01', 'RR04'], ['RR06', 'RR08']]
        
        self.fig, self.axs = plt.subplots(2, 2)
        
        # Parameters
        self.track_history = 30

    def setup_plt(self):
        for i in range(self.axs.shape[0]):
            for j in range(self.axs.shape[1]):
                self.axs[i,j].clear()
                self.axs[i,j].set_xlim([-9, 9])
                self.axs[i,j].set_ylim([-7.5, 7.5])
                self.axs[i,j].set_aspect('equal')
                self.axs[i,j].set_title(self.rovers[i][j])
                self.axs[i,j].grid(True)
    
    def view(self, framenum):
        '''
        Shows the desired frame number
        
        Parameters
        ----------
        framenum : int
        '''
        print(framenum)
        self.fig.suptitle(framenum)
        self.setup_plt()
        for i in range(2):
            for j in range(2):
                rover = self.rovers[i][j]
                body_pt, heading_pt = self.parsers[rover].get_two_pt_rover(framenum)
                self.axs[i,j].plot([body_pt[0]], [body_pt[1]], 'o', color='k')
                self.axs[i,j].plot([body_pt[0],  heading_pt[0]], [body_pt[1], heading_pt[1]], color='k')
                
                cones = self.parsers[rover].get_cones(framenum)
                tracks = [self.parsers[rover].get_tracks(k) for k in range(framenum, framenum+self.track_history) if self.parsers[rover].get_tracks(k).shape[0] > 0]

                if cones.shape[0] > 0:
                    self.axs[i,j].plot(cones[:,0], cones[:,1], 'o', color='orange')
                if len(tracks) > 0:
                    tracks = np.vstack(tracks)
                    self.axs[i,j].plot(tracks[:,0], tracks[:,1], '.', color='green', markersize=3)


animator = SingleAgentMapsTracks(args.root)
if not args.num_frames:
    num_frames = int(6000*fps/30)
else:
    num_frames = args.num_frames

if not args.single_frame:
    # saving to m4 using ffmpeg writer
    start_frame = 6*30
    ani = FuncAnimation(animator.fig, lambda i: animator.view(int(i*30/fps + start_frame)), frames=num_frames, interval=1, repeat=False) 
    print('saving video...')
    writervideo = FFMpegWriter(fps=int(fps*args.rate))
    ani.save(video_out, writer=writervideo)
    plt.close()
else:
    animator.view(args.single_frame)
    plt.show()