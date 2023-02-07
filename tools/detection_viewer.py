import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

import sys
sys.path.append('..')
from src.frontend.detections import get_epfl_frame_info, get_static_test_detections, GroundTruth
GT = GroundTruth('/home/masonbp/ford-project/data/static-20221216/run01_filtered.bag', ['1', '2', '3'], 'RR01')

def animate(i, start_frame, cams, bounds):
    minx, maxx, miny, maxy = bounds
    colors = ['green', 'red', 'blue', 'orange', 'magenta']
    ax.clear()
    t = cams[0].time(i + start_frame)

    # plot camera detections
    for j, cam in enumerate(cams):
        x = []
        y = []
        for pose in cam.pos(t):
            x.append(pose[0])
            y.append(pose[1])
        ax.plot([cam.cam_pos()[0]], cam.cam_pos()[1], 'o', color=colors[j], linewidth=3)
        ax.plot(x, y, 'x', color=colors[j], linewidth=3)

    # plot ground truth
    x = []; y = []
    for pos in GT.ped_positions(t)[1]:
        x.append(pos[0])
        y.append(pos[1])
        print(pos.T)
    ax.plot(x, y, '^', color='black', linewidth=3)

    ax.set_xlim([minx,maxx])
    ax.set_ylim([miny,maxy])
    
start_frame = 40

########### Set up detections ############
frame_infos = get_static_test_detections(run=1, sigma_r = 0*np.pi/180, num_cams=1, cam_type='t265')
num_cams = len(frame_infos)

mins = [np.inf, np.inf]
maxs = [-np.inf, -np.inf]
for xy_idx in [0, 1]:
    for cam_idx in range(num_cams):
        mins[xy_idx] = min(mins[xy_idx], frame_infos[cam_idx].cam_pos()[xy_idx])
        maxs[xy_idx] = max(maxs[xy_idx], frame_infos[cam_idx].cam_pos()[xy_idx])
minx_v = mins[0] - (maxs[0]-mins[0])*.1
maxx_v = maxs[0] + (maxs[0]-mins[0])*.1
miny_v = mins[1] - (maxs[1]-mins[1])*.1
maxy_v = maxs[1] + (maxs[1]-mins[1])*.1


animate_lambda = lambda i : animate(i, start_frame, frame_infos, [minx_v, maxx_v, miny_v, maxy_v])

fig, ax = plt.subplots()

ani = FuncAnimation(fig, animate_lambda, frames=frame_infos[0].num_frames-start_frame, interval=10, repeat=False)

plt.show()