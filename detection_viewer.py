import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

from frame_info import get_epfl_frame_info

def animate(i, start_frame, cams, bounds):
    minx, maxx, miny, maxy = bounds
    colors = ['green', 'red', 'blue', 'orange']
    ax.clear()
    print('hey')
    for j, cam in enumerate(cams):
        x = []
        y = []
        # print(len(cam.pos(i + start_frame)))
        print(i + start_frame)
        for pose in cam.pos(i + start_frame):
            # print(pose)
            x.append(pose[0])
            y.append(pose[1])
        ax.plot([cam.cam_pos()[0]], cam.cam_pos()[1], 'o', color=colors[j], linewidth=3)
        ax.plot(x, y, 'x', color=colors[j], linewidth=3)
    ax.set_xlim([minx,maxx])
    ax.set_ylim([miny,maxy])
    
start_frame = 40

########### Set up detections ############
frame_infos = get_epfl_frame_info(sigma_r = 0*np.pi/180)

minx, maxx = min(frame_infos[0].cam_pos()[0], frame_infos[1].cam_pos()[0], frame_infos[2].cam_pos()[0], frame_infos[3].cam_pos()[0]), max(frame_infos[0].cam_pos()[0], frame_infos[1].cam_pos()[0], frame_infos[2].cam_pos()[0], frame_infos[3].cam_pos()[0])
miny, maxy = min(frame_infos[0].cam_pos()[1], frame_infos[1].cam_pos()[1], frame_infos[2].cam_pos()[1], frame_infos[3].cam_pos()[1]), max(frame_infos[0].cam_pos()[1], frame_infos[1].cam_pos()[1], frame_infos[2].cam_pos()[1], frame_infos[3].cam_pos()[1])
minx_v = minx - (maxx-minx)*.1
maxx_v = maxx + (maxx-minx)*.1
miny_v = miny - (maxy-miny)*.1
maxy_v = maxy + (maxy-miny)*.1


animate_lambda = lambda i : animate(i, start_frame, frame_infos, [minx_v, maxx_v, miny_v, maxy_v])

fig, ax = plt.subplots()

ani = FuncAnimation(fig, animate_lambda, frames=frame_infos[0].num_frames-start_frame, interval=10, repeat=False)

plt.show()

# xs = list()
# ys = list()
# dirs_x = list()
# dirs_y = list()
# for R, T in zip(Rs, frame_infos):
#     x = (R.T @ T)[0,0]
#     y = (R.T @ T)[1,0]
#     xs.append(x)
#     ys.append(y)
#     dir = R.T @ np.array([[0,0,-1]]).T
#     dirs_x.append([x, x+1000*dir[0,0]])
#     dirs_y.append([y, y+1000*dir[1,0]])
# plt.plot(xs,ys, 'x')
# for dir_x, dir_y in zip(dirs_x, dirs_y):
#     plt.plot(dir_x, dir_y)

# maxs = []
# mins = []
# for i in range(3):
#     maxs.append(-float('inf'))
#     mins.append(float('inf'))
# for pose_list in poses.poses:
#     for pose in pose_list:
#         for i, el in enumerate(pose):
#             maxs[i] = max(maxs[i], el)
#             mins[i] = min(mins[i], el)