import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import time

import sys
sys.path.append('..')
sys.path.append('../src')
from src.frontend.detections import get_rover_detections, GroundTruth
GT = GroundTruth('/home/masonbp/ford-project/data/dynamic-final/run1.bag')

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
        ax.plot([cam.T_WC(t)[0,3]], [cam.T_WC(t)[1,3]], '-o', color=colors[j], linewidth=3)
        ax.plot([cam.T_WC(t+.5)[0,3]], [cam.T_WC(t+.5)[1,3]], '-o', color=colors[j], linewidth=3)
        # ax.plot([[cam.T_WC(t)[0,3], cam.T_WC(t+.5)[0,3]], [cam.T_WC(t)[1,3], cam.T_WC(t+.5)[1,3]]], '-o', color=colors[j], linewidth=3)
        ax.plot(x, y, 'x', color=colors[j], linewidth=3)

    # plot ground truth
    x = []; y = []
    for pos in GT.ped_positions(t)[1]:
        x.append(pos[0])
        y.append(pos[1])
    ax.plot(x, y, '^', color='black', linewidth=3)

    ax.set_xlim([minx,maxx])
    ax.set_ylim([miny,maxy])
    
start_frame = 40
end_frame = 2000

########### Set up detections ############
# frame_infos = get_static_test_detections(run=1, sigma_r = 0*np.pi/180, num_cams=1, cam_type='t265')
print('reading data...')
frame_infos = get_rover_detections(
    # bagfile=f'/home/masonbp/ford-project/data/dynamic-final/centertrack_detections/t265/0.5_0.1/run1_{{}}.bag',
    bagfile=f'/home/masonbp/ford-project/data/dynamic-final/centertrack_detections/new_run/0.5_2.0/run1_RR01.bag',
    rovers=['RR01'], #['RR01', 'RR04', 'RR06', 'RR08'],
    cam_type='l515',
    rover_pose_topic='/world'
)
num_cams = len(frame_infos)

# mins = [np.inf, np.inf]
# maxs = [-np.inf, -np.inf]
# for xy_idx in [0, 1]:
#     for cam_idx in range(num_cams):
#         mins[xy_idx] = min(mins[xy_idx], frame_infos[cam_idx].cam_pos()[xy_idx])
#         maxs[xy_idx] = max(maxs[xy_idx], frame_infos[cam_idx].cam_pos()[xy_idx])
# minx_v = mins[0] - (maxs[0]-mins[0])*.1
# maxx_v = maxs[0] + (maxs[0]-mins[0])*.1
# miny_v = mins[1] - (maxs[1]-mins[1])*.1
# maxy_v = maxs[1] + (maxs[1]-mins[1])*.1

minx_v = -8
maxx_v = 8
miny_v = -8
maxy_v = 8

animate_lambda = lambda i : animate(i, start_frame, frame_infos, [minx_v, maxx_v, miny_v, maxy_v])

fig, ax = plt.subplots()
fig.set_dpi(240)

print('preparing video...')
ani = FuncAnimation(fig, animate_lambda, frames=end_frame-start_frame, interval=10, repeat=False)

record = False
if record:
    # saving to m4 using ffmpeg writer
    print('saving video...')
    writervideo = FFMpegWriter(fps=60)
    ani.save('detection_viewer_no_fix.mp4', writer=writervideo)
    plt.close()
else:
    plt.show()