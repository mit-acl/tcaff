import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import sys

sys.path.append('/home/masonbp/ford-project/dkfmot/src/')
from utils.transform import transform
from realign.realign_frames import my_realign_cones

############# OPTIONS #################3
RECORD = True
GT = True
ARROW_SCALE = 3.0

# filename = '/home/masonbp/ford-project/data/mot_metrics/dynamic/2_rover_debug/single_rover_debug.yaml'
# filename = '/home/masonbp/ford-project/data/mot_metrics/dynamic/2_rover_debug/single_rover_debug_map.yaml'
filename = '/home/masonbp/ford-project/data/mot_metrics/dynamic-final/debug_files/big_R_8_big_realign_means_big_R_too.yaml'
video_file = '/home/masonbp/ford-project/data/mot_metrics/dynamic-final/debug_files/debug.mp4'

with open(filename, 'r') as f:
    data = yaml.full_load(f)

fig, (axs) = plt.subplots(1, 2)
fig.set_dpi(240)

def apply_T_to_cones(T, cones):
    ret = np.zeros(cones.shape)
    for i in range(cones.shape[0]):
        ret[i,:] = transform(T, cones[i,:])
    return ret

def get_cone_gt():
    with open('/home/masonbp/ford-project/data/dynamic-final/cone_sample.yaml', 'r') as f:
        markers = yaml.full_load(f)['markers']
    gt = []
    for m in markers:
        pos = m['position']
        gt.append([pos['x'], pos['y'], pos['z']])
    gt = np.array(gt).reshape((-1, 3))
    return gt

first_time = data[0][0]['frametime']

class Viewer():

    def __init__(self, local_idx, neighbor_idx):
        self.last_T_fix = np.eye(4).reshape((16,))
        self.local_idx = local_idx
        self.neighbor_idx = neighbor_idx

    def single_frame_view(self, frame_idx):
        df_idx = None
        for i, realign in enumerate(data[frame_idx]):
            if realign['local'] == self.local_idx and realign['neighbor'] == self.neighbor_idx:
                df_idx = i
                break
        df = data[frame_idx][i]
        thing = df['frametime']
        fig.suptitle(f'{thing - first_time}')
        for ax_idx, ax in enumerate(axs):
            ax.clear()
            ax.set_xlim([-7, 7])
            ax.set_ylim([-7, 7])
            ax.set_aspect('equal')

            if ax_idx == 0: # true
                T_fix = df['T_fix']
                ax.set_title('Correct T_fix')
            else: # estimate
                T_fix = df['T_fix_est']
                ax.set_title('Estimated T_fix')
                ax.grid(not np.all(np.isclose(self.last_T_fix, T_fix)))
                self.last_T_fix = T_fix
                # actually
                T_fix = my_realign_cones(np.array(df['cones'][0]), np.array(df['cones'][1]), np.array(df['ages'][0]), np.array(df['ages'][1]), np.eye(4))
            T_fix = np.array(T_fix).reshape((4, 4))


            for r in df['rovers']:
                for T, color in zip(['T_est', 'T'], ['red', 'green']):
                    rover = np.array(r[T]['translation']).reshape((3,1))
                    psi = (r[T]['heading'] + 90) * np.pi / 180
                    rover_dir = np.concatenate([
                        rover + np.array([[np.cos(psi-30*np.pi/180), np.sin(psi-30*np.pi/180), 0]]).T,
                        rover, 
                        rover + np.array([[np.cos(psi+30*np.pi/180), np.sin(psi+30*np.pi/180), 0]]).T], axis=1)
                    ax.plot(rover.item(0), rover.item(1), color=color, marker='o')#, markerSize=15)
                    ax.plot(rover_dir[0,:], rover_dir[1,:], color='black')
            if GT:
                gt = get_cone_gt().T
                ax.scatter(gt[0,:], gt[1,:], color='lightgreen', marker='x')
            if df['cones'][0]:
                cones = np.array(df['cones'][0])
                ax.scatter(cones[:,0], cones[:,1])
            
            if df['cones'][1]:
                cones = np.array(df['cones'][1])
                cones = apply_T_to_cones(T_fix, cones)
                ax.scatter(cones[:,0], cones[:,1])
            ax.arrow(0, 0, T_fix[0,3]*ARROW_SCALE, T_fix[1,3]*ARROW_SCALE, color='black')
    
# single_frame_view(10)
# plt.show()

# for i in range(100):
#     T = np.array(data[i]['T']['raw']).reshape((4, 4))
#     R = Rot.from_matrix(T[:3,:3])
#     print(R.as_euler('xyz', degrees=True))

viewer = Viewer(4, 5)
ani = FuncAnimation(fig, viewer.single_frame_view, frames=len(data), interval=1, repeat=False)

if RECORD:
    # saving to m4 using ffmpeg writer
    print('saving video...')
    writervideo = FFMpegWriter(fps=10)
    vid_file = ''
    ani.save(video_file, writer=writervideo)
    plt.close()
else:
    plt.show()