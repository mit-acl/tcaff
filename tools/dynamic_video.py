import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot
import cv2 as cv
import sys

sys.path.append('../src')
from utils.transform import transform

############# OPTIONS #################3
RECORD = True
VIDEO = False
ROVER_NUM = 2
ROVER = ['RR04', 'RR06', 'RR08'][ROVER_NUM]

if len(sys.argv) > 1:
    datafile1 = sys.argv[1]
else:
    datafile1 = '/home/masonbp/ford-project/data/motlee/cam_project_test_realign.json'
    datafile2 = '/home/masonbp/ford-project/data/motlee/cam_project_test_nofix.json'
video_out = '/home/masonbp/ford-project/data/figure1.mp4'
video_in = f'/home/masonbp/ford-project/data/dynamic-final/videos/l515/run1_{ROVER}.avi'

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
        self.fig, (self.axs) = plt.subplots(2, 2)
        self.fig.set_dpi(240)
        self.xlim = [-8., 8.]
        self.ylim = [-8., 8.]

        self.data = []
        for df in datafiles:
            f = open(df)
            self.data.append(json.load(f))
        self.cap = cv.VideoCapture(vidfile)
        self.rover = rover
        self.K = params['K']
        
    def view(self, framenum):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, framenum-1)
        res, frame_orig = self.cap.read()
        self.fig.suptitle(f'Frame: {framenum}')

        # haha
        for i, datum in enumerate(self.data):
            # new plot setup
            self.axs[0, i].clear()
            self.axs[0, i].set_xlim(self.xlim)
            self.axs[0, i].set_ylim(self.ylim)
            self.axs[0, i].grid(True)
            frame = np.copy(frame_orig)

            df_idx = None
            for j, df in enumerate(datum):
                if df['framenum'] == framenum:
                    df_idx = j
                    break
            df = datum[df_idx]
            
            for gt in df['groundtruth']:
                self.axs[0, i].plot(gt[0], gt[1], 'o', color='green')

            rover = df['rovers'][self.rover]
            for track in rover['tracks']:
                T_WC = np.array(rover['T_WC']).reshape((4,4))
                T_WC_bel = np.array(rover['T_WC_bel']).reshape((4,4))

                track_corrected = transform(T_WC @ inv(T_WC_bel), np.array(track + [0]))
                track_corrected[2] = 0
                self.axs[0, i].plot(track_corrected[0], track_corrected[1], 'x', color='blue')

                # T_WC_bel[3:2] = T_WC[3:2]
                # track_c = transform(inv(T_WC_bel), np.array(track + [0])).reshape((3,1))
                track_c = transform(inv(T_WC), track_corrected)
                if track_c.item(2) > 0:
                    uvs = (self.K @ track_c)
                    uvs /= uvs.item(2)
                    pixel_coords = uvs.reshape(-1)[:2]
                    pixel_coords /= 1
                    cv.drawMarker(frame, tuple(pixel_coords[:].astype(int).tolist()), self.get_track_color(0), cv.MARKER_TILTED_CROSS, thickness=10, markerSize=90)
            
            self.axs[1, i].imshow(frame[...,::-1])

        
        return

    def get_track_color(self, i):
        c = np.array(plt.get_cmap('tab10').colors[i])
        c = (c * 255).astype(int)
        return tuple(v.item() for v in c[::-1])


viewer = Viewer([datafile1, datafile2], video_in, ROVER)
# viewer.single_frame_view(150)
start_frame = 125*30 #3750 #120*30
fps = 5
num_frames = (180-125) * fps
# viewer.view(4200)
ani = FuncAnimation(viewer.fig, lambda i: viewer.view(i*fps + start_frame), frames=num_frames, interval=1, repeat=False)    

if RECORD:
    # saving to m4 using ffmpeg writer
    print('saving video...')
    writervideo = FFMpegWriter(fps=int(fps))
    ani.save(video_out, writer=writervideo)
    plt.close()
else:
    plt.show()