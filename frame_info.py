import os
import numpy as np
from scipy.spatial.transform import Rotation
from bagpy import bagreader
import pandas as pd

class FrameInfo():
    
    def __init__(self, R, T, frame_file, topic, sigma_r=0, sigma_t=0):
        # sigma_r = .4*np.pi/180
        # sigma_t = 0
        scaling = 37.5
        self.R = R
        self.T = T * -scaling/1000
        
        b = bagreader(frame_file)
        # print(frame_file)
        # print(topic)
        annot_csv = b.message_by_topic(topic)
        annot_df = pd.read_csv(annot_csv)
        object_lists = annot_df.objects.values.tolist()
        
        self.data = []
        self.num_frames = len(object_lists)
        for objects in object_lists:
            objects = objects.split(', id: ')
            if objects[0] == '[]':
                self.data.append({'bbox2d': [], 'pos3d': []})
                continue
            bbox2d = []
            pos3d = []
            for obj in objects:
                bbox2d.append(self._list_from_str('bbox_2d: ', 'dim: ', obj))
                pos3d_new = np.array(self._list_from_str('center_3d: ', 'theta: ', obj)).reshape((3,1))
                if sigma_r != 0 or sigma_t != 0:
                    # TODO: Adding subsequent rotations isn't really Gaussian, if small should be close enough?
                    # Maybe using rotvec?
                    R_noise = Rotation.from_euler('xyz', [np.random.normal(0, sigma_r), np.random.normal(0, sigma_r), np.random.normal(0, sigma_r)]).as_matrix()
                    t_noise = np.array([np.random.normal(0, sigma_t), np.random.normal(0, sigma_t), np.random.normal(0, sigma_t)]).reshape((3,1))*scaling
                    pos3d_new = R_noise @ self.R.T @ pos3d_new * scaling + self.T + t_noise
                else:
                    pos3d_new = self.R.T @ pos3d_new * scaling + self.T
                pos3d_new[0] = pos3d_new[0] + 40
                pos3d_new[1] = -pos3d_new[1] + 375
                pos3d.append(pos3d_new)
            self.data.append({'bbox2d': bbox2d, 'pos3d': pos3d})
    
    def cam_pos(self):
        # return self.T
        altered_t = [self.T[0], self.T[1], self.T[2]]
        altered_t[0] = altered_t[0] + 40
        altered_t[1] = -altered_t[1] + 375
        return altered_t
            
    def at(self, idx):
        return self.data[idx]
    
    def pos(self, idx):
        return self.data[idx]['pos3d']
    
    def bbox(self, idx):
        return self.data[idx]['bbox2d']

    def _list_from_str(self, start_indicator, end_indicator, obj):
        list_of_str = obj.split(start_indicator)[1].split(end_indicator)[0].replace('[', '').replace(']', '').strip().split(', ')
        list_num = []
        for num_str in list_of_str:
            list_num.append(float(num_str))
        return list_num

def get_epfl_frame_info(sigma_r=0, sigma_t=0):
    
    ########## Setu p cameras ############
    num_cams = 4
    Rvec0 = np.array([1.9007833770e+00, 4.9730769727e-01, 1.8415452559e-01])
    Rvec1 = np.array([1.9347282363e+00, -7.0418616982e-01, -2.3783238362e-01])
    Rvec2 = np.array([-1.8289537286e+00, 3.7748154985e-01, 3.0218614321e+00])
    Rvec3 = np.array([-1.8418460467e+00, -4.6728290805e-01, -3.0205552749e+00])
    R0 = Rotation.from_euler('xyz', 
        Rvec0, degrees=False).as_matrix()
    R1 = Rotation.from_euler('xyz', 
        Rvec1, degrees=False).as_matrix()
    R2 = Rotation.from_euler('xyz', 
        Rvec2, degrees=False).as_matrix()
    R3 = Rotation.from_euler('xyz', 
        Rvec3, degrees=False).as_matrix()

    scaling = 1
    T0 = np.array([[-4.8441913843e+03, 5.5109448682e+02, 4.9667438357e+03]]).T * scaling
    T1 = np.array([[-65.433635, 1594.811988, 2113.640844]]).T * scaling
    T2 = np.array([[1.9782813424e+03, -9.4027627332e+02, 1.2397750058e+04]]).T * scaling
    T3 = np.array([[4.6737509054e+03, -2.5743341287e+01, 8.4155952460e+03]]).T * scaling

    Rs = [R0, R1, R2, R3]
    Ts = [R0.T @ T0, R1.T @ T1, R2.T @ T2, R3.T @ T3]
    
    fis = []
    for i in range(num_cams):
        # print(i)
        fis.append(FrameInfo(Rs[i], Ts[i], f'detections/cam{i}.bag', f'/camera{i}/centertrack/annotations', sigma_r=sigma_r, sigma_t=sigma_t))
        
    return fis

if __name__ == '__main__':
        
    fis = get_epfl_frame_info()
    
    print(fis[0].at(0))
    print(fis[1].at(0))
    print(fis[2].at(0))
    print(fis[3].at(0))
    



# detections_dir = 'detections'
# processed_detections_dir = 'processed_detections'

# scaling = -1000

# try:
#     os.mkdir(processed_detections_dir)
# except:
#     print('folder already exists... not creating folder')



# ########### Set up detections ############
# detections = [cam0.detections, cam1.detections, cam2.detections, cam3.detections]
# for i, cam in enumerate(detections):
#     for frame in cam:
#         for j in range(len(frame)):
#             det = frame[j]
#             pos = np.array(det[1]).reshape(3,1)
#             frame[j] = scaling * Rs[i].T @ pos + Ts[i]
#     with open(f'{processed_detections_dir}/cam{i}.py', 'w') as f:
#         print('import numpy as np', file=f)
#         print('detections = [', file=f)
#         for frame in cam:
#             print('\t[', file=f)
#             for det in frame:
#                 det_str = f'np.array([[{det.item(0)}, {det.item(1)}, {det.item(2)}]]).T'
#                 print(f'\t\t{det_str},', file=f)
#             print('\t],', file=f)
#         print(']', file=f)