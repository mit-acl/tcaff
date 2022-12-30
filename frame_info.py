import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from bagpy import bagreader
import pandas as pd

class FrameInfo():
    
    def __init__(self, T_BC, frame_file, pose_topic, detection_topic, sigma_r=0, sigma_t=0): 

        # Extract data from bag files       
        b = bagreader(frame_file)
        annot_csv = b.message_by_topic(detection_topic)
        annot_df = pd.read_csv(annot_csv)
        object_lists = annot_df.detections.values.tolist()
        object_times = annot_df.Time.values.tolist()
        pose_csv = b.message_by_topic(pose_topic)
        pose_df = pd.read_csv(pose_csv, usecols=['Time', 'pose.position.x', 'pose.position.y', 'pose.position.z',
            'pose.orientation.x', 'pose.orientation.y', 'pose.orientation.z', 'pose.orientation.w'])
        positions = pd.DataFrame.to_numpy(pose_df.iloc[:, 1:4])
        orientations = pd.DataFrame.to_numpy(pose_df.iloc[:, 4:8])
        pose_times = pose_df.iloc[:,0]
        
        self.data = []
        self.num_frames = len(object_lists)
        # Iterate across each frame
        for i, objects in enumerate(object_lists):
            time_indices = np.where(pose_times >= object_times[i])[0]
            if len(time_indices) == 0:
                break
            curr_position = positions[time_indices[0],:]
            curr_orientation = orientations[time_indices[0],:]
            T_WB = np.eye(4)
            T_WB[:3,:3] = Rot.from_quat(curr_orientation).as_matrix()
            T_WB[:3,3] = curr_position
            T_WC = T_WB @ T_BC
            R = T_WC[:3,:3]
            T = T_WC[:3,3].reshape((3,1))
            np.set_printoptions(precision=2, suppress=True)
            # Only use this for setting initial setup
            # TODO Change this
            if i == 0:
                self.R = R
                self.T = T
            objects = objects.split('centertrack_id: ')
            if objects[0] == '[]':
                self.data.append({'bbox2d': [], 'pos3d': []})
                continue
            bbox2d = []
            pos3d = []
            for obj in objects:
                if obj == '[':
                    continue
                bbox2d.append(self._list_from_str('bbox2d: ', 'extent: ', obj))
                pos3d_new = []
                pos3d_new.append(float(obj.split('x: ')[1].split('y: ')[0].strip()))
                pos3d_new.append(float(obj.split('y: ')[1].split('z: ')[0].strip()))
                pos3d_new.append(float(obj.split('z: ')[1].split('orientation: ')[0].strip()))
                pos3d_new = np.array(pos3d_new).reshape((3,1))
                if sigma_r != 0 or sigma_t != 0:
                    # TODO: Adding subsequent rotations isn't really Gaussian, if small should be close enough?
                    # Maybe using rotvec?
                    R_noise = Rot.from_euler('xyz', [np.random.normal(0, sigma_r), np.random.normal(0, sigma_r), np.random.normal(0, sigma_r)]).as_matrix()
                    t_noise = np.array([np.random.normal(0, sigma_t), np.random.normal(0, sigma_t), np.random.normal(0, sigma_t)]).reshape((3,1))
                    pos3d_new = R_noise @ R @ pos3d_new + T + t_noise
                else:
                    pos3d_new = R @ pos3d_new + T
                pos3d.append(pos3d_new)
            self.data.append({'bbox2d': bbox2d, 'pos3d': pos3d})
    
    def cam_pos(self):
        return self.T
            
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
    
    ########## Set up cameras ############
    num_cams = 4
    Rvec0 = np.array([1.9007833770e+00, 4.9730769727e-01, 1.8415452559e-01])
    Rvec1 = np.array([1.9347282363e+00, -7.0418616982e-01, -2.3783238362e-01])
    Rvec2 = np.array([-1.8289537286e+00, 3.7748154985e-01, 3.0218614321e+00])
    Rvec3 = np.array([-1.8418460467e+00, -4.6728290805e-01, -3.0205552749e+00])
    R0 = Rot.from_euler('xyz', 
        Rvec0, degrees=False).as_matrix()
    R1 = Rot.from_euler('xyz', 
        Rvec1, degrees=False).as_matrix()
    R2 = Rot.from_euler('xyz', 
        Rvec2, degrees=False).as_matrix()
    R3 = Rot.from_euler('xyz', 
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

def get_static_test_frame_info(run=1, sigma_r=0, sigma_t=0):

    ########## Setu p cameras ############
    # T_WC = T_WB @ self.T_BC
    num_cams = 5
    T_BC_01 = np.array([0.0, 0.01919744239968966, 0.9998157121216441, 0.077, -1.0, 0.0, 0.0, 0.28, 0.0, -0.9998157121216442, 0.019197442399689665, -0.06999999999999999, 0.0, 0.0, 0.0, 1.0]).reshape(4,4)
    T_BC_04 = np.array([0.022684654329523025, -0.02268573610666359, 0.9994852494335512, 0.077, -0.9997426373806936, -0.00025889700461809384, 0.022684619799232468, 0.03, -0.0002558535612070688, -0.9997426120505415, -0.02268577063525499, -0.09, 0.0, 0.0, 0.0, 1.0]).reshape(4,4)
    T_BC_05 = np.array([-0.0174226746835367, -0.05495036561082756, 0.9983370711969513, 0.07769690336094144, -0.9998476997771804, -5.4828602412673115e-05, -0.017452055583951746, 0.32999512637861894, 0.0010137342613491236, -0.998489085725558, -0.05494104139699781, -0.07004079327681155, 0.0, 0.0, 0.0, 1.0]).reshape(4,4)
    T_BC_06 = np.array([-0.01050309850203959, -0.0015231189739128672, 0.9999436809293045, 0.07672078798849649, -0.999718957839354, 0.02127014830982589, -0.010468339289214947, 0.24999506647546407, -0.021253005868642767, -0.9997726045953996, -0.0017460933761166166, -0.05000935206550797, 0.0, 0.0, 0.0, 1.0]).reshape(4,4)
    T_BC_08 = np.array([-0.07500987988876373, -0.034584016329649164, 0.9965828935585749, 0.07672078798849649, -0.9969596158127689, 0.023743789411150146, -0.07421426347310038, 0.24999506647546407, -0.021096027055562884, -0.9991197016768858, -0.03625988642510408, -0.05000935206550797, 0.0, 0.0, 0.0, 1.0]).reshape(4,4)
    T_BCs = [T_BC_01, T_BC_04, T_BC_05, T_BC_06, T_BC_08]
    rover_nums = ['01', '04', '05', '06', '08']
   
    # Rs = [R0, R1, R2, R3]
    # Ts = [R0.T @ T0, R1.T @ T1, R2.T @ T2, R3.T @ T3]
    
    fis = []
    for i, rover_num in zip(range(num_cams), rover_nums[:num_cams]):        
        fis.append(FrameInfo(T_BCs[i],
            f'/home/masonbp/ford-project/data/static-20221216/centertrack_detections/run0{run}_RR{rover_num}.bag', 
            f'/RR{rover_num}/world',
            f'/RR{rover_num}/detections', 
            sigma_r=sigma_r, sigma_t=sigma_t))
        
    return fis

if __name__ == '__main__':
        
    fis = get_static_test_frame_info()