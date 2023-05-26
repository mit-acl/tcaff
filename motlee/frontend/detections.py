import os
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot
import pandas as pd
import gtsam
import glob
import yaml

import motlee.config.data_params as PARAMS
from motlee.utils.transform import transform
from motlee.utils.cam_utils import pixel2groundplane

def csv2poses(csvfile):
    pose_df = pd.read_csv(csvfile, usecols=['header.stamp.secs', 'header.stamp.nsecs', 'pose.position.x', 'pose.position.y', 'pose.position.z',
        'pose.orientation.x', 'pose.orientation.y', 'pose.orientation.z', 'pose.orientation.w'])
    positions = pd.DataFrame.to_numpy(pose_df.iloc[:, 2:5])
    orientations = pd.DataFrame.to_numpy(pose_df.iloc[:, 5:9])
    pose_times = pd.DataFrame.to_numpy(pose_df.iloc[:,0:1]) + pd.DataFrame.to_numpy(pose_df.iloc[:,1:2])*1e-9
    return positions, orientations, pose_times

class Detections():
    
    def __init__(self, rover, T_BC, csv_dir, use_noisy_odom=False, register_time=0.0, sigma_r=0, sigma_t=0): 
        self.data = []
        self.time_diff_allowed = .4 # TODO: tunable parameter...?
        
        noisy_odom_addon = 't265_registered-' if use_noisy_odom else ''
        csv_annot = f'{csv_dir}/{rover}-detections.csv'
        csv_true_pose = f'{csv_dir}/{rover}-world.csv'
        csv_bel_pose = f'{csv_dir}/{rover}-{noisy_odom_addon}world.csv'
        

        # Extract data from bag files       
        annot_df = pd.read_csv(csv_annot, usecols= \
            ['header.stamp.secs', 'header.stamp.nsecs', 'detections'])
        object_lists = annot_df.detections.values.tolist()
        self.times = pd.DataFrame.to_numpy(annot_df.iloc[:,0:1]) + pd.DataFrame.to_numpy(annot_df.iloc[:,1:2])*1e-9
        positions_bel, orientations_bel, pose_times_bel = csv2poses(csv_bel_pose)
        positions_true, orientations_true, pose_times_true = csv2poses(csv_true_pose)
        
        self.num_frames = len(object_lists)
        
        self.T_offset = np.eye(4)
        if sigma_r != 0:
            # TODO: Adding subsequent rotations isn't really Gaussian, if small should be close enough?
            # Maybe using rotvec?
            R_offset = Rot.from_euler('z', np.random.normal(0, sigma_r)).as_matrix()
            self.T_offset[:3, :3] = R_offset
        if sigma_t != 0:
            t_offset = np.array([np.random.normal(0, sigma_t/np.sqrt(2)), np.random.normal(0, sigma_t/np.sqrt(2)), 0.0]).reshape(-1)
            self.T_offset[:3, 3] = t_offset
        self.T_BC = T_BC
        self.T_registered = None
        
        # Iterate across each frame
        start_time = self.times.item(0)
        for i, objects in enumerate(object_lists):
            T_WB_bel = self.find_T_WB(self.times.item(i), pose_times_bel, positions_bel, orientations_bel)
            T_WB_true = self.find_T_WB(self.times.item(i), pose_times_true, positions_true, orientations_true)
            if T_WB_bel is None or T_WB_true is None:
                break
            if self.times.item(i) - start_time < register_time:
                T_WB_bel = T_WB_true
            elif self.T_registered is None:
                self.T_registered = T_WB_true @ inv(T_WB_bel)
                T_WB_bel = T_WB_true
            else:
                T_WB_bel = self.T_registered @ T_WB_bel
            T_WC_bel = T_WB_bel @ T_BC
            # Apply offset (a little weird, but I think this is right as opposed to T_offset @ T_WC_bel)
            T_WC_bel[0:3, 0:3] = self.T_offset[0:3, 0:3] @ T_WC_bel[0:3, 0:3]
            T_WC_bel[0:3, 3] = self.T_offset[0:3, 3] + T_WC_bel[0:3, 3]
            T_WC_true = T_WB_true @ T_BC
            
            np.set_printoptions(precision=2, suppress=True)
            objects = objects.split('centertrack_id: ')
            if objects[0] == '[]':
                self.data.append({'time': self.times.item(i), 'bbox2d': [], 'pos3d': [], 'T_WB_bel': T_WB_bel, 'T_WB_true': T_WB_true})
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
                # pos3d_new_W_bel_test = R_offset @ T_WC_true[0:3, 0:3] @ pos3d_new + T_WC_true[0:3, 3].reshape((3, 1)) + t_offset.reshape((3,1))
                # TODO: am I doing this right?
                pos3d_new_W_bel = transform(T_WC_bel, pos3d_new)
                pos3d.append(pos3d_new_W_bel)
            self.data.append({'time': self.times.item(i), 'bbox2d': bbox2d, 'pos3d': pos3d, 'T_WB_bel': T_WB_bel, 'T_WB_true': T_WB_true})

    def find_T_WB(self, time, pose_times, positions, orientations):
        time_indices = np.where(pose_times >= time)[0]
        if len(time_indices) == 0:
            return None
        curr_position = positions[time_indices[0],:]
        curr_orientation = orientations[time_indices[0],:]
        T_WB = np.eye(4)
        T_WB[:3,:3] = Rot.from_quat(curr_orientation).as_matrix()
        T_WB[:3,3] = curr_position
        return T_WB
    
    def T_WC(self, time, T_BC=None, true_pose=True):
        if T_BC is None:
            T_BC = self.T_BC
        idx = self.idx(time)
        if true_pose:
            return self.data[idx]['T_WB_true'] @ T_BC
        else:
            return self.data[idx]['T_WB_bel'] @ T_BC
            
    def at(self, time):
        idx = self.idx(time)
        if abs(self.times[idx] - time) < self.time_diff_allowed:
            return self.data[idx]
        else:
            return dict()
    
    def pos(self, time):
        idx = self.idx(time)
        if abs(self.times[idx] - time) < self.time_diff_allowed:
            return self.data[idx]['pos3d']
        else:
            return list()
    
    def bbox(self, time):
        idx = self.idx(time)
        if abs(self.times[idx] - time) < self.time_diff_allowed:
            return self.data[idx]['bbox2d']
        else:
            return list()

    def time(self, idx):
        return self.data[idx]['time']
    
    def idx(self, time):
        op1_exists = np.where(self.times >= time)[0].shape[0]
        op2_exists = np.where(self.times <= time)[0].shape[0]
        if not op1_exists and not op2_exists:
            return None
        if op1_exists:
            op1 = np.where(self.times >= time)[0][0]
        if op2_exists:
            op2 = np.where(self.times <= time)[0][-1]
        if not op1_exists: return op2
        if not op2_exists: return op1
        if abs(time - self.times[op1]) < abs(time - self.times[op2]):
            return op1
        else:
            return op2

    def _list_from_str(self, start_indicator, end_indicator, obj):
        list_of_str = obj.split(start_indicator)[1].split(end_indicator)[0].replace('[', '').replace(']', '').strip().split(', ')
        list_num = []
        for num_str in list_of_str:
            list_num.append(float(num_str))
        return list_num

class GroundTruth():

    def __init__(self, csv_dir, ped_list=[*range(1, 6)]):
        self.positions = dict()
        self.orientations = dict()
        self.times = dict()
        self.time_tol = .2
        self.ped_list = ped_list

        for ped in self.ped_list[:]:
            try:
                ped_csv = f'{csv_dir}/PED{ped}-world.csv'
                self.positions[ped], self.orientations[ped], self.times[ped] = csv2poses(ped_csv)
            except:
                self.ped_list.remove(ped)
                continue
            
    def ped_positions(self, time):
        positions = []
        ped_list = []
        for ped_id in self.ped_list:
            time_indices = np.where(self.times[ped_id] >= time)[0]
            if len(time_indices) == 0:
                continue
            if abs(self.times[ped_id][time_indices[0]] - time) > self.time_tol:
                continue
            positions.append(self.positions[ped_id][time_indices[0]])
            ped_list.append(ped_id)
        return ped_list, positions
    
    
class ConeDetections():
    
    def __init__(self, yamlfile, T_BC):
        with open(yamlfile, 'r') as f:
            self.detections = yaml.full_load(f)
        self.times = np.array([x['time'] for x in self.detections])
        self.T_BC = T_BC
        
            
    def detection3d(self, frametime, T_WC):
        idx = np.where(self.times >= frametime)[0][0]
        frame_dets_c = self.detections[idx]['detections']
        frame_dets_w = [transform(T_WC, np.array(det).reshape((3, 1))) for det in frame_dets_c]
        return frame_dets_w
    
class ArtificialConeDetections():
    
    def __init__(self, K, T_BC, noisy=False):
        self.K = K
        with open('/home/masonbp/ford-project/data/dynamic-final/cone_sample.yaml', 'r') as f:
            markers = yaml.full_load(f)['markers']
        self.cones = []
        for m in markers:
            pos = m['position']
            self.cones.append([pos['x'], pos['y'], pos['z']])
        self.cones = np.array(self.cones).reshape((-1, 3))
        self.width = 1280
        self.height = 720
        self.T_BC = T_BC
        self.noisy = noisy

    def detection3d(self, idx, T_WC, T_WC_bel):
        #
        # draw out FOV lines
        #
        #              C
        #             / \
        #            p1  p0
        #
        p0 = np.array([0, self.height, 1]).reshape((3, 1))
        p1 = np.array([self.width, self.height, 1]).reshape((3, 1))
        p03d = pixel2groundplane(self.K, T_WC, p0)
        p13d = pixel2groundplane(self.K, T_WC, p1)
        t = T_WC[:3, 3]
        
        l0_slope = (p03d.item(1) - t.item(1)) / (p03d.item(0) - t.item(0))
        l0_intrcpt = p03d.item(1) - l0_slope*p03d.item(0)
        l1_slope = (p13d.item(1) - t.item(1)) / (p13d.item(0) - t.item(0))
        l1_intrcpt = p13d.item(1) - l1_slope*p13d.item(0)
        
        # if p03d is to the right of c, cone must be less than
        if p03d.item(0) > t.item(0):
            l0_check = lambda x : x.item(1) <= l0_slope * x.item(0) + l0_intrcpt
        else:
            l0_check = lambda x : x.item(1) >= l0_slope * x.item(0) + l0_intrcpt
            
        # if p13d is to the right of c, cone must be greater than
        if p13d.item(0) > t.item(0):
            l1_check = lambda x : x.item(1) >= l1_slope * x.item(0) + l1_intrcpt
        else:
            l1_check = lambda x : x.item(1) <= l1_slope * x.item(0) + l1_intrcpt
            
        seeable_cones = []
        for i in range(self.cones.shape[0]):
            cone = self.cones[i,:].reshape((3, 1))
            if l0_check(cone) and l1_check(cone):
                transformed_cone = transform(T_WC_bel @ np.linalg.inv(T_WC), cone)
                seeable_cones.append(transformed_cone)
                
        sigma = .5
        if self.noisy:
            for i in range(len(seeable_cones)):
                t_offset = np.array([np.random.normal(0, sigma/np.sqrt(2)), np.random.normal(0, sigma/np.sqrt(2)), 0.0]).reshape(seeable_cones[i].shape)
                seeable_cones[i] = t_offset + seeable_cones[i]
        return seeable_cones
    
def intrinsic_calib(cam_name):
    if cam_name == 't265':
        fx = 285.72650146484375
        fy = 285.77301025390625
        cx = 425.5404052734375
        cy = 400.3540954589844      
        D = np.array([-0.006605582777410746, 0.04240882024168968, -0.04068116843700409, 0.007674722000956535])
        fisheye = True
    elif cam_name == 'd435':
        fx = 617.114013671875
        fy = 617.222412109375
        cx = 326.21502685546875
        cy = 246.99037170410156
        D = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        fisheye = False
    elif cam_name == 'l515':
        fx = 901.47021484375
        fy = 901.8353881835938
        cx = 649.6925048828125
        cy = 365.004150390625
        D = np.array([0.19411328434944153, -0.5766593217849731, -0.001459299004636705, 0.0013330483343452215, 0.5116369724273682])
        fisheye = False
    intrinsics = ([fx, fy, cx, cy])
    K = np.array([[fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]])
    return K

def get_rover_detections(csv_dir, register_time, sigma_r=0.0, sigma_t=0.0, 
        rovers=['RR01', 'RR04', 'RR05', 'RR06', 'RR08'], cam_type='t265', use_noisy_odom=False): 
    if 'dynamic' in csv_dir:
        calibrations_exist = os.path.isfile(PARAMS.CAMERA_CALIB_FILE)
        
        if calibrations_exist:
            with open(PARAMS.CAMERA_CALIB_FILE, 'r') as f:
                if cam_type not in yaml.full_load(f):
                    calibrations_exist = False
        if not calibrations_exist:
            assert False, f'cannot find calibration file: {PARAMS.CAMERA_CALIB_FILE}'
            # Deprecated support for not including calibration
        
        with open(PARAMS.CAMERA_CALIB_FILE, 'r') as f:
            T_BCs = yaml.full_load(f)[cam_type]
    else:
        ########## Set up cameras ############
        T_BCs = dict()
        T_BCs['RR01'] = np.array([0.0, 0.01919744239968966, 0.9998157121216441, 0.077, -1.0, 0.0, 0.0, 0.28, 0.0, -0.9998157121216442, 0.019197442399689665, -0.06999999999999999, 0.0, 0.0, 0.0, 1.0]).reshape(4,4)
        T_BCs['RR04'] = np.array([0.022684654329523025, -0.02268573610666359, 0.9994852494335512, 0.077, -0.9997426373806936, -0.00025889700461809384, 0.022684619799232468, 0.03, -0.0002558535612070688, -0.9997426120505415, -0.02268577063525499, -0.09, 0.0, 0.0, 0.0, 1.0]).reshape(4,4)
        T_BCs['RR05'] = np.array([-0.0174226746835367, -0.05495036561082756, 0.9983370711969513, 0.07769690336094144, -0.9998476997771804, -5.4828602412673115e-05, -0.017452055583951746, 0.32999512637861894, 0.0010137342613491236, -0.998489085725558, -0.05494104139699781, -0.07004079327681155, 0.0, 0.0, 0.0, 1.0]).reshape(4,4)
        T_BCs['RR06'] = np.array([-0.01050309850203959, -0.0015231189739128672, 0.9999436809293045, 0.07672078798849649, -0.999718957839354, 0.02127014830982589, -0.010468339289214947, 0.24999506647546407, -0.021253005868642767, -0.9997726045953996, -0.0017460933761166166, -0.05000935206550797, 0.0, 0.0, 0.0, 1.0]).reshape(4,4)
        T_BCs['RR08'] = np.array([-0.07500987988876373, -0.034584016329649164, 0.9965828935585749, 0.07672078798849649, -0.9969596158127689, 0.023743789411150146, -0.07421426347310038, 0.24999506647546407, -0.021096027055562884, -0.9991197016768858, -0.03625988642510408, -0.05000935206550797, 0.0, 0.0, 0.0, 1.0]).reshape(4,4)
    
    detections = []
    for rover in rovers:
        detections.append(Detections(
            rover=rover,
            T_BC=np.array(T_BCs[rover]).reshape((4, 4)),
            csv_dir=csv_dir.format(rover),
            use_noisy_odom=use_noisy_odom,
            register_time=register_time,
            sigma_r=sigma_r, sigma_t=sigma_t))
        
    return detections

def get_cone_detections(yamlfile='/home/masonbp/ford/data/mot_dynamic/dynamic_motlee_iros/cone_detections/{}_detections.yaml', rovers=['RR01', 'RR04', 'RR06', 'RR08'], vicon=False):
    cone_detections = []
    with open(PARAMS.CAMERA_CALIB_FILE, 'r') as f:
        T_BCs = yaml.full_load(f)['l515']
    for rover in rovers:
        # cone_detections.append(
        #     ConeDetections(yamlfile=yamlfile.format(rover), K=intrinsic_calib('d435'), T_BC=T_BCs[rover])
        # )
        if vicon:
            cone_detections.append(
                ArtificialConeDetections(K=intrinsic_calib('l515'), T_BC=T_BCs[rover], noisy=False)
            )
        else:
            cone_detections.append(
                ConeDetections(yamlfile=yamlfile.format(rover), T_BC=np.array(T_BCs[rover]).reshape((4,4)))
            )
        
    return cone_detections

if __name__ == '__main__':
        
    # detections = get_rover_detections(bagfile='/home/masonbp/ford-project/data/dynamic-final/centertrack_detections/fisheye/run01_{}.bag',
    #                                   rovers=['RR01', 'RR04', 'RR06', 'RR08'], sigma_r=4.0*np.pi/180, sigma_t=0.5, cam_type='l515')
    get_cone_detections(rovers=['RR04'])
    # GT = GroundTruth('/home/masonbp/ford-project/data/static-20221216/run01_filtered.bag')