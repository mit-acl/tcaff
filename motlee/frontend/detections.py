import os
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot
from bagpy import bagreader
import pandas as pd
import gtsam
import glob
import yaml

# if __name__ == '__main__':
#     from camera_calibration_mocap.cam_calib_mocap import get_calibrator
#     import sys
#     sys.path.append('..')
# else:
#     from .camera_calibration_mocap.cam_calib_mocap import get_calibrator
# from motlee.frontend.camera_calibration_mocap import get_calibrator
import motlee.config.data_params as PARAMS
from motlee.utils.transform import transform
from motlee.utils.cam_utils import pixel2groundplane

def bag2poses(bagfile, topic):
    pose_csv = bagfile.message_by_topic(topic)
    pose_df = pd.read_csv(pose_csv, usecols=['header.stamp.secs', 'header.stamp.nsecs', 'pose.position.x', 'pose.position.y', 'pose.position.z',
        'pose.orientation.x', 'pose.orientation.y', 'pose.orientation.z', 'pose.orientation.w'])
    positions = pd.DataFrame.to_numpy(pose_df.iloc[:, 2:5])
    orientations = pd.DataFrame.to_numpy(pose_df.iloc[:, 5:9])
    pose_times = pd.DataFrame.to_numpy(pose_df.iloc[:,0:1]) + pd.DataFrame.to_numpy(pose_df.iloc[:,1:2])*1e-9
    return positions, orientations, pose_times

class Detections():
    
    def __init__(self, T_BC, bagfile, pose_topic, detection_topic, register_time, sigma_r=0, sigma_t=0): 
        self.data = []
        self.time_diff_allowed = .4 # TODO: tunable parameter...?

        # Extract data from bag files       
        b = bagreader(bagfile, verbose=False)
        annot_csv = b.message_by_topic(detection_topic)
        annot_df = pd.read_csv(annot_csv, usecols= \
            ['header.stamp.secs', 'header.stamp.nsecs', 'detections'])
        object_lists = annot_df.detections.values.tolist()
        self.times = pd.DataFrame.to_numpy(annot_df.iloc[:,0:1]) + pd.DataFrame.to_numpy(annot_df.iloc[:,1:2])*1e-9
        positions_bel, orientations_bel, pose_times_bel = bag2poses(bagfile=b, topic=pose_topic)
        veh = pose_topic.split('/')[1]
        true_pose_topic = f'/{veh}/world'
        positions_true, orientations_true, pose_times_true = bag2poses(bagfile=b, topic=true_pose_topic)
        
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

    def __init__(self, bagfile, ped_list=[*range(1, 6)]):
        self.positions = dict()
        self.orientations = dict()
        self.times = dict()
        self.time_tol = .2
        self.ped_list = ped_list

        # Extract data from bag files       
        b = bagreader(bagfile, verbose=False)

        for ped in self.ped_list[:]:
            try:
                topic = f'/PED{ped}/world'
                self.positions[ped], self.orientations[ped], self.times[ped] = bag2poses(b, topic)
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
        fis.append(Detections(Rs[i], Ts[i], f'detections/cam{i}.bag', f'/camera{i}/centertrack/annotations', sigma_r=sigma_r, sigma_t=sigma_t))
        
    return fis

def get_rover_detections(bagfile, register_time, sigma_r=0.0, sigma_t=0.0, 
        rovers=['RR01', 'RR04', 'RR05', 'RR06', 'RR08'], cam_type='t265', rover_pose_topic='/world'): 
    if 'dynamic' in bagfile:
        tag_size = 0.1655
        T_tag_fix = gtsam.Pose3(gtsam.Rot3.Rz(-np.pi/2),np.array([0,0,0])).matrix()
        calibrations_exist = os.path.isfile(PARAMS.CAMERA_CALIB_FILE)
        
        if calibrations_exist:
            with open(PARAMS.CAMERA_CALIB_FILE, 'r') as f:
                if cam_type not in yaml.full_load(f):
                    calibrations_exist = False
        if not calibrations_exist:
            with open(PARAMS.CAMERA_CALIB_FILE, 'a') as f:
                print(f'{cam_type}:', file=f)
                for rover in rovers:
                    calibrator = get_calibrator(cam_name=cam_type)
                    if cam_type == 't265':
                        cam_topic = f'/t265/fisheye1/image_raw/compressed'
                    elif cam_type == 'd435':
                        cam_topic = f'/d435/color/image_raw/compressed'
                    elif cam_type == 'l515':
                        cam_topic = f'/l515/color/image_raw/compressed'
                    bagfiles = glob.glob(f'{PARAMS.DYNAMIC_DATA_DIR}/calib/{rover}/*.bag')
                    T_BC = calibrator.find_T_body_camera(
                        bagfiles=bagfiles,
                        camera_topic=f'/{rover}{cam_topic}',
                        tag_pose_topic='/tag_box/world',
                        camera_pose_topic=f'/{rover}/world',
                        tag_size=tag_size,
                        T_tag_fix=T_tag_fix,
                    )
                    print(f'  {rover}: [', file=f, end='')
                    for val in T_BC.reshape(-1).tolist():
                        print(f'{val}, ', file=f, end='')
                    print(f']', file=f)
        
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
            T_BC=np.array(T_BCs[rover]).reshape((4, 4)),
            bagfile=bagfile.format(rover), 
            pose_topic=f'/{rover}{rover_pose_topic}',
            detection_topic=f'/{rover}/detections', 
            register_time=register_time,
            sigma_r=sigma_r, sigma_t=sigma_t))
        
    return detections

def get_cone_detections(yamlfile='/home/masonbp/ford-project/data/dynamic-final/cone_detections/{}_detections.yaml', rovers=['RR01', 'RR04', 'RR06', 'RR08'], vicon=False):
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