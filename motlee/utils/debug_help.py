import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot

# def print_cone_debug():
# cone_str = 
                
                # print('-') 
                # print(f'  framenum: {framenum}')
                # print(f'  T:')
                # print(f'    heading: {heading}')
                # print(f'    translation: {translation}')
                # print(f'  T_est:')
                # print(f'    heading: {heading}')
                # print(f'    translation: {translation}')
                # print(f'  cones:')
                # print(cones)
# def T2tpsi(T):
#     t = T[:,:3].reshape((3, 1))
#     psi = Rot.from_matrix(T[:3, :3]).as_euler('z', degress=True)
#     return t, psi

def T2t(T):
    t = T[:3,3].reshape(-1).tolist()
    return t

def T2psi(T):
    psi = Rot.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True).item(2)
    return psi 

def get_transform_dict(T, raw=False):
    d = dict()
    if raw: d['raw'] = T.reshape(-1).tolist()
    d['heading'] = T2psi(T)
    d['translation'] = T2t(T)
    return d

def get_true_est_transform_dict(frametime, detections):
    T = detections.T_WC(frametime, T_BC=np.eye(4), true_pose=True)
    T_est = detections.T_WC(frametime, T_BC=np.eye(4), true_pose=False)
    d = dict()
    d['T'] = get_transform_dict(T)
    d['T_est'] = get_transform_dict(T_est)
    return d
           
def get_cone_debug_dict(frametime, cones, detections):
    d = get_true_est_transform_dict(frametime, detections)
    d['frametime'] = frametime
    d['cones'] = []
    for cone in cones:
        d['cones'].append(cone.tolist())
    return d

def get_map_debug_dict(frametime, mot, detections):
    cones = []
    for cone in mot.cones:
        cones.append(cone.state[:2, :].reshape(-1).tolist() + [0])

    d = get_true_est_transform_dict(frametime, detections)
    d['frametime'] = frametime
    d['cones'] = cones
    return d

def get_realign_debug_dict(frametime, mot1, mot2, detections1, detections2, T_fix):
    cones1 = [cone.state[:2, :].reshape(-1).tolist() + [0] for cone in mot1.cones]
    cones2 = [cone.state[:2, :].reshape(-1).tolist() + [0] for cone in mot2.cones]
    ages1 = [[cone.ell] for cone in mot1.cones]
    ages2 = [[cone.ell] for cone in mot2.cones]

    d = dict()
    d['local'] = mot1.camera_id
    d['neighbor'] = mot2.camera_id
    d['rovers'] = list()
    d['rovers'].append(get_true_est_transform_dict(frametime, detections1))
    d['rovers'].append(get_true_est_transform_dict(frametime, detections2))
    d['cones'] = [cones1, cones2]
    d['ages'] = [ages1, ages2]
    d['T_fix_est'] = mot1.realigner.transforms[mot2.camera_id].reshape(-1).tolist()
    d['T_fix'] = T_fix.reshape(-1).tolist()
    d['frametime'] = frametime
    return d
    
def dump_everything_in_the_whole_world(frametime, framenum, rovers, mots, detections, gt_list):
    d = dict()
    d['frametime'] = frametime
    d['framenum'] = framenum
    d['rovers'] = dict()
    d['groundtruth'] = gt_list
    
    for r, m, det in zip(rovers, mots, detections):
        r_dict = dict()
        r_dict['T_WC'] = det.T_WC(frametime, T_BC=det.T_BC, true_pose=True).reshape(-1).tolist()
        r_dict['T_WC_bel'] = det.T_WC(frametime, T_BC=det.T_BC, true_pose=False).reshape(-1).tolist()
        r_dict['tracks'] = m.get_tracks(format='list')
        r_dict['T_fix'] = dict()
        for r_id, T in m.realigner.transforms.items():
            r_dict['T_fix'][r_id] = T.reshape(-1).tolist()
        d['rovers'][r] = r_dict
    return d

def dump_single_rover_mapping_tracks(frametime, framenum, rover, mot, cone_mapper, det):
    d = dict()
    d['frametime'] = frametime
    d['framenum'] = framenum
    d['rover'] = rover
    
    d['T_WC'] = det.T_WC(frametime, T_BC=det.T_BC, true_pose=True).reshape(-1).tolist()
    d['T_WC_bel'] = det.T_WC(frametime, T_BC=det.T_BC, true_pose=False).reshape(-1).tolist()
    d['tracks'] = mot.get_tracks(format='list')
    d['cones'] = cone_mapper.get_tracks(format='list')
    return d

def dump_mapping_info(frametime, framenum, rovers, mots, detections):
    d = dict()
    d['frametime'] = frametime
    d['framenum'] = framenum
    d['rovers'] = dict()
    
    for i, (r, m, det) in enumerate(zip(rovers, mots, detections)):
        r_dict = dict()
        r_dict['T_WC'] = det.T_WC(frametime, T_BC=det.T_BC, true_pose=True).reshape(-1).tolist()
        r_dict['T_WC_bel'] = det.T_WC(frametime, T_BC=det.T_BC, true_pose=False).reshape(-1).tolist()
        r_dict['cones'] = [cone.state[:2, :].reshape(-1).tolist() + [0] for cone in m.cones]
        r_dict['cones_cov'] = [cone.P[:2,:2].tolist() for cone in m.cones]
        r_dict['Tfix_hat'] = dict()
        r_dict['Tfix'] = dict()
        for r_id, T in m.realigner.transforms.items():
            if r_id < len(rovers): continue
            r_dict['Tfix_hat'][rovers[r_id - len(rovers)]] = T.reshape(-1).tolist()
            r_dict['Tfix'][rovers[r_id - len(rovers)]] = calc_Tfix(det, detections[r_id - len(rovers)], frametime).reshape(-1).tolist()
        d['rovers'][r] = r_dict
    return d

def calc_Tfix(det1, det2, frame_time):
    T_WC1_true = det1.T_WC(frame_time, T_BC=det1.T_BC, true_pose=True)
    T_WC1_bel = det1.T_WC(frame_time, T_BC=det1.T_BC, true_pose=False)
    T_WC2_true = det2.T_WC(frame_time, T_BC=det2.T_BC, true_pose=True)
    T_WC2_bel = det2.T_WC(frame_time, T_BC=det2.T_BC, true_pose=False)
    
    return inv(T_WC1_true @ inv(T_WC1_bel)) @ T_WC2_true @ inv(T_WC2_bel)

def dump_local_association(frametime, framenum, rovers, mots, detections, gt_list):
    d = dict()
    d['frametime'] = frametime
    d['framenum'] = framenum
    d['rovers'] = dict()
    d['groundtruth'] = gt_list
    
    for r, m, det in zip(rovers, mots, detections):
        r_dict = dict()
        r_dict['T_WC'] = det.T_WC(frametime, T_BC=det.T_BC, true_pose=True).reshape(-1).tolist()
        r_dict['T_WC_bel'] = det.T_WC(frametime, T_BC=det.T_BC, true_pose=False).reshape(-1).tolist()
        r_dict['local_da'] = m.track_debug_info
        d['rovers'][r] = r_dict
    return d
