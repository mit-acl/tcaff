import numpy as np
import argparse
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from motlee.mot.multi_object_tracker import MultiObjectTracker as MOT
from motlee.mot.motion_model import MotionModel
from motlee.tcaff.tcaff_manager import TCAFFManager
from motlee.utils.transform import xypsi_2_transform

from robot_utils.robot_data.robot_data import NoDataNearTimeException
from robot_utils.robot_data.pose_data import PoseData
from robot_utils.transform import T_FLURDF, T_RDFFLU, transform, transform_2_xytheta
from plot_utils import square_equal_aspect, plot_pose2d

try:
    from .motlee_data_processor import MOTLEEDataProcessor
    from .robot import Robot
    from .detections import DetectionData
    from .alignment_results import AlignmentResults
except:
    from motlee_data_processor import MOTLEEDataProcessor
    from robot import Robot
    from detections import DetectionData
    from alignment_results import AlignmentResults

KDM_CSV_OPTIONS = {'cols': {'time': ['#timestamp_kf'],
                                'position': ['x', 'y', 'z'],
                                'orientation': ['qx', 'qy', 'qz', 'qw']},
                    'col_nums': {'time': [0],
                                'position': [1,2,3],
                                'orientation': [5,6,7,4]},
                    'timescale': 1e-9}

parser = argparse.ArgumentParser()
parser.add_argument("--params", "-p", type=str, required=True)
parser.add_argument("--viz", "-v", action="store_true")

args = parser.parse_args()


# Load the parameters
with open(args.params, "r") as f:
    params = yaml.full_load(f)
    
robot_names = [params['robot1']['name'], params['robot2']['name']]
params[robot_names[0]] = params['robot1']
params[robot_names[1]] = params['robot2']
robots = []
pose_gt_data = dict()
pose_estimate_file_type = params['pose_estimate']['type']
show_passing_time = 'show_passing_time' in params and params['show_passing_time']

for name, other_name in zip(robot_names, robot_names[::-1]):

    pose_data = dict()
    for pose_type in ['pose_estimate', 'pose_gt']:
        if params[pose_type]['type'] == 'bag':
            pose_data[pose_type] = PoseData(
                data_file=f"{params[pose_type]['root']}/{name}.bag",
                file_type='bag',
                topic=f"/{name}/{params[pose_type]['topic']}",
                time_tol=10.,
                interp=True,
                T_postmultiply=T_RDFFLU if params[pose_type]['frame'] == 'RDF' else None,
                T_premultiply=xypsi_2_transform(*params[name]['premultiply'])
            )
        elif params[pose_type]['type'] == 'csv':
            pose_data[pose_type] = PoseData(
                data_file=f"{params[pose_type]['root']}/{name}.csv",
                file_type='csv',
                interp=True,
                csv_options=KDM_CSV_OPTIONS,
                time_tol=5.0,
                T_premultiply=xypsi_2_transform(*params[name]['premultiply'])
            )

    pose_est_data = pose_data['pose_estimate']
    pose_gt_data[name] = pose_data['pose_gt']

    dim = params['mapping']['dim']
    Q_el = params['mapping']['Q_el']
    P0_el = params['mapping']['P0_el']
    Q = np.diag([Q_el, Q_el, Q_el, params['mapping']['Q_el_w'], params['mapping']['Q_el_h']])
    P0 = np.diag([P0_el, P0_el, P0_el, params['mapping']['P0_el_w'], params['mapping']['P0_el_h']])
    landmark_model = MotionModel(A=np.eye(dim), H=np.eye(dim), Q=Q, R=np.array([]), P0=P0)

    mapper = MOT(
        camera_id=name,
        connected_cams=[other_name],
        track_motion_model=landmark_model,
        tau_global=0.,
        tau_local=params['mapping']['tau'],
        alpha=2000,
        kappa=params['mapping']['kappa'],
        nu=params['mapping']['nu'],
        track_storage_size=1,
        dim_association=dim
    )
    
    tcaff_manager = TCAFFManager(
        prob_no_match=params['tcaff']['prob_no_match'],
        exploring_branching_factor=params['tcaff']['exploring_branching_factor'],
        window_len=params['tcaff']['window_len'],
        max_branch_exp=params['tcaff']['max_branch_exp'],
        max_branch_main=params['tcaff']['max_branch_main'],
        rho=params['tcaff']['rho'],
        clipper_epsilon=params['tcaff']['clipper_epsilon'],
        clipper_sigma=params['tcaff']['clipper_sigma'],
        clipper_mult_repeats=params['tcaff']['clipper_mult_repeats'],
        max_obj_width=params['tcaff']['max_obj_width'],
        h_diff=params['tcaff']['h_diff'],
        wh_scale_diff=params['tcaff']['wh_scale_diff'],
        num_objs_req=params['tcaff']['num_objs_req'],
        max_opt_fraction=params['tcaff']['max_opt_fraction'],
        steps_before_main_tree_deletion=params['tcaff']['steps_before_main_tree_deletion'],
        main_tree_obj_req=params['tcaff']['main_tree_obj_req']
    )
    
    fastsam3d_detections = DetectionData(
        data_file=f"{params['fastsam3d_data']['root']}/{name}.json",
        file_type='json',
        time_tol=1.0,
        # T_BC=np.eye(4) if pose_estimate_file_type == 'bag' else T_FLURDF,
        T_BC=T_FLURDF if 'T_BC' not in params[name] else np.array(params[name]['T_BC']).reshape((4,4)),
        zmin=params['mapping']['zmin'],
        zmax=params['mapping']['zmax']
    )

    if params['synchronize_timing']:
        pose_gt_data[name].times -= params[name]['t_start']
        pose_est_data.times -= params[name]['t_start']
        fastsam3d_detections.times -= params[name]['t_start']

    robots.append(Robot(
        name=name,
        neighbors=[other_name],
        pose_est_data=pose_est_data,
        mapper=mapper,
        frame_align_filters={other_name: tcaff_manager},
        mot=None,
        fastsam3d_detections=fastsam3d_detections,
        person_detector_3d_detections=None
    ))

# Reset timing
if params['synchronize_timing']:    
    t0 = 0.0 #np.max([robot.pose_est_data.t0 for robot in robots])
    tf = np.min([params[name]['t_end'] - params[name]['t_start'] for name in robot_names])
else:
    if 't_start' in params:
        t0 = params['t_start']
    else:
        t0 = np.max([robot.pose_est_data.t0 for robot in robots]) + t_delay
    tf = np.min([robot.pose_est_data.tf for robot in robots])

for robot in robots:
    if 'start_odom_at_origin' in params and params['start_odom_at_origin']:
        robot.pose_est_data.T_premultiply = np.linalg.inv(robot.pose_est_data.T_WB(t0))
# for robot in robots:
#     robot.pose_est_data.set_t0(t0)
#     robot.fastsam3d_detections.set_t0(t0)
# tf = np.min([robot.pose_est_data.tf for robot in robots])
    
# Create the data processor
motlee_data_processor = MOTLEEDataProcessor(
    robots=robots,
    mapping_ts=params['mapping']['ts'],
    frame_align_ts=params['tcaff']['ts'],
    perform_mot=False
)

# setup results
results = {robot.name: AlignmentResults() for robot in robots}
# if args.viz:
#     object_file = "/home/masonbp/data/motlee_jan_2024/objects.txt"
#     objects = []
#     with open(object_file, "r") as f:
#         for l in f.readlines():
#             objects.append([float(n.strip()) for n in l.strip().split(',')])
#     objects = np.array(objects)
    
# set up visualizer
if args.viz:
    xlim = [np.inf, -np.inf]
    ylim = [np.inf, -np.inf]
    for robot in robots:
        for t in np.arange(t0, tf, params['mapping']['ts']):
            T_WB = pose_gt_data[robot.name].T_WB(t)
            xlim = [min(xlim[0], T_WB[0,3]), max(xlim[1], T_WB[0,3])]
            ylim = [min(ylim[0], T_WB[1,3]), max(ylim[1], T_WB[1,3])]
    xlim = [xlim[0] - 10.0, xlim[1] + 10.0]
    ylim = [ylim[0] - 10.0, ylim[1] + 10.0]
    
# Run the data processor
T_errs = []
printed_err = {robot.name: False for robot in robots}
for t in tqdm(np.arange(t0, tf, params['mapping']['ts'])):
    motlee_data_processor.update(t)
    if motlee_data_processor.fa_updated:
        if args.viz:
            plt.gca().clear()
            plt.gcf().set_size_inches(10.,10.)
            # plt.plot(objects[:,0], objects[:,1], 'k.')
            r0_map = robots[0].get_map()
            r1_map = robots[1].get_map()
            Tij_gt = xypsi_2_transform(*results[robots[0].name].get_Tij_gt(t, pose_gt_data[robots[0].name], 
                    pose_gt_data[robots[1].name], robots[0].pose_est_data, robots[1].pose_est_data))
            if len(r0_map) > 0:
                T = pose_gt_data[robots[0].name].T_WB(t) @ np.linalg.inv(robots[0].pose_est_data.T_WB(t))
                r0_map.centroids = transform(T, r0_map.centroids)
            if len(r1_map) > 0:
                T = pose_gt_data[robots[1].name].T_WB(t) @ np.linalg.inv(robots[1].pose_est_data.T_WB(t))
                r1_map.centroids = transform(T, r1_map.centroids)
            if len(r0_map) > 0:
                plt.plot(r0_map.centroids[:,0], r0_map.centroids[:,1], 'r.')
            if len(r1_map) > 0:
                plt.plot(r1_map.centroids[:,0], r1_map.centroids[:,1], 'b.')
            # square_equal_aspect()
            plt.gca().set_aspect('equal')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plot_pose2d(pose_gt_data[robots[0].name].T_WB(t))
            plot_pose2d(pose_gt_data[robots[1].name].T_WB(t))
            plt.pause(0.01)


        for (i, j) in zip([0,1], [1,0]):
            try:
                xytheta_ij_gt = results[robots[i].name].get_Tij_gt(t, pose_gt_data[robots[i].name], 
                    pose_gt_data[robots[j].name], robots[i].pose_est_data, robots[j].pose_est_data)
            except NoDataNearTimeException:
                xytheta_ij_gt = np.zeros(3) * np.nan
            # Tij_gt = results[robots[i].name].get_Tij_gt(t, pose_gt_data[robots[i].name], 
            #     pose_gt_data[robots[j].name], robots[i].pose_est_data, robots[j].pose_est_data, xytheta=False)
            # print(Tij_gt)
            results[robots[i].name].update_from_tcaff_manager(t, robots[i].frame_align_filters[robots[j].name], xytheta_ij_gt)
            if not np.any(np.isnan(robots[i].frame_align_filters[robots[j].name].T)):
                T_oi_ri = robots[i].pose_est_data.T_WB(t)
                T_oj_rj = robots[i-1].pose_est_data.T_WB(t)
                T_oi_oj = robots[i].frame_align_filters[robots[j].name].T
                T_ri_rj_est = np.linalg.inv(T_oi_ri) @ T_oi_oj @ T_oj_rj
                # print(T_ri_rj_est)

                try:
                    T_w_ri = pose_gt_data[robots[i].name].T_WB(t)
                    T_w_rj = pose_gt_data[robots[i-1].name].T_WB(t)
                    T_ri_rj_gt = np.linalg.inv(T_w_ri) @ T_w_rj
                    # print(T_ri_rj_gt)

                    T_errs.append(np.linalg.inv(T_ri_rj_gt) @ T_ri_rj_est)
                except:
                    continue

        for robot in robots:
            if not printed_err[robot.name] and not np.isnan(np.nanmean(results[robot.name].errors_translation)):
                print(f"initial translation error (m): {np.nanmean(results[robot.name].errors_translation)}")
                print(f"initial rotation error (deg): {np.rad2deg(np.nanmean(results[robot.name].errors_rotation))}")
                printed_err[robot.name] = True

avg_trans_err = np.mean([np.linalg.norm(T_err[:2,3]) for T_err in T_errs])
avg_rot_err = np.mean(np.abs([transform_2_xytheta(T_err)[2] for T_err in T_errs]))
print(f"average translation error (m): {avg_trans_err}")
print(f"average rotation error (deg): {np.rad2deg(avg_rot_err)}")



if show_passing_time:
    # Find rover passing time
    min_dist = np.inf
    min_dist_t = None
    for t in np.arange(t0, tf, params['mapping']['ts']):
        T_WB1 = pose_gt_data[robots[0].name].T_WB(t)
        T_WB2 = pose_gt_data[robots[1].name].T_WB(t)
        dist = np.linalg.norm(T_WB1[:3,3] - T_WB2[:3,3])
        if dist < min_dist:
            min_dist = dist
            min_dist_t = t

    t = min_dist_t
    print(f"Estimated pose of {robots[1].name} relative to {robots[0].name} at passing time")
    T_oi_ri = robots[0].pose_est_data.T_WB(t)
    T_oj_rj = robots[1].pose_est_data.T_WB(t)
    T_oi_oj = robots[0].frame_align_filters[robots[1].name].T
    if np.any(np.isnan(T_oi_oj)):
        for est in results[robots[0].name].est[::-1]:
            if not np.any(np.isnan(est)):
                T_oi_oj = xypsi_2_transform(*est)
                break
    try: 
        T_ri_rj = np.linalg.inv(T_oi_ri) @ T_oi_oj @ T_oj_rj
        print(T_ri_rj)
    except:
        print("could not be found")

    print(f"True pose of {robots[1].name} relative to {robots[0].name} at passing time")
    T_w_ri = pose_gt_data[robots[0].name].T_WB(t)
    T_w_rj = pose_gt_data[robots[1].name].T_WB(t)
    T_ri_rj = np.linalg.inv(T_w_ri) @ T_w_rj
    print(T_ri_rj)

for robot in robots:
    try:
        print(f"average translation error (m): {np.nanmean(results[robot.name].errors_translation)}")
        print(f"average rotation error (deg): {np.rad2deg(np.nanmean(results[robot.name].errors_rotation))}")
    except:
        import ipdb; ipdb.set_trace()

    fig, ax = results[robot.name].plot()
    if show_passing_time:
        for axi in ax:
            axi.plot([min_dist_t, min_dist_t], axi.get_ylim(), 'y--')
    plt.show()
    