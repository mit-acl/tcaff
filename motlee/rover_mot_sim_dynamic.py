import argparse
import pathlib

import numpy as np
import cv2 as cv
from tqdm import tqdm

from metrics.metric_evaluator import MetricEvaluator, print_metric_results
from config import rover_mot_params

from frontend.rover_mot_frontend import RoverMotFrontend

def get_videos(root, cam_type, run, first_frame, rovers):
    caps = []
    for r in rovers:
        videopath = root /  f"videos/{cam_type}/run{run}_{r}.avi"
        print(videopath)
        caps.append(cv.VideoCapture(videopath.as_posix()))
    for cap in caps:
        for i in range(first_frame):
            cap.read()
    return caps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate video sequence with ground truth')
    parser.add_argument('-r', '--root',
            type=str,
            default='/home/masonbp/ford/data/mot_dynamic/dynamic_motlee_iros',
            help='Dataset root directory')
    parser.add_argument('--run',
            type=int,
            default=1,
            help='Rover test run number')  
    parser.add_argument('--viewer',
            action='store_true',
            help='Shows camera views, network estimates, and ground truth')
    parser.add_argument('-q', '--quiet',
            action='store_true',
            help='Disables progress bar and metric output')
    parser.add_argument('--metric-frequency',
            type=float,
            default=2,
            help='Frequency (hz) to show metric summary')
    parser.add_argument('--num-rovers',
            type=int,
            default=4,
            help='Number of cameras to use in simulation')
    parser.add_argument('--rovers',
            type=str,
            default='468')
    parser.add_argument('--realign',
            action='store_true',
            help='Robots will perform frame realignment')
    parser.add_argument('--wls-only',
            action='store_true')
    parser.add_argument('--use-odom', action='store_true')
    parser.add_argument('--init-cnt',
            type=int, default=3,
            help='Number of detections required before initializing a track')
    parser.add_argument('--t265-conf',
            type=float, default=.5)
    parser.add_argument('--l515-conf',
            type=float, default=.6)
    parser.add_argument('--json-file', type=str, default=None)
    parser.add_argument('--metric-dist', type=float, default=1.0)
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    if args.run == 1:
        ped_dir = 'gt/run1'
        detection_dirs = [f'ped_detections/t265/run1_{{}}',
                          f'ped_detections/l515/run1_{{}}']
        # final plot
        # FIRST_FRAME = 60*30
        # LAST_FRAME = 175*30 #150*30 #220*30 #7650
        # START_METRIC_FRAME = 60*30 #500
        # register_time = 65
        
        # longer but not the whole time
        # FIRST_FRAME = 60*30
        # LAST_FRAME = 240*30 #150*30 #220*30 #7650
        # START_METRIC_FRAME = 65*30 #500
        # register_time = 65
        
        # FIRST_FRAME = 60*30
        # LAST_FRAME = 110*30 #220*30 #7650
        # START_METRIC_FRAME = 65*30 #500
        
        # ######### THE REAL ONE IS HEREEE########
        FIRST_FRAME = 15*30
        LAST_FRAME = 7650
        START_METRIC_FRAME = 15*30
        register_time = 20
        REALIGN_PERIOD = 1
        
        # for dynamic plotting
        # FIRST_FRAME = 15*30
        # LAST_FRAME = 180*30
        # START_METRIC_FRAME = 15*30
        # register_time = 0
        # REALIGN_PERIOD = 1
    ped_dir = str(root / ped_dir)
    for i in range(len(detection_dirs)):
        detection_dirs[i] = str(root / detection_dirs[i])
    PARAMS = rover_mot_params.RoverMOTParams()
    if args.wls_only:
        PARAMS.realign_algorithm = rover_mot_params.RealignAlgorithm.REALIGN_WLS
    num_rovers = args.num_rovers
    PARAMS.n_meas_to_init_tracker = args.init_cnt
    PARAMS.kappa = 4
    inst_metrics = True
    vicon_cones = False
    metric_d = args.metric_dist

    rovers = [f'RR0{i}' for i in args.rovers]
    cam_types = ['t265', 'l515']
    sim = RoverMotFrontend(
        ped_dir=ped_dir,
        mot_params=PARAMS,
        detection_dirs=detection_dirs,
        rovers=rovers[:num_rovers],
        use_noisy_odom=args.use_odom,
        cam_types=cam_types,
        vids=get_videos(root, 't265', run=args.run, first_frame=FIRST_FRAME, rovers=rovers) + \
            get_videos(root, 'l515', run=args.run, first_frame=FIRST_FRAME, rovers=rovers),
        noise=(0.0, 0.0),
        register_time=register_time,
        metric_d=metric_d,
        vicon_cones=vicon_cones,
        viewer=args.viewer
    )
    sim.TRIGGER_AUTO_CYCLE_TIME = .1
    # if args.run == 3:
    #     sim.start_time = 1675727098.0

    if args.quiet:
        frame_range = range(FIRST_FRAME, LAST_FRAME)
    else:
        frame_range = tqdm(range(FIRST_FRAME, LAST_FRAME))
        
    #######################    
    # MAIN LOOP
    #######################
    
    last_printed_metrics = sim.frame_time
    for framenum in frame_range:

        realign = args.realign and framenum >= START_METRIC_FRAME + (0*30) and (framenum - FIRST_FRAME) % (REALIGN_PERIOD*30) == 0
        if realign:
            print('REALIGN')
        # if framenum < START_METRIC_FRAME:
        #     sim.inform_true_pairwise_T(framenum)
        sim.update(framenum, run_metrics=(framenum > START_METRIC_FRAME), realign=realign)

        if args.metric_frequency != 0 and \
            sim.frame_time - last_printed_metrics > 1 / args.metric_frequency:
            print_metric_results(sim.mes[0], sim.inconsistencies, sim.mots, mota_only=False)
            print(f'T_mag: {sim.T_mag()}')
            # print(f'T_diff: {sim.calc_T_diff()}')
            # print(f'T_diff_list: {sim.get_all_T_diffs()}')
            v = sim.calc_T_diff(filter='psi')
            print(f'psi_diff: {v}')
            v = sim.get_all_T_diffs(filter='psi')
            print(f'psi_diff_list: {v}')
            v = sim.calc_T_diff(filter='t')
            print(f't_diff: {v}')
            v = sim.get_all_T_diffs(filter='t')
            print(f't_diff_list: {v}')
            residuals, num_cones = sim.get_residual_and_cones()
            print(f'residuals: {residuals}')
            print(f'num_cones: {num_cones}')
            # det_v = []
            # det_p = []
            # det_q = []
            # det_r = []
            # MDs = []
            # for mot in sim.mots:
            #     det_v += [np.linalg.det(track.V) for track in mot.tracks]
            #     det_p += [np.linalg.det(track.P) for track in mot.tracks]
            #     det_q += [np.linalg.det(track.Q) for track in mot.tracks]
            #     det_r += [np.linalg.det(track.R) for track in mot.tracks]
            #     MDs += mot.MDs
            # print(f'det(V): {det_v}')
            # print(f'det(P): {det_p}')
            # print(f'det(Q): {det_q}')
            # print(f'det(R): {det_r}')
            # print(f'MDs: {MDs}')
            print()
            if inst_metrics:
                if len(sim.mes) >= 20:
                    del sim.mes[0]
                new_mes = []
                for i in range(len(sim.mes[0])):
                    new_mes.append(MetricEvaluator(max_d=metric_d))
                sim.mes.append(new_mes)
                    
            last_printed_metrics = sim.frame_time
        if framenum >= LAST_FRAME:
            break

    for vid in sim.vids:
        vid.release()
    cv.destroyAllWindows()
        
    print('FINAL RESULTS')
    print('final ', end='')
    print_metric_results(sim.full_mes, sim.inconsistencies, sim.mots, mota_only=False)

    if args.json_file is not None:
        with open(args.json_file, 'w') as f:
            import json
            json_str = json.dumps(sim.debug)
            print(json_str, file=f)