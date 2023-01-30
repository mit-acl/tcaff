import pickle # delete
import argparse
import pathlib

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm

from frontend.person_detector import PersonDetector
from frontend.detections import GroundTruth
from mot.multi_object_tracker import MultiObjectTracker
from metrics.metric_evaluator import MetricEvaluator, print_metric_results
from metrics.inconsistency_counter import InconsistencyCounter
import config.rover_mot_params as PARAMS


def get_videos(root, run, nums=['01', '04', '05', '06', '08'], num_cams=4):
    caps = []
    for i in range(num_cams):
        videopath = root /  f"run0{run}_RR{nums[i]}.avi"
        caps.append(cv.VideoCapture(videopath.as_posix()))
    return caps

def get_track_color(i):
    c = np.array(plt.get_cmap('tab10').colors[i])
    c = (c * 255).astype(int)
    return tuple(v.item() for v in c[::-1])
    
# def give_cam_transform(cam, detector, num_cams=4, incl_noise=False):
#     for i in range(num_cams):
#         if i == cam.camera_id:
#             continue
#         cam.T_other[i] = detector.get_T_obj2_obj1(cam.camera_id, i, incl_noise=incl_noise)            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate video sequence with ground truth')
    parser.add_argument('-r', '--root',
            type=str,
            default='/home/masonbp/ford-project/data/static-20221216',
            help='Dataset root directory')
    parser.add_argument('--run',
            type=int,
            default=1,
            help='Rover test run number')
    parser.add_argument('-d', '--debug',
            type=str,
            default=None,
            help='Print debug logging')
    parser.add_argument('--std-dev-rotation',
            type=float,
            default=0,
            help='Camera rotation error standard deviation')
    parser.add_argument('--std-dev-translation',
            type=float,
            default=0,
            help='Camera translation error standard deviation')
    parser.add_argument('--viewer',
            action='store_true',
            help='Shows camera views, network estimates, and ground truth')
    parser.add_argument('-q', '--quiet',
            action='store_true',
            help='Disables progress bar and metric output')
    parser.add_argument('--metric-frequency',
            type=float,
            default=0,
            help='Frequency (hz) to show metric summary')
    parser.add_argument('--num-cams',
            type=int,
            default=4,
            help='Number of cameras to use in simulation')
    parser.add_argument('--init-transform',
            action='store_true',
            help='Cameras know the transformations between each other')
    parser.add_argument('--realign',
            action='store_true',
            help='Robots will perform frame realignment')
    parser.add_argument('--cam-type',
            type=str,
            default="d435",
            help="Color d435 or fisheye t265")
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    if args.run == 1:
        bagfile = 'run01_filtered.bag'
        num_peds = 3
        FIRST_FRAME = 300
        START_METRIC_FRAME = FIRST_FRAME
        LAST_FRAME = 5640
    elif args.run == 3:
        bagfile = 'run03_filtered.bag'
        num_peds = 5
        FIRST_FRAME = 600
        LAST_FRAME = 7500
        START_METRIC_FRAME = (LAST_FRAME - FIRST_FRAME) / 2
    bagfile = str(root / bagfile)
    GT = GroundTruth(bagfile, [f'{i+1}' for i in range(num_peds)], 'RR01')

    num_cams = args.num_cams
    caps = get_videos(root, run=args.run, num_cams=num_cams)
    robots = []
    mes = []
    detector = PersonDetector(run=args.run, sigma_r=args.std_dev_rotation*np.pi/180, sigma_t=args.std_dev_translation, num_cams=num_cams, cam_type=args.cam_type)
    for i in range(num_cams):
        T_cam = detector.get_cam_T(i)
        connected_cams = [*range(num_cams)]; connected_cams.remove(i)
        robots.append(MultiObjectTracker(i, connected_cams=connected_cams, params=PARAMS, T=T_cam))
        # give_cam_transform(robots[i], detector, num_cams=num_cams, incl_noise=args.init_transform)
        _, t_cam, R_noise, T_noise = detector.get_cam_pose(i)
        R_noise, T_noise = R_noise[0:2, 0:2], T_noise[0:2, :]
        mes.append(MetricEvaluator(t_cam=t_cam[0:2,0:1], noise_rot=R_noise, noise_tran=T_noise))
        
    inconsistencies = 0
    ic = InconsistencyCounter()
    last_printed_metrics = 0

    TRIGGER_AUTO_CYCLE_TIME = .2 # .1: .698158, .2: .695
    last_frame_time = 0

    if args.debug: 
        with open(args.debug, 'w') as f: print('<<<debug log>>>\n', file=f)
    if args.quiet:
        frame_range = range(FIRST_FRAME, LAST_FRAME)
    else:
        frame_range = tqdm(range(FIRST_FRAME, LAST_FRAME))
        
    #######################    
    # MAIN LOOP
    #######################
    
    for framenum in frame_range:
        frame_time = framenum / 30 + detector.start_time
        
        # Frame Realignment
        if args.realign and framenum > (FIRST_FRAME + 10*30) and (framenum - FIRST_FRAME) % (10*30) == 0:
            # with open('robot_data.pkl', 'wb') as outp:
            #     pickle.dump(robots, outp, pickle.HIGHEST_PROTOCOL)
            #     Ts = dict()
            #     for i in range(num_cams):
            #         Ts[i] = dict()
            #         for j in range(num_cams):
            #             if i == j:
            #                 continue
            #             Ts[i][j] = detector.get_T_obj2_obj1(i, j, incl_noise=True)   
            #     pickle.dump(Ts, outp, pickle.HIGHEST_PROTOCOL)
            #     exit(0)
            for rob in robots: rob.frame_realign()
            
        # Continues to next frame when robots have new detections
        if not detector.times_different(frame_time, last_frame_time) and abs(frame_time - last_frame_time) < TRIGGER_AUTO_CYCLE_TIME:
            for cap in caps:
                cap.read()
            continue
        last_frame_time = frame_time
        
        if args.debug:
            with open(args.debug, 'a') as f:
                print('//////////////////////////////////////////////////', file=f)
                print(f'framenum: {framenum}', file=f)
                print('//////////////////////////////////////////////////\n', file=f)

        current_frames = []
        observations = []
        for i, (cap, rob) in enumerate(zip(caps, robots)):

            if framenum == FIRST_FRAME:
                for j in range(FIRST_FRAME):
                    cap.read()

            ret, frame = cap.read()

            if not ret:
                break

            Zs = []
            positions, boxes, feature_vecs = detector.get_person_boxes(frame, i, frame_time)
            for pos, box in zip(positions, boxes):
                x0, y0, x1, y1 = box
                Zs.append(np.array([[pos[0], pos[1], 20, 50]]).T)
                if args.viewer:
                    cv.rectangle(frame, (int(x0),int(y0)), (int(x1),int(y1)), (0,255,0), 4)

            rob.local_data_association(Zs, feature_vecs)
            observations += rob.get_observations()  

            if args.viewer:
                height, width, channels = frame.shape
                resized = cv.resize(frame, (int(width/3), int(height/3)))
                cv.imshow(f"frame{i}", resized)

        for rob in robots:
            rob.add_observations(observations)
            rob.dkf()
            rob.tracker_manager()
            # if args.realign:
            #     rob.frame_realign()
            ic.add_groups(rob.camera_id, rob.groups_by_id)
            rob.groups_by_id = []
            if args.debug:
                with open(args.debug, 'a') as f:
                    np.set_printoptions(precision=1)
                    print(rob, file=f)
        inconsistencies += ic.count_inconsistencies()

        if framenum == FIRST_FRAME:
            topview_size = 600
            topview = np.ones((topview_size,topview_size,3))*255
            topview = topview.astype(np.uint8)
            cv.rectangle(topview, 
                (int(-7*topview_size/20 + topview_size/2), int(-7*topview_size/20 + topview_size/2)), 
                (int(7*topview_size/20 + topview_size/2), int(7*topview_size/20 + topview_size/2)), 
                color=(0, 0, 0), thickness=4)
            cv.rectangle(topview, (int(topview_size/2), int(-7*topview_size/20 + topview_size/2)),
                (int(3*topview_size/20 + topview_size/2), int(-7*topview_size/20 + topview_size/2)),
                color=(255, 255, 255), thickness=4)

        else:
            combined = topview.copy()

            if args.viewer:
                for i, rob in enumerate(robots):

                    Xs, colors = rob.get_trackers()   

                    for X, color in zip(Xs, colors):
                        x, y, w, h, _, _ = X.reshape(-1).tolist()
                        x, y = x*topview_size/20 + topview_size/2, y*topview_size/20 + topview_size/2
                        cv.circle(combined, (int(x),int(y)), int(2*w/3), color, 4)
            
            ped_ids, peds = GT.ped_positions(frame_time)
            gt_dict = dict()
            for ped_id, ped_pos in zip(ped_ids, peds):
                gt_dict[ped_id] = ped_pos[0:2]
                if args.viewer:
                    pt = (int(ped_pos[0]*topview_size/20 + topview_size/2), int(ped_pos[1]*topview_size/20 + topview_size/2))
                    cv.drawMarker(combined, pt, get_track_color(int(ped_id)), cv.MARKER_STAR, thickness=2, markerSize=30)
            
            if framenum > START_METRIC_FRAME:    
                for rob, me in zip(robots, mes):
                    me.update(gt_dict, rob.get_trackers(format='dict'))
                if not args.quiet and args.metric_frequency != 0 and \
                    frame_time - last_printed_metrics > 1 / args.metric_frequency:
                    print_metric_results(mes, inconsistencies, robots)
                    last_printed_metrics = frame_time

            if args.viewer:
                cv.imshow('topview', combined)

        if args.viewer:
            cv.waitKey(5)
        if framenum >= LAST_FRAME:
            break

    cap.release()
    cv.destroyAllWindows()
        
    print('FINAL RESULTS')
    print_metric_results(mes, inconsistencies, robots)