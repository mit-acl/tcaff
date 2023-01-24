import argparse
import pathlib

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm

from person_detector import PersonDetector
from camera import Camera
from metric_evaluator import MetricEvaluator
from detections import GroundTruth
from inconsistency_counter import InconsistencyCounter
import config.rover_mot_params as PARAMS
import realign_frames


def getVideos(root, run, nums=['01', '04', '05', '06', '08'], numCams=4):
    caps = []
    for i in range(numCams):
        videopath = root /  f"run0{run}_RR{nums[i]}.avi"
        caps.append(cv.VideoCapture(videopath.as_posix()))
    return caps

def getTrackColor(i):
    c = np.array(plt.get_cmap('tab10').colors[i])
    c = (c * 255).astype(int)
    return tuple(v.item() for v in c[::-1])

def getMarkerType(cam):
    if cam == 0:
        return cv.MARKER_CROSS
    if cam == 1:
        return cv.MARKER_STAR
    if cam == 2:
        return cv.MARKER_SQUARE
    if cam == 3:
        return cv.MARKER_TRIANGLE_UP
    
def give_cam_transform(cam, detector, numCams=4, incl_noise=False):
    for i in range(numCams):
        if i == cam.camera_id:
            continue
        cam.T_other[i] = detector.get_T_obj2_obj1(cam.camera_id, i, incl_noise=incl_noise)
        
def perform_frame_realignment(agents):
    # print('\n\n\nHAPPENING NOW!!!\n\n\n')
    for i, a1 in enumerate(agents):
        d1 = realign_frames.detections2pointcloud(a1.get_recent_detections(), org_by_tracks=False)
        for j, a2 in enumerate(agents):
            if i == j: continue
            d2 = realign_frames.detections2pointcloud(a2.get_recent_detections(), org_by_tracks=False)
            if j in a1.T_other:
                a1.T_other[j] = realign_frames.realign_frames(d1, d2, initial_guess=a1.T_other[j])
            else:
                a1.T_other[j] = realign_frames.realign_frames(d1, d2)
            
        
def get_avg_metric(metric, mes, divide_by_frames=False):
    num_cams=len(mes)
    m_avg = 0
    for me in mes:
        if divide_by_frames:
            m_val = me.get_metric(metric) / (me.get_metric('num_frames') * num_cams)
        else:
            m_val = me.get_metric(metric) / num_cams
        m_avg += m_val
    return m_avg

def print_results(mes, inconsistencies, agents):
    mota = get_avg_metric('mota', mes)
    motp = get_avg_metric('motp', mes)
    fp = get_avg_metric('num_false_positives', mes, divide_by_frames=True)
    fn = get_avg_metric('num_misses', mes, divide_by_frames=True)
    switch = get_avg_metric('num_switches', mes, divide_by_frames=True)
    precision = get_avg_metric('precision', mes)
    recall = get_avg_metric('recall', mes)
    total_num_tracks = sum([len(a.tracker_mapping) / numCams for a in agents]) / numCams
    incon_per_track = inconsistencies / total_num_tracks if total_num_tracks else 0.0

    print(f'mota: {mota}')
    print(f'motp: {motp}')
    print(f'fp: {fp}')
    print(f'fn: {fn}')
    print(f'switch: {switch}')
    print(f'inconsistencies: {inconsistencies}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'num_tracks: {total_num_tracks}')
    print(f'incon_per_track: {incon_per_track}')

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
    parser.add_argument('--std_dev_rotation',
            type=float,
            default=0,
            help='Camera rotation error standard deviation')
    parser.add_argument('--std_dev_translation',
            type=float,
            default=0,
            help='Camera translation error standard deviation')
    parser.add_argument('--viewer',
            action='store_true',
            help='Shows camera views, network estimates, and ground truth')
    parser.add_argument('--no-progress-bar',
            action='store_true',
            help='Disables progress bar')
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
    parser.add_argument('--num-frame-fixes',
            type=int,
            default=1,
            help='Number of attempts to make to improve frame alignment')
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    if args.run == 1:
        bagfile = 'run01_filtered.bag'
        num_peds = 3
        FIRST_FRAME = 300
        # FIRST_FRAME = 30*70
        START_METRIC_FRAME = 30*90
        LAST_FRAME = 5640
    elif args.run == 3:
        bagfile = 'run03_filtered.bag'
        num_peds = 5
        FIRST_FRAME = 600
        LAST_FRAME = 7500
    bagfile = str(root / bagfile)
    GT = GroundTruth(bagfile, [f'{i+1}' for i in range(num_peds)], 'RR01')

    numCams = args.num_cams
    caps = getVideos(root, run=args.run, numCams=numCams)
    agents = []
    mes = []
    detector = PersonDetector(run=args.run, sigma_r=args.std_dev_rotation*np.pi/180, sigma_t=args.std_dev_translation, num_cams=numCams)
    for i in range(numCams):
        T_cam = detector.get_cam_T(i)
        agents.append(Camera(i, Tau_LDA=PARAMS.TAU_LDA, Tau_GDA=PARAMS.TAU_GDA, kappa=PARAMS.KAPPA,
                             alpha=PARAMS.ALPHA, n_meas_init=PARAMS.N_MEAS_TO_INIT_TRACKER, T=T_cam))
        give_cam_transform(agents[i], detector, numCams=numCams, incl_noise=args.init_transform)
        _, t_cam, R_noise, T_noise = detector.get_cam_pose(i)
        R_noise, T_noise = R_noise[0:2, 0:2], T_noise[0:2, :]
        mes.append(MetricEvaluator(t_cam=t_cam[0:2,0:1], noise_rot=R_noise, noise_tran=T_noise))
        
    inconsistencies = 0
    ic = InconsistencyCounter()
    last_printed_metrics = 0

    SKIP_FRAMES = 1
    COLORS = [(255,0,0), (0,255,0), (0,0,255), (255, 255, 255)]
    TRIGGER_AUTO_CYCLE_TIME = .2 # .1: .698158, .2: .695

    if args.debug: 
        with open(args.debug, 'w') as f:
            print('<<<debug log>>>\n', file=f)
            
    last_frame_time = 0
    if args.no_progress_bar:
        frame_range = range(FIRST_FRAME, LAST_FRAME)
    else:
        frame_range = tqdm(range(FIRST_FRAME, LAST_FRAME))
    for framenum in frame_range:
        frame_time = framenum / 30 + detector.start_time
        
        if args.num_frame_fixes == 1:
            if framenum == 90*30:
                # print('\n\n\nperforming frame alignment!\n\n\n')
                for a in agents: a.frame_realign()
                # perform_frame_realignment(agents)
        elif args.num_frame_fixes != 0:
            if framenum == 60*30 or framenum == 75*30 or framenum == 90*30 or framenum == 105*30 or framenum == 130*30 or framenum == 145*30 or framenum == 160*30:
            # if framenum == 30*30 or framenum == 50*30 or framenum == 70*30 or framenum == 90*30 or framenum == 110*30 or framenum == 130*30 or framenum == 150*30:
                # perform_frame_realignment(agents)
                # print('\n\n\nperforming frame alignment!\n\n\n')
                for a in agents: a.frame_realign()
                
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
        for i, (cap, a) in enumerate(zip(caps, agents)):

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

            a.local_data_association(Zs, feature_vecs)
            observations += a.get_observations()  

            if args.viewer:
                height, width, channels = frame.shape
                resized = cv.resize(frame, (int(width/3), int(height/3)))
                cv.imshow(f"frame{i}", resized)


        for a in agents:
            a.add_observations(observations)
            a.dkf()
            a.tracker_manager()
            ic.add_groups(a.camera_id, a.groups_by_id)
            a.groups_by_id = []
            if args.debug:
                with open(args.debug, 'a') as f:
                    np.set_printoptions(precision=1)
                    print(a, file=f)
        inconsistencies += ic.count_inconsistencies()

        if framenum == FIRST_FRAME:
            topview_size = 600
            topview = np.ones((topview_size,topview_size,3))*255
            topview = topview.astype(np.uint8)

        else:
            combined = topview.copy()

            if args.viewer:
                for i, a in enumerate(agents):

                    Xs, colors = a.get_trackers()   

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
                    cv.drawMarker(combined, pt, getTrackColor(int(ped_id)), getMarkerType(i), thickness=2, markerSize=30)
            
            if framenum > START_METRIC_FRAME:    
                for a, me in zip(agents, mes):
                    me.update(gt_dict, a.get_trackers(format='dict'))
                if args.metric_frequency != 0 and frame_time - last_printed_metrics > 1 / args.metric_frequency:
                    print_results(mes, inconsistencies, agents)
                    last_printed_metrics = frame_time

            if args.viewer and framenum % SKIP_FRAMES == 0:
                cv.imshow('topview', combined)

        if args.viewer:
            cv.waitKey(5)
        if framenum >= LAST_FRAME:
            break

    cap.release()
    cv.destroyAllWindows()
        
    print('FINAL RESULTS')
    print_results(mes, inconsistencies, agents)