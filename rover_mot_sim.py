import argparse
import pathlib

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import cv_bridge
from tqdm import tqdm

from person_detector import PersonDetector
from camera import Camera
from metric_evaluator import MetricEvaluator
from detections import GroundTruth


def getVideos(root, nums=['01', '04', '05', '06', '08'], numCams=4):
    caps = []
    for i in range(numCams):
        videopath = root /  f"run01_RR{nums[i]}.avi"
        caps.append(cv.VideoCapture(videopath.as_posix()))
    return caps

def getTrackColor(i):
    c = np.array(plt.get_cmap('tab10').colors[i])
    c = (c * 255).astype(int)
    # import ipdb; ipdb.set_trace()
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate video sequence with ground truth')
    parser.add_argument('-r', '--root',
            type=str,
            default='/home/masonbp/ford-project/data/static-20221216',
            help='Dataset root directory')
    parser.add_argument('-b', '--bag-file',
            type=str,
            default='run01_2022-12-16-15-40-22.bag',
            help='ROS Bag file')
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
    parser.add_argument('--num-peds',
            type=int,
            default=3,
            help='Number of pedestrians to track in simulation (used for ground truth only)')
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    bagfile = str(root / args.bag_file)
    GT = GroundTruth(bagfile, [f'{i+1}' for i in range(args.num_peds)], 'RR01')

    numCams = args.num_cams
    caps = getVideos(root, numCams=numCams)
    agents = []
    mes = []
    for i in range(numCams):
        agents.append(Camera(i))
        mes.append(MetricEvaluator())
    detector = PersonDetector(sigma_r=args.std_dev_rotation*np.pi/180, sigma_t=args.std_dev_translation, num_cams=numCams)
    inconsistencies = 0

    SKIP_FRAMES = 1
    FIRST_FRAME = 300
    LAST_FRAME = 5690
    COLORS = [(255,0,0), (0,255,0), (0,0,255), (255, 255, 255)]

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
        if not detector.times_different(frame_time, last_frame_time):
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

            if framenum % SKIP_FRAMES == 0:
                Zs = []
                positions, boxes, feature_vecs = detector.get_person_boxes(frame, i, frame_time)
                for pos, box in zip(positions, boxes):
                    x0, y0, x1, y1 = box
                    Zs.append(np.array([[pos[0], pos[1], 20, 50]]).T)
                    cv.rectangle(frame, (int(x0),int(y0)), (int(x1),int(y1)), (255,0,0), 2)

                a.local_data_association(Zs, feature_vecs)
                observations += a.get_observations()  

                if args.viewer:
                    cv.imshow(f"frame{i}", frame)

        if framenum % SKIP_FRAMES == 0:
            for a in agents:
                a.add_observations(observations)
                a.dkf()
                a.tracker_manager()
                inconsistencies += a.inconsistencies
                if args.debug:
                    with open(args.debug, 'a') as f:
                        np.set_printoptions(precision=1)
                        print(a, file=f)

        if framenum == FIRST_FRAME:
            topview = np.ones((400,400,3))*255
            topview = topview.astype(np.uint8)

        else:
            combined = topview.copy()

            for i, a in enumerate(agents):

                Xs, colors = a.get_trackers()   

                for X, color in zip(Xs, colors):
                    x, y, w, h, _, _ = X.reshape(-1).tolist()
                    x, y = x*20 + 200, y*20 + 200
                    cv.circle(combined, (int(x),int(y)), int(w/2), color, 4)
            
            ped_ids, peds = GT.ped_positions(frame_time)
            gt_dict = dict()
            for ped_id, ped_pos in zip(ped_ids, peds):
                gt_dict[ped_id] = ped_pos[0:2]
                pt = (int(ped_pos[0]*20 + 200), int(ped_pos[1]*20 + 200))
                cv.drawMarker(combined, pt, getTrackColor(int(ped_id)), getMarkerType(i), thickness=2)
            
            if framenum > 250:    
                for a, me in zip(agents, mes):
                    me.update(gt_dict, a.get_trackers(format='dict'))
                    if args.metric_frequency != 0 and framenum % int((1 / args.metric_frequency) * 30) == 0:
                        me.display_results()
                        print(f'inconsistencies: {inconsistencies}')

            if args.viewer and framenum % SKIP_FRAMES == 0:
                cv.imshow('topview', combined)

        cv.waitKey(5)
        if framenum >= LAST_FRAME:
            break

    cap.release()
    cv.destroyAllWindows()
    
    mota, motp = 0, 0
    for me in mes:
        mota += me.mota / numCams
        motp += me.motp / numCams
        
    print('FINAL RESULTS')
    print(f'mota: {mota}')
    print(f'motp: {motp}')
    print(f'inconsistencies: {inconsistencies}')