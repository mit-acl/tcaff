import argparse
import pathlib

import numpy as np
import cv2 as cv
from tqdm import tqdm

from metrics.metric_evaluator import print_metric_results
import config.rover_mot_params as PARAMS

from frontend.rover_mot_frontend import RoverMotFrontend

def get_videos(root, run, first_frame, nums=['01', '04', '05', '06', '08'], num_cams=4):
    caps = []
    for i in range(num_cams):
        videopath = root /  f"run0{run}_RR{nums[i]}.avi"
        caps.append(cv.VideoCapture(videopath.as_posix()))
    for cap in caps:
        for i in range(first_frame):
            cap.read()
    return caps
    
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
    parser.add_argument('--wls-only',
            action='store_true')
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    if args.run == 1:
        detection_bag = 'centertrack_detections/fisheye/run01_{}.bag'
        ped_bag = 'run01_filtered.bag'
        num_peds = 3
        FIRST_FRAME = 300
        START_METRIC_FRAME = FIRST_FRAME
        LAST_FRAME = 5640
    elif args.run == 3:
        detection_bag = 'centertrack_detections/fisheye/run03_{}.bag'
        ped_bag = 'run03_filtered.bag'
        num_peds = 5
        FIRST_FRAME = 600
        LAST_FRAME = 7500
        START_METRIC_FRAME = (LAST_FRAME - FIRST_FRAME) / 2
    detection_bag = str(root / detection_bag)
    ped_bag = str(root / ped_bag)
    if args.wls_only:
        PARAMS.realign_algorithm = PARAMS.RealignAlgorithm.REALIGN_WLS
    num_cams = args.num_cams

    sim = RoverMotFrontend(
        detection_bag=detection_bag,
        ped_bag=ped_bag,
        mot_params=PARAMS,
        rovers=['RR01', 'RR04', 'RR05', 'RR06', 'RR08'][:num_cams],
        rover_pose_topic='/world',
        cam_type=args.cam_type,
        vids=get_videos(root, run=args.run, first_frame=FIRST_FRAME, num_cams=num_cams),
        noise=(args.std_dev_translation, args.std_dev_rotation*np.pi/180),
        viewer=args.viewer
    )

    if args.quiet:
        frame_range = range(FIRST_FRAME, LAST_FRAME)
    else:
        frame_range = tqdm(range(FIRST_FRAME, LAST_FRAME))
        
    #######################    
    # MAIN LOOP
    #######################
    
    last_printed_metrics = sim.frame_time
    for framenum in frame_range:

        realign = args.realign and framenum > (FIRST_FRAME + 20*30) and (framenum - FIRST_FRAME) % (20*30) == 0
        sim.update(framenum, run_metrics=(framenum > START_METRIC_FRAME), realign=realign)

        if not args.quiet and args.metric_frequency != 0 and \
            sim.frame_time - last_printed_metrics > 1 / args.metric_frequency:
            print_metric_results(sim.mes, sim.inconsistencies, sim.mots)
            last_printed_metrics = sim.frame_time
        if framenum >= LAST_FRAME:
            break

    for vid in sim.vids:
        vid.release()
    cv.destroyAllWindows()
        
    print('FINAL RESULTS')
    print_metric_results(sim.mes, sim.inconsistencies, sim.mots)