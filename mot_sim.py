import argparse
import pathlib

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from person_detector import PersonDetector
from camera import Camera
from metric_evaluator import MetricEvaluator


def getHomographies():
    H0 = np.array([
            [-1.6688907435,   -6.9502305710,  940.69592392565],
            [ 1.1984806153,  -10.7495778320,  868.29873467315],
            [ 0.0004069210,   -0.0209324057,    0.42949125235]
        ])

    H1 = np.array([
            [0.6174778372,   -0.4836875683,   147.00510919005],
            [0.5798503075,    3.8204849039,  -386.096405131],
            [0.0000000001,    0.0077222239,    -0.01593391935]
        ])

    H2 = np.array([
            [-0.2717592338,    1.0286363982,    -17.6643219215],
            [-0.1373600672,   -0.3326731339,    161.0109069274],
            [ 0.0000600052,    0.0030858398,     -0.04195162855]
        ])

    H3 = np.array([
            [-0.3286861858,   0.1142963200,    130.25528281945],
            [ 0.1809954834,    -0.2059386455,   125.0260427323],
            [ 0.0000693641,    0.0040168154,    -0.08284534995]
        ])

    return [H0, H1, H2, H3]


def getAnnotations(root, numCams=4):
    GTs = []
    for i in range(numCams):
        annopath = root / f"ground-truth-detections/terrace1-c{i}.txt"
        GTs.append(np.loadtxt(annopath, usecols=(0,1,2,3,4,5,6,7,8)))
    return GTs


def getVideos(root, numCams=4):
    caps = []
    for i in range(numCams):
        videopath = root /  f"terrace1-c{i}.avi"
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

# TODO: Figure out why the scale seems so weird (why is each )

def projectToGroundPlane(x, y, H):
    pt = H @ np.r_[x, y, 1]
    pt /= pt[2]
    # pt = pt.astype(int)
    pt = (pt[0], pt[1])
    return pt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate video sequence with ground truth')
    parser.add_argument('-r', '--root',
            type=str,
            default='/media/plusk01/721fc902-b795-4e13-980a-ef6014eb03f0/datasets/epfl/terrace',
            help='Dataset root directory')
    parser.add_argument('-v', '--video-file',
            type=str,
            default='terrace1-c0.avi',
            help='Video file')
    parser.add_argument('-t', '--truth-file',
            type=str,
            default='ground-truth-detections/terrace1-c0.txt',
            help='Ground truth bounding box annotations')
    parser.add_argument('-c', '--calibration-file',
            type=str,
            default='calibration.txt',
            help='Ground truth calibration file')
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
    args = parser.parse_args()



    root = pathlib.Path(args.root)


    # (288, 360) (h, w)
    numCams = 4
    caps = getVideos(root)
    GTs = getAnnotations(root)
    Hs = getHomographies()
    agents = []
    mes = []
    for i in range(numCams):
        agents.append(Camera(i))
        mes.append(MetricEvaluator())
    detector = PersonDetector(sigma_r=args.std_dev_rotation*np.pi/180, sigma_t=args.std_dev_translation)

    SKIP_FRAMES = 1
    FIRST_FRAME = 60
    # FIRST_FRAME = 300
    # FIRST_FRAME = 450
    COLORS = [(255,0,0), (0,255,0), (0,0,255), (255, 255, 255)]

    framenum = -1
    firstframes = []
    warpcams = False

    if args.debug: 
        with open(args.debug, 'w') as f:
            print('<<<debug log>>>\n', file=f)
    while caps[0].isOpened():
        if framenum == 0:
            framenum = FIRST_FRAME
        framenum += 1
        if args.debug:
            with open(args.debug, 'a') as f:
                print('//////////////////////////////////////////////////', file=f)
                print(f'framenum: {framenum}', file=f)
                print('//////////////////////////////////////////////////\n', file=f)

        current_frames = []
        observations = []
        for i, (cap, GT, H, a) in enumerate(zip(caps, GTs, Hs, agents)):

            if framenum == 0:
                for j in range(FIRST_FRAME):
                    cap.read()

            ret, frame = cap.read()

            if not ret:
                break

            if framenum == 0:
                firstframes.append(frame)

            if warpcams:
                frame = cv.warpPerspective(frame, H, frame.shape[1::-1])

            # get ground truth annotations of this frame
            gt = GT[GT[:,5]==framenum,:]
            for row in gt:
                id, x0, y0, x1, y1, _, lost, occluded, generated = row

                if not lost and not occluded:
                    x, y = x0 + (x1 - x0) / 2, y1

                    if warpcams:
                        pt = projectToGroundPlane(x,y,H)
                        pt = (int(pt[0]), int(pt[1]))
                    else:
                        pt = int(x), int(y)

                    cv.drawMarker(frame, pt, getTrackColor(int(id)), getMarkerType(i), thickness=2)
                    # cv.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0), 4)

            if framenum % SKIP_FRAMES == 0:
                Zs = []
                positions, boxes, feature_vecs = detector.get_person_boxes(frame, i, framenum)
                for pos, box in zip(positions, boxes):
                    x0, y0, x1, y1 = box
                    Zs.append(np.array([[pos[0], pos[1], 20, 50]]).T)
                    cv.rectangle(frame, (int(x0),int(y0)), (int(x1),int(y1)), (255,0,0), 2)

                a.local_data_association(Zs, feature_vecs)
                observations += a.get_observations()  

                cv.imshow(f"frame{i}", frame)

        if framenum % SKIP_FRAMES == 0:
            for a in agents:
                a.add_observations(observations)
                a.dkf()
                a.tracker_manager()
                if args.debug:
                    with open(args.debug, 'a') as f:
                        np.set_printoptions(precision=1)
                        print(a, file=f)

        if framenum == 0:
            α = 1. / len(firstframes)
            topview = np.zeros_like(firstframes[0], dtype=np.float64)
            topview = np.concatenate([topview, topview], 0)
            topview = np.concatenate([topview, topview], 1)
            for img, H in zip(firstframes, Hs):
                topview += α * cv.warpPerspective(img, H, topview.shape[1::-1])
            topview = topview.astype(np.uint8)

            # cv.imshow('topview', topview)

        else:
            combined = topview.copy()

            gt_pts = dict()
            for i, (GT, a, H) in enumerate(zip(GTs, agents, Hs)):

                # get ground truth annotations of this frame
                gt = GT[GT[:,5]==framenum,:]

                for row in gt:
                    gt_id, x0, y0, x1, y1, _, lost, occluded, generated = row

                    if not lost and not occluded:
                        x, y = x0 + (x1 - x0) / 2, y1
                        pt = projectToGroundPlane(x, y, H)
                        if gt_id not in gt_pts:
                            gt_pts[gt_id] = []
                        gt_pts[gt_id].append(np.array(pt))

                Xs, colors = a.get_trackers()   

                for X, color in zip(Xs, colors):
                    x, y, w, h, _, _ = X.reshape(-1).tolist()
                    cv.circle(combined, (int(x),int(y)), int(w/2), color, 4)
            
            gt_avgs = dict()   
            for gt_id, pt_list in gt_pts.items():
                gt_avg = sum(pt_list) / len(pt_list)
                gt_avgs[gt_id] = gt_avg
                pt = (int(gt_avg[0]), int(gt_avg[1]))
                cv.drawMarker(combined, pt, getTrackColor(int(gt_id)), getMarkerType(i), thickness=2)
                
            for a, me in zip(agents, mes):
                me.update(gt_avgs, a.get_trackers(format='dict'))
                # if framenum % 1000 == 0:
                #     me.display_results()

            if framenum % SKIP_FRAMES == 0:
                cv.imshow('topview', combined)

        cv.waitKey(5)
        if framenum > 4950:
            break

    cap.release()
    cv.destroyAllWindows()
    
    print('FINAL RESULTS')
    for me in mes:
        me.display_results()