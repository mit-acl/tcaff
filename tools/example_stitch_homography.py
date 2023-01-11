import argparse
import pathlib

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



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
    args = parser.parse_args()



    root = pathlib.Path(args.root)


    # (288, 360) (h, w)
    numCams = 4
    caps = getVideos(root)
    GTs = getAnnotations(root)
    Hs = getHomographies()

    COLORS = [(255,0,0), (0,255,0), (0,0,255), (255, 255, 255)]

    framenum = -1
    firstframes = []
    warpcams = False
    while caps[0].isOpened():
        framenum += 1

        current_frames = []
        for i, (cap, GT, H) in enumerate(zip(caps, GTs, Hs)):
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
                        pt = H @ np.r_[x, y, 1]
                        pt /= pt[2]
                        pt = pt.astype(int)
                        pt = (pt[0], pt[1])
                    else:
                        pt = int(x), int(y)

                    cv.drawMarker(frame, pt, getTrackColor(int(id)), getMarkerType(i))
                    # cv.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0), 4)

            cv.imshow(f"frame{i}", frame)

        if framenum == 0:
            α = 1. / len(firstframes)

            topview = np.zeros_like(firstframes[0], dtype=np.float64)
            for img, H in zip(firstframes, Hs):
                topview += α * cv.warpPerspective(img, H, img.shape[1::-1])
            topview = topview.astype(np.uint8)

            cv.imshow('topview', topview)

        else:
            combined = topview.copy()

            for i, (GT, H, color) in enumerate(zip(GTs, Hs, COLORS)):

                # get ground truth annotations of this frame
                gt = GT[GT[:,5]==framenum,:]

                for row in gt:
                    id, x0, y0, x1, y1, _, lost, occluded, generated = row

                    if not lost and not occluded:
                        x, y = x0 + (x1 - x0) / 2, y1

                        pt = H @ np.r_[x, y, 1]
                        pt /= pt[2]
                        pt = pt.astype(int)
                        pt = (pt[0], pt[1])
                        # import ipdb; ipdb.set_trace()

                        cv.drawMarker(combined, pt, getTrackColor(int(id)), getMarkerType(i))

            cv.imshow('groundtruth', combined)

        cv.waitKey(15)

    cap.release()
    cv.destroyAllWindows()