import argparse
import pathlib

import numpy as np
import cv2 as cv

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

    caps = []
    for i in range(4):
        videopath = root /  f"terrace1-c{i}.avi"
        caps.append(cv.VideoCapture(videopath.as_posix()))

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

    Hs = [H0, H1, H2, H3]


    framenum = -1
    while caps[0].isOpened():
        framenum += 1

        for i in range(4):

            ret, frame = caps[i].read()

            if not ret:
                break

            frame = cv.warpPerspective(frame, Hs[i], frame.shape[:2])
            cv.imshow(f"frame{i}", frame)
        cv.waitKey(100)

    cap.release()
    cv.destroyAllWindows()