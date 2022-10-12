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



    videopath = root /  args.video_file
    cap = cv.VideoCapture(videopath.as_posix())



    annopath = root / args.truth_file
    GT = np.loadtxt(annopath, usecols=(0,1,2,3,4,5,6,7,8))


    H = np.array([
            [0.6174778372,   -0.4836875683,   147.00510919005],
            [0.5798503075,    3.8204849039,  -386.096405131],
            [0.0000000001,    0.0077222239,    -0.01593391935]
        ])


    framenum = -1
    while cap.isOpened():
        framenum += 1
        ret, frame = cap.read()

        # get ground truth annotations of this frame
        gt = GT[GT[:,5]==framenum,:]


        frame = cv.warpPerspective(frame, H, frame.shape[:2])

        # for row in gt:
        #     id, x0, y0, x1, y1, _, lost, occluded, generated = row

        #     if not lost:

        #         cv.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0), 4)

            # import ipdb; ipdb.set_trace()

        if ret:
            cv.imshow('frame', frame)
            cv.waitKey(1)
        else:
            break

    cap.release()
    cv.destroyAllWindows()