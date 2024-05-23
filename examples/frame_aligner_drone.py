import numpy as np
import matplotlib.pyplot as plt

import motlee.realign.frame_aligner
from motlee.utils.transform import transform, xypsi_2_transform

import argparse
import os
import pickle

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description='Mapper')
    parser.add_argument("dir", help="Pkl file.")
    args = parser.parse_args()

    # Read maps
    with open(os.path.join(args.dir, 'map_array_frame_500_to_1000.pkl'), 'rb') as f:
        map1 = np.array(pickle.load(f))
    # with open(os.path.join(args.dir, 'map_array_frame_500_to_1000_case1.pkl'), 'rb') as f:
    with open(os.path.join(args.dir, 'map_array_frame_500_to_1000_case2.pkl'), 'rb') as f:
        map2 = np.array(pickle.load(f))
    
    # Read object ground truth
    with open(os.path.join(args.dir, 'object_gt_mean.pkl'), 'rb') as f:
        object_gt_mean = pickle.load(f)
    
    # Find Transformation btwn maps
    frame_aligner = motlee.realign.frame_aligner.FrameAligner(
        method=motlee.realign.frame_aligner.AssocMethod.CLIPPER,
        num_objs_req=8,
        clipper_epsilon=.5,
        clipper_sigma=.5
    )
    sol = frame_aligner.align_objects(static_objects=[map1, map2])
    print(sol)
    map2_frame_align = transform(sol.transform, map2, stacked_axis=0)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(map1[:,0], map1[:,1], color='red', label='map1')
    ax.scatter(map2_frame_align[:,0], map2_frame_align[:,1], color='blue', label='map2 w/ estimated alignment')
    # plot object gt mean
    for idx, obj_pos in enumerate(object_gt_mean.values()):
        if idx == 0:
            plt.plot(obj_pos[0], obj_pos[1], 'x', color='k', markersize=10, markeredgewidth=2, label='object gt')
        else:
            plt.plot(obj_pos[0], obj_pos[1], 'x', color='k', markersize=10, markeredgewidth=2)

    ax.legend()
    plt.grid()
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.savefig(os.path.join(args.dir, 'frame_aligner_drone.png'))
    # plt.show()

if __name__ == "__main__":
    main()