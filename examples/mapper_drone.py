#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mapper import Mapper
import pickle
import argparse
import os
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from scipy.spatial.transform import Rotation as R
import quaternion

def get_quaternion_from_euler(roll, pitch, yaw): #https://automaticaddison.com/how-to-convert-euler-angles-to-quaternions-using-python/
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return np.quaternion(qw, qx, qy, qz)

def quaternion_multiply(quaternion1, quaternion0): #https://stackoverflow.com/questions/39000758/how-to-multiply-two-quaternions-by-python-or-numpy
    w0 = quaternion0.w
    x0 = quaternion0.x
    y0 = quaternion0.y
    z0 = quaternion0.z
    w1 = quaternion1.w
    x1 = quaternion1.x
    y1 = quaternion1.y
    z1 = quaternion1.z
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def get_rotation_matrix(roll, pitch, yaw):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    R_r = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_p = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_y = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    R = np.matmul(R_y, np.matmul(R_p, R_r))
    return R

def corrupt_poses(poses):

    # define yaw drift and x,y,z translation drift
    yaw_drift_deg = 45.0
    x_drift = 0.0
    y_drift = 0.0
    z_drift = 0.0

    # corrupt poses
    new_poses = []
    for pose in poses:
    
        # note that (to calculate A*(B*point), you cannot do (A*B)*point))
        # convert pose to transformation matrix
        # T_original = np.zeros((4,4))
        # T_original[:3, :3] = np.linalg.inv(R.from_quat(pose[3:]).as_matrix()) #from_quat() takes as input a quaternion in the form [x, y, z, w]
        # T_original[:3, 3] = -np.matmul(T_original[:3, :3], np.array(pose[:3]))
        # T_original[3, 3] = 1.0

        # # corrupt pose
        # T_corrupt = np.zeros((4,4))
        # T_corrupt[:3, :3] = np.linalg.inv(get_rotation_matrix(0, 0, yaw_drift_deg))
        # T_corrupt[:3, 3] = -np.matmul(T_corrupt[:3, :3], np.array([x_drift, y_drift, z_drift]))
        # T_corrupt[3, 3] = 1.0
        # T_new = np.matmul(T_corrupt, T_original)

        # t_new = T_new[:3, 3]
        # q_new = R.from_matrix(T_new[:3, :3]).as_quat()
        # new_poses.append([t_new[0], t_new[1], t_new[2], q_new[0], q_new[1], q_new[2], q_new[3]])

        # corrupt pose
        q_original = np.quaternion(pose[6], pose[3], pose[4], pose[5])
        q_corrupt = quaternion_multiply(get_quaternion_from_euler(0, 0, yaw_drift_deg), q_original)

        # new_poses.append([pose[0] + x_drift, pose[1] + y_drift, pose[2] + z_drift, pose[3], pose[4], pose[5], pose[6]])
        new_poses.append([pose[0] + x_drift, pose[1] + y_drift, pose[2] + z_drift, q_corrupt[1], q_corrupt[2], q_corrupt[3], q_corrupt[0]])

    return new_poses

def corrupt_positions(positions):

    # corrupt poses
    new_positions = []
    for position in positions:
        new_positions.append(corrupt_points(position))
    return new_positions

def corrupt_points(points):

    # define yaw drift and x,y,z translation drift
    yaw_drift_rad = np.deg2rad(0.0)
    x_drift = 0.0
    y_drift = 0.0
    z_drift = 0.0

    # corrupt poses
    new_points = []
    for point in points:
    
        T_corrupt = np.zeros((4,4))
        T_corrupt[:3, :3] = R.from_euler('XYZ', [0, 0, yaw_drift_rad]).as_matrix() #from_quat() takes as input a quaternion in the form [x, y, z, w]
        T_corrupt[:3, 3] = np.array([x_drift, y_drift, z_drift])
        T_corrupt[3, 3] = 1.0

        point_homogeneous = np.array([point[0], point[1], 0.0, 1.0])
        new_points.append(np.matmul(T_corrupt, point_homogeneous)[:2].tolist())

    return new_points

def quaternion_to_euler_angle_vectorized1(w, x, y, z): #https://stackoverflow.com/questions/56207448/efficient-quaternions-to-euler-transformation
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2

    t2 = np.where(t2<-1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)

    return X, Y, Z 

def plot_pos_and_fov(p, label=None):
    # get yaw
    roll, pitch, yaw = quaternion_to_euler_angle_vectorized1(p[6], p[3], p[4], p[5])
    
    # get triangle vertices for fov
    horizontal_angle = np.deg2rad(133) # horizontal fov https://docs.modalai.com/M0014/
    theta = 0.5 * horizontal_angle
    depth = 2.0 # there's no depth in image camera so we just set it to 5m
    l = depth / np.cos(theta)

    point1 = p[:2]
    point2 = p[:2] + np.array([l*np.cos(yaw+theta), l*np.sin(yaw+theta)])
    point3 = p[:2] + np.array([l*np.cos(yaw-theta), l*np.sin(yaw-theta)])

    # plot vehicle position
    # point1 = corrupt_points([point1])[0]
    plt.plot(point1[0], point1[1], 'o', color='r', label=label)

    # plot fov
    coord1 = [point1, point2]
    coord3 = [point1, point3]
    xs1, ys1 = zip(*coord1) #create lists of x and y values
    xs3, ys3 = zip(*coord3) #create lists of x and y values
    # plt.plot(xs1, ys1, color='r', alpha=0.7)
    # plt.plot(xs3, ys3, color='r', alpha=0.7)
    
    # plt.arrow(p[1], p[2], 0.5*np.cos(yaw), 0.5*np.sin(yaw), color='r', width=.01)

def plot_map(args, filename, map_array, veh_pose, mapper_starting_frame, mapper_num_frame, object_gt_mean, show_plot=True):

    plt.figure(figsize=(20,20))

    # plot vehicle position and fov as a rectangle
    cnt = 0
    for p in veh_pose[mapper_starting_frame:mapper_starting_frame+mapper_num_frame]:
        if cnt % 10 == 0:  
            plot_pos_and_fov(p)
        cnt += 1

    # make sure the last pose is plotted
    plot_pos_and_fov(veh_pose[mapper_starting_frame+mapper_num_frame-1], label='vehicle')

    # plot object gt mean
    for idx, obj_pos in enumerate(object_gt_mean.values()):
        if idx == 0:
            plt.plot(obj_pos[0], obj_pos[1], 'x', color='b', markersize=20, markeredgewidth=10, label='object gt')
        else:
            plt.plot(obj_pos[0], obj_pos[1], 'x', color='b', markersize=20, markeredgewidth=10)
    
    # plot the map
    plt.plot(map_array[:,0], map_array[:,1], 'o', color='g', markersize=20, label='map')

    # legend
    plt.legend(fontsize=10)

    # plot the map
    plt.title(filename[:-4])
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.grid()
    plt.savefig(os.path.join(args.dir, filename))
    if show_plot:
        plt.show()
    plt.close()

def plot_cosecutive_frames(args, positions, veh_pose, first_frame, starting_frame, num_frames, object_gt_mean, show_plot=True):
    
    # generate num_frames random colors
    colors = []
    for i in range(num_frames):
        colors.append(np.random.rand(3,))

    # plot the map
    plt.figure(figsize=(20,20))

    for frame_idx in range(starting_frame, starting_frame+num_frames):
        for idx, obj_pos in enumerate(positions[frame_idx]):
            if idx == 0:
                # plt.plot(obj_pos[0], obj_pos[1], 'o', color=colors[frame_idx], label='frame '+str(frame_idx))
                plt.plot(obj_pos[0], obj_pos[1], 'o', color=colors[frame_idx-starting_frame])
            else:
                plt.plot(obj_pos[0], obj_pos[1], 'o', color=colors[frame_idx-starting_frame])
    
    # plot vehicle position and fov as a rectangle
    cnt = 0
    for p in veh_pose[starting_frame:starting_frame+num_frames]:
        if cnt % 10 == 0:
            plot_pos_and_fov(p)
        cnt += 1

    # make sure the last pose is plotted
    plot_pos_and_fov(veh_pose[starting_frame+num_frames-1], label='vehicle')

    # plot object gt mean
    for idx, obj_pos in enumerate(object_gt_mean.values()):
        if idx == 0:
            plt.plot(obj_pos[0], obj_pos[1], 'x', color='b', markersize=20, markeredgewidth=10, label='object gt')
        else:
            plt.plot(obj_pos[0], obj_pos[1], 'x', color='b', markersize=20, markeredgewidth=10)

    # legend
    plt.legend()
    
    # plot the map
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    # plt.axis("equal")
    plt.grid()
    plt.savefig(os.path.join(args.dir, f"raw_global_map_{starting_frame+first_frame}_to_{first_frame+num_frames}.png"))
    if show_plot:
        plt.show()
    plt.close()

def main():

    # Parameters
    PLOT_CONSECUTIVE_FRAMES = True # plot n consecutive frames of the map
    FIRST_FRAME = 500 #frame 500
    STARTING_FRAME = 500 - FIRST_FRAME
    NUM_FRAMES = 500
    MAPPER_STARTING_FRAME = 500 - FIRST_FRAME
    MAPPER_NUM_FRAME = 500
    USE_RAW_DATA = False # use raw data (position) or mapper-processed data
    IS_MAP_IN_GLOBAL_FRAME = True # plot map in global frame or vehicle frame

    # Parse arguments
    parser = argparse.ArgumentParser(description='Mapper')
    parser.add_argument("dir", help="Directory.")
    parser.add_argument("bag_file", help="Ground truth rosbag.")
    parser.add_argument("camera", help="Camera name.")
    args = parser.parse_args()

    # Create a mapper object 
    mapper = Mapper(kappa=400)
    
    # Read list of obstacles
    # with open(os.path.join(args.dir, 'positions.pkl'), 'rb') as f:
    # with open(os.path.join(args.dir, 'case1-corrupted-positions.pkl'), 'rb') as f:
    with open(os.path.join(args.dir, 'case2-corrupted-positions.pkl'), 'rb') as f:
        positions = pickle.load(f)

    # Read list of colors
    with open(os.path.join(args.dir, 'colors.pkl'), 'rb') as f:
        colors = pickle.load(f)
    
    # Read list of poses (csv files)
    veh_pose = []
    filenames = os.listdir(os.path.join(args.dir, f'csvs/{args.camera}'))
    filenames.sort()
    for file in filenames:
        with open(os.path.join(args.dir, f'csvs/{args.camera}', file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                veh_pose.append([float(x) for x in line.split(',')])

    # Corrupt the vehicle pose
    veh_pose = corrupt_poses(veh_pose) #not sure why it's not working
    # Corrupt the positions
    # positions = corrupt_positions(positions)

    # verify the size of positions and veh_pose are the same
    if len(positions) != len(veh_pose):
        print("len(positions): ", len(positions))
        print("len(veh_pose): ", len(veh_pose))
        raise ValueError("len(positions) != len(veh_pose)")

    # Get the ground truth map from rosbag
    # get the bag
    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    
    # topic name
    topic_names = []
    for obj_num in range(1, 14):
        topic_names.append(f"/OBJ{obj_num}/world")

    # get the ground truth of objects (TODO: no need to take all the recorded positions because they are static)
    object_gt = {}
    for topic_name in topic_names:
        object_gt[topic_name] = []
        for topic, msg, t in bag.read_messages(topics=topic_name):
            # save images
            if topic == topic_name:
                object_gt[topic_name].append((msg.pose.position.x, msg.pose.position.y, msg.pose.position.z))

    # get the mean of the ground truth of objects
    object_gt_mean = {}
    for topic_name in topic_names:
        object_gt_mean[topic_name] = np.mean(object_gt[topic_name], axis=0)

    # write the mean of the ground truth of objects to a pkl file
    with open(os.path.join(args.dir, 'object_gt_mean.pkl'), 'wb') as f:
        pickle.dump(object_gt_mean, f)

    # Plot n consecutive frames of the map
    if PLOT_CONSECUTIVE_FRAMES:
        plot_cosecutive_frames(args, positions, veh_pose, FIRST_FRAME, STARTING_FRAME, NUM_FRAMES, object_gt_mean, show_plot=True)
        exit()

    # Update the mapper
    for i in range(MAPPER_STARTING_FRAME, MAPPER_STARTING_FRAME+MAPPER_NUM_FRAME):
        zs = positions[i]
        Rs = [np.eye(2)*.01 for z in zs]
        zs = [np.array(z).reshape((2,1)) + np.random.multivariate_normal(np.zeros(2), R).reshape((2,1)) for z, R in zip(zs, Rs)]
        mapper.update(zs, Rs)

    # Plot map 
    if IS_MAP_IN_GLOBAL_FRAME:

        # use raw data or mapper-processed data
        if USE_RAW_DATA:
            map_array = np.array(positions[0])
        else:
            map_array = mapper.map_as_array()

        # plot the map
        filename = f"global_map_{MAPPER_STARTING_FRAME+FIRST_FRAME}_to_{MAPPER_STARTING_FRAME+FIRST_FRAME+MAPPER_NUM_FRAME}.png"
        
        plot_map(args, filename, map_array, veh_pose, MAPPER_STARTING_FRAME, MAPPER_NUM_FRAME, object_gt_mean, show_plot=True)

        # save the map_array as pickle file
        with open(os.path.join(args.dir, f'map_array_frame_{MAPPER_STARTING_FRAME+FIRST_FRAME}_to_{MAPPER_STARTING_FRAME+FIRST_FRAME+MAPPER_NUM_FRAME}_case2.pkl'), 'wb') as f:
            pickle.dump(map_array, f)

    else: # keep in mind this will only work if you visualize the map using only one frame. Otherwise, the agent will move and point of view will shift
        
        # use raw data or mapper-processed data
        if USE_RAW_DATA:
            tmp_array = np.array(positions[0])
        else:
            tmp_array = mapper.map_as_array()
        
        # get yaw
        roll, pitch, yaw = quaternion_to_euler_angle_vectorized1(veh_pose[0][7], veh_pose[0][4], veh_pose[0][5], veh_pose[0][6])
        
        # get rotation matrix
        rotation_matrix = np.array([[np.cos(-yaw+np.deg2rad(90)), -np.sin(-yaw+np.deg2rad(90))], [np.sin(-yaw+np.deg2rad(90)), np.cos(-yaw+np.deg2rad(90))]])

        # rotate the map to the first person view
        map_array = np.zeros((len(mapper.map_as_array()), 2))
        for idx, pos in enumerate(tmp_array):
            map_array[idx,:] = np.matmul(rotation_matrix, (pos-np.array(veh_pose[0][0:2])))
        
        # plot the map
        plt.figure()
        for idx in range(len(map_array)):
            plt.plot(map_array[idx,0], map_array[idx,1], 'o', color=colors[idx])
        
        # plot vehicle position and orientatio as a red arrow
        plt.arrow(0, 0, 0, 0.5, color='r', width=.1)
        
        # plot the origin as a black cross
        corrected_origin_pos = rotation_matrix @ (- np.array(veh_pose[0][0:2]))
        plt.plot(corrected_origin_pos[0], corrected_origin_pos[1], 'kx')
        
        plt.savefig(os.path.join(args.dir, "fpv_map.png"))
        # plt.show()
        plt.close()

    return

# main 
if __name__ == "__main__":
    main()