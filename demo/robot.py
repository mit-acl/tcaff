import numpy as np
from typing import List, Dict
from dataclasses import dataclass

from motlee.realign.frame_align_filter import FrameAlignFilter
from motlee.mot.multi_object_tracker import MultiObjectTracker
from motlee.mot.track import Track
from motlee.utils.transform import transform, transform_covariances
from motlee.realign.object_map import ObjectMap

from robot_utils.robot_data.pose_data import PoseData
from robot_utils.robot_data.img_data import ImgData, CameraParams

try:
    from .detections import Detections
except:
    from detections import Detections

CENTROID_DIM = 3

class Robot():

    def __init__(
        self,
        name: str,
        neighbors: List[str],
        pose_est_data: PoseData,
        mapper: MultiObjectTracker,
        frame_align_filters: Dict[str, FrameAlignFilter],
        mot: MultiObjectTracker,
        fastsam3d_detections: Detections,
        person_detector_3d_detections: Detections,
    ) -> None:
        self.name = name
        self.neighbors = neighbors
        self.pose_est_data = pose_est_data
        self.mapper = mapper
        self.frame_align_filters = frame_align_filters
        self.mot = mot
        self.fastsam3d_detections = fastsam3d_detections
        self.person_detector_3d_detections = person_detector_3d_detections

        self.neighbor_maps = {robot: ObjectMap() for robot in self.neighbors}
        # self.neighbor_mot_info = []
        # camera_names: List[str],
        # camera_params: Dict[str, CameraParams],
        # camera_data: Dict[str, ImgData],
        # camera_uses: Dict[str, Dict[str, bool]],
        # mot_params: any,
        # self.camera_names = camera_names
        # self.camera_params = camera_params
        # self.camera_data = camera_data
        # self.camera_uses = camera_uses
        # self.mot_params = mot_params

    def update_mapping(self, t: float):
        dim = self.mapper.dim_association
        try:
            zs, Rs = self.fastsam3d_detections.detections(t)
        except:
            return # no fastsam detections available
        try:
            t_closest = self.fastsam3d_detections.get_vals(self.fastsam3d_detections.times, t)
        except:
            t_closest = t
        T_WB = self.pose_est_data.T_WB(t_closest)
        
        if len(zs) != 0:
            zs_centroids = transform(T_WB, zs[:,:3].reshape((-1,3)), stacked_axis=0)
            zs = np.hstack((zs_centroids, zs[:,3:])).reshape((-1,dim,1))
            Rs[:,:3,:3] = transform_covariances(T_WB, Rs[:,:3,:3])

            zs = [np.array(z) for z in zs.tolist()]
            Rs = [np.array(R) for R in Rs.tolist()]
        self.mapper.local_data_association(zs, np.arange(len(zs)), Rs)
        self.mapper.dkf()
        self.mapper.track_manager()
    
    def update_frame_alignments(self):
        # update MOT frame alignments
        for robot in self.neighbors:
            self.frame_align_filters[robot].update(self.get_map(), self.neighbor_maps[robot])
            if self.mot is not None:
                self.mot.neighbor_frame_align[robot] = self.frame_align_filters[robot].T
                self.mot.neighbor_frame_align_cov[robot] = self.frame_align_filters[robot].P

    def update_mot_local_info(self, t):
        zs, Rs = self.person_detector_3d_detections.detections(t)
        T_WB = self.pose_est_data.T_WB(t)
        
        zs = transform(T_WB, zs, stacked_axis=0)
        Rs = transform_covariances(T_WB, Rs)

        self.mot.local_data_association(zs, np.arange(len(zs)), Rs)

    def update_mot_global_info(self, mot_info):
        self.mot.add_observations([obs for obs in mot_info if obs.destination == self.mot.camera_id])
        self.mot.dkf()
        self.mot.track_manager()
        self.mot.groups_by_id = [] # TODO: what does this do?
    
    def get_map(self) -> ObjectMap:
        centroids = []
        widths = []
        heights = []
        ages = []
        
        track: Track
        for track in self.mapper.tracks:
            centroids.append(track.state[:CENTROID_DIM, :].reshape(-1))
            widths.append(track.state[CENTROID_DIM, :].reshape(-1))
            heights.append(track.state[CENTROID_DIM+1, :].reshape(-1))
            ages.append(track.ell)
        
        return ObjectMap(
            np.array(centroids), 
            np.array(widths), 
            np.array(heights), 
            np.array(ages)
        )

    def get_mot_info(self) -> list:
        return self.mot.get_observations()

    def set_neighbor_map(self, neighbor, neighbor_map):
        self.neighbor_maps[neighbor] = neighbor_map

    
