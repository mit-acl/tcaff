import numpy as np
from abc import ABC, abstractclassmethod
from typing import List
import json

from motlee.utils.transform import transform
from motlee.realign.object_map import ObjectMap

from robotdatapy.exceptions import NoDataNearTimeException
from robotdatapy.data import RobotData, ImgData
from robotdatapy.camera import CameraParams
# from fastsam3D.fastsam3D import FastSAM3D
from motlee.fastsam3D.fastsam3D import FastSAM3D

try:
    from .person_detector_3d import PersonDetector3D
except:
    from person_detector_3d import PersonDetector3D

# R_SHOULD_BE_PARAM = np.eye(3)*.3**2
R_SHOULD_BE_PARAM = np.eye(5)*.3**2

class Detections(ABC):

    @abstractclassmethod
    def detections(self, t: float):
        pass

class PersonDetector3DDetections(Detections):

    def __init__(
            self, 
            person_detector_3d: PersonDetector3D, 
            color_data: ImgData,
            depth_data: ImgData, 
            camera_params: CameraParams
        ):
        self.person_detector_3d = person_detector_3d
        self.color_data = color_data
        self.depth_data = depth_data
        self.camera_params = camera_params
    
    def detections(self, t: float):
        try:
            detections = self.person_detector_3d.detect(self.color_data.img(t), self.depth_data.img(t))
        except NoDataNearTimeException:
            detections = []
        return transform(self.camera_params.T, detections)
    
class MultiCameraPersonDetector3DDetections(Detections):

    def __init__(self, detections_list: List[PersonDetector3DDetections]):
        self.detections_list = detections_list

    def detections(self, t: float):
        detections_all = []
        for detections_i in self.detections_list:
            detections_all += detections_i.detections(t)
        return detections_all
    
class FastSAM3DDetections(Detections):

    def __init__(
            self, 
            fastsam3D: FastSAM3D, 
            color_data: ImgData,
            depth_data: ImgData, 
            camera_params: CameraParams
        ):
        self.fastsam3D = fastsam3D
        self.color_data = color_data
        self.depth_data = depth_data
        self.camera_params = camera_params
    
    def detections(self, t: float):
        try:
            measurements = self.fastsam3D.run(self.color_data.img(t), self.depth_data.img(t), self.camera_params.K)
            detections = ObjectMap(
                centroids=transform(self.T_BC, measurements['detections']), 
                widths=measurements['widths'], 
                heights=measurements['heights'], 
                ages=np.zeros(len(measurements['detections'])))
        except NoDataNearTimeException:
            detections = []
        return detections
    
    
class DetectionData(RobotData, Detections):

    def __init__(self, data_file, file_type, time_tol=1.0, t0=None, T_BC=np.eye(4), zmin=0., zmax=10., R=np.eye(5)*.3**2, dim=3):
        super().__init__(time_tol=time_tol, interp=False)
        if file_type == 'json':
            self._extract_json_data(data_file)
        else:
            assert False, "file_type not supported, please choose from: json"
        if t0 is not None:
            self.set_t0(t0)
        self.T_BC = T_BC
        self.zmin = zmin
        self.zmax = zmax
        self.R = R
        self.dim = dim
        
    def _extract_json_data(self, json_file):
        """
        Extracts Detection data from JSON file.

        Args:
            json_file (str): JSON file path
        """
        f = open(json_file)
        data = json.load(f)
        f.close()

        self.set_times(np.array([item['time'] for item in data]))
        self._detections = [item['detections'] for item in data]
        self._widths = [item['widths'] for item in data if 'widths' in item]
        self._heights = [item['heights'] for item in data if 'heights' in item]
        
    def detections(self, t):
        # TODO: dist filtering magic numbers!!
        detections = np.array(self._detections[self.idx(t)])
        if len(detections) == 0:
            return [], []
        transformed_centroids = np.array([np.array(z).reshape(-1)[:self.dim-2] for z in transform(self.T_BC, detections, stacked_axis=0).tolist()])
        if len(self._widths) > 0 and len(self._heights) > 0:
            zs = np.hstack((transformed_centroids, np.array(self._widths[self.idx(t)]).reshape((-1,1)), np.array(self._heights[self.idx(t)]).reshape((-1,1))))
        
        Rs = np.array([self.R.copy() for z in zs])

        # Rs = np.delete(Rs, np.bitwise_or(zs[:,0] > 15, zs[:,0] < 0.), axis=0) # TODO: < 1.5???
        # zs = np.delete(zs, np.bitwise_or(zs[:,0] > 15, zs[:,0] < 0.), axis=0)
        Rs = np.delete(Rs, np.bitwise_or(zs[:,0] > self.zmax, zs[:,0] < self.zmin), axis=0) # TODO: < 1.5???
        zs = np.delete(zs, np.bitwise_or(zs[:,0] > self.zmax, zs[:,0] < self.zmin), axis=0)
    
        return zs, Rs
    
class MultiDetectionData(RobotData, Detections):

    def __init__(self, data_files, file_types, T_BCs, time_tol=1.0, t0=None):
        self.detection_data = []
        for df, ft, T_BC in zip(data_files, file_types, T_BCs):
            self.detection_data.append(
                DetectionData(df, ft, time_tol, t0, T_BC)
            )
        
    def detections(self, t):
        dets = []
        Rs = []
        for dd in self.detection_data:
            # try:
            ddi = dd.detections(t)
            dets += ddi[0]
            Rs += ddi[1]
            # except:
            #     print(ddi)
        return dets, Rs
        # print([d for dd in self.detection_data for (d, _) in dd.detections(t)])
        # return [d for dd in self.detection_data for (d, _) in dd.detections(t)], \
        #        [R for dd in self.detection_data for (_, R) in dd.detections(t)]