import numpy as np

from motlee.mot.multi_object_tracker import MultiObjectTracker

class StaticObjParams():
    '''
    Object parameter class
    Covariances may need some tuning
    '''
    def __init__(self, ts=.1):
        self.ts = ts
        self.A = np.array([
            [1, 0],
            [0, 1],
        ], dtype=np.float64)
        self.H = np.eye(2)
        self.Q = .8*np.eye(2) #.25*np.eye(2)
        self.P = .25*np.eye(2)

        self.n_dets = 1 # how many recent detections to store in bank (not needed for mapping)
        
class MapperParams():
    """
    Mapper parameter class
    """
    def __init__(self):
        self.Tau_LDA = 2.0 # Mahalanobis distance threshold
        self.Tau_GDA = 0. # not used
        self.alpha = 2000 # data association parameter, leave at 2000
        self.kappa = 5 * 30 # number of timesteps to let objects exist (deletes after kappa timesteps)
        self.n_meas_to_init_track = 3 # requires this many measurements before an object is instantiated
        self.merge_range_m = 0. # can be used to merge duplicate detections
    

class Mapper():
    
    def __init__(self, robot_id=0, tau=2.0, kappa=150, meas_to_init_obj=3):
        """
        Mapper object to create a local map

        Args:
            robot_id (int, optional): Unique robot identifier. Defaults to 0.
            tau (float, optional): Mahalanobids distance threshold for which a detection can be 
                associated with an existing object. Defaults to 2.0.
            kappa (int, optional): Number of timesteps that can pass without a detection before 
                an object is deleted. Defaults to 150.
            meas_to_init_obj (int, optional): Number of detections needed to create a new object 
                in map. Defaults to 3.
        """
        
        mapper_params = MapperParams()
        mapper_params.Tau_LDA = tau
        mapper_params.kappa = kappa
        mapper_params.n_meas_to_init_track = meas_to_init_obj
        
        self.mot = MultiObjectTracker(camera_id=robot_id, connected_cams=[], params=mapper_params, track_params=StaticObjParams())
        
    def update(self, zs, Rs):
        """
        Mapper update

        Args:
            zs (list of numpy arrays with shape(2 or 3, 1)): measurements in local or global frame
            Rs (list of numpy arrays with shape(2 or 3, 2 or 3)): measurement covariances in local or global frame
        """
        self.mot.local_data_association(zs, np.arange(len(zs)), Rs)
        self.mot.dkf()
        self.mot.track_manager()
        
    def map_as_array(self):
        """
        Returns numpy array of map objects

        Returns:
            numpy array shape(n, 2 or 3): map as an array
        """
        return np.array(self.mot.get_tracks(format='list'))
        