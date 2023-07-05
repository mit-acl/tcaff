from enum import Enum

class RoverMOTParams():
    
    def __init__(self):

        self.Tau_LDA = 2.0
        self.Tau_GDA = .75
        self.alpha = 2000
        self.kappa = 4
        self.n_meas_to_init_track = 3
        self.merge_range_m = .15
        self.cone_Tau = .5
        self.cone_max_life = 600
        self.cone_merge_range_m = 0.0

        self.DEG_2_M = 8.1712
        self.detections_min_num = 100
        self.transform_mag_unity_tolerance = 1.0
        self.tolerance_growth_rate = 2.5
        self.realign_algorithm = RealignAlgorithm.REALIGN_CONTINUOUS
        self.RealignAlgorithm = RealignAlgorithm
        self.ts_realign = 1.0

class RealignAlgorithm(Enum):
    REALIGN_LS = 'least-squares-realignment'
    REALIGN_WLS = 'weighted-least-squares-realignment'
    REALIGN_WLS_REACTIVE_TAU = 'wls-reactive-tolerance-realignment'
    REALIGN_CONTINUOUS = 'continuous-realignment'
    