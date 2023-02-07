from enum import Enum

Tau_LDA = .8
Tau_GDA = .75
alpha = 2000
kappa = 4
n_meas_to_init_tracker = 3

DEG_2_M = 8.1712
detections_min_num = 100
transform_mag_unity_tolerance = 1.0
# TAU_GROWTH_MIN_TRACKERS = 5
tolerance_growth_rate = 2.5
# TAU_GDA_GROWTH_FACTOR = 1.0

class RealignAlgorithm(Enum):
    REALIGN_LS = 'least-squares-realignment'
    REALIGN_WLS = 'weighted-least-squares-realignment'
    REALIGN_WLS_REACTIVE_TAU = 'wls-reactive-tolerance-realignment'
    REALIGN_CONTINUOUS = 'continuous-realignment'
    
realign_algorithm = RealignAlgorithm.REALIGN_CONTINUOUS