from enum import Enum

Tau_LDA = .8
Tau_GDA = .75
# Tau_LDA = 2.5
# Tau_GDA = 2.0
alpha = 2000
kappa = 4
n_meas_to_init_track = 3

DEG_2_M = 8.1712
detections_min_num = 100
transform_mag_unity_tolerance = 1.0
tolerance_growth_rate = 2.5
# detections_min_num = 20 # dynamic
# tolerance_growth_rate = 1.25 # dynamic

class RealignAlgorithm(Enum):
    REALIGN_LS = 'least-squares-realignment'
    REALIGN_WLS = 'weighted-least-squares-realignment'
    REALIGN_WLS_REACTIVE_TAU = 'wls-reactive-tolerance-realignment'
    REALIGN_CONTINUOUS = 'continuous-realignment'
    
realign_algorithm = RealignAlgorithm.REALIGN_CONTINUOUS