DATA_ROOT_PATH = '/home/masonbp/ford/data'
DYNAMIC_DATA_DIR = DATA_ROOT_PATH + '/mot_dynamic/dynamic_motlee_iros'
DYNAMIC_RUNS = [
    'run0_2023-02-06-18-21-24.bag' # rovers weren't driving
    'run1_2023-02-06-18-24-57.bag' # hard moving run
    'run2_2023-02-06-18-32-37.bag' # hard static run
    'run3_filtered.bag' # easy moving run
    'run4_2023-02-06-18-52-11.bag' # quick iphone vid
]
CAMERA_CALIB_FILE = DYNAMIC_DATA_DIR + '/calib/calib.yaml'