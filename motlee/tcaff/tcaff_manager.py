from motlee.realign.frame_align_filter import FrameAlignFilter

import numpy as np
from scipy.linalg import block_diag

from motlee.tcaff.tcaff import TCAFF
from motlee.tcaff.tree import Tree as TCAFFTree
from motlee.realign import frame_aligner
from motlee.realign.object_map import ObjectMap
from motlee.utils.transform import transform_2_xypsi, xypsi_2_transform

SCALING = 1
SE2_STORAGE_DIM = 3
# MAX_OPT_FRACTION = .75

class TCAFFManager(FrameAlignFilter):

    def __init__(
        self,
        prob_no_match=.001,
        exploring_branching_factor=2,
        window_len=5,
        max_branch_exp=50,
        max_branch_main=200,
        rho=1.1,
        clipper_epsilon=.5,
        clipper_sigma=.2, #.3
        clipper_mult_repeats=4,
        max_obj_width = 0.8,
        h_diff = 0.08,
        wh_scale_diff=1.25, #2.5
        num_objs_req=6,
        max_opt_fraction=.5
    ):
        super().__init__()

        self.fa = frame_aligner.FrameAligner(
            method=frame_aligner.AssocMethod.CLIPPER_MULT_SOL,
            num_objs_req=num_objs_req,
            clipper_epsilon=clipper_epsilon,
            clipper_sigma=clipper_sigma,
            clipper_mult_downweight=0.,
            clipper_mult_repeats=clipper_mult_repeats
        )

        self.faf = setup_frame_align_filter(
            window_len=window_len,
            max_leaves_exp=max_branch_exp,
            max_leaves_main=max_branch_main,
            rho=rho,
            prob_no_match=prob_no_match, 
            exploring_branching_factor=exploring_branching_factor
        )
        self.R = setup_measurement_model()

        self.wh_scale_diff = wh_scale_diff
        self.h_diff = h_diff
        self.latest_zs = []
        self.max_opt_fraction = max_opt_fraction
    
    def update(self, ego_map: ObjectMap, other_map: ObjectMap):

        measurements = self.get_frame_align_measurements(ego_map, other_map)

        faf_zs = [measurements[j,:3].reshape((3,1)) for j in range(len(measurements)) if not np.any(np.isnan(measurements[j,:3]))]
        faf_Rs = [self.R for j in range(len(measurements)) if not np.any(np.isnan(measurements[j,:]))]
        self.faf.update(faf_zs, faf_Rs)
        self.latest_zs = faf_zs

    @property
    def T(self):
        if self.faf.main_tree is None:
            return np.zeros((4,4)) * np.nan
        else:
            return xypsi_2_transform(*self.faf.main_tree.optimal.xhat[:3].reshape(-1).tolist())
        
    @property
    def P(self):
        if self.faf.main_tree is None:
            return np.zeros((3,3)) * np.nan
        else:
            return self.faf.main_tree.optimal.P[:3,:3]

    def get_frame_align_measurements(self, ego_map: ObjectMap, other_map: ObjectMap):
        A_put = self.get_putative_assoc(ego_map, other_map)
        
        # Solution filtering and data setup
        if len(ego_map) == 0 or len(other_map) == 0 or (A_put is not None and A_put.shape[0] == 0):
            return np.array([])
        else:
            sols = self.fa.align_objects(
                static_objects=[ego_map.centroids[:,:2], other_map.centroids[:,:2]], 
                static_ages=[ego_map.ages, other_map.ages], static_put_assoc=A_put)
            if len([None for sol in sols if sol.success]) == 0:
                return np.array([])
            else:
                align = []
                align_obj = []
                align_cov = []
                max_opt = max([sol.objective_score for sol in sols if sol.objective_score is not None])
                for sol in sols:
                    # filter out alignment failures and reflections
                    if sol.success and np.allclose(sol.transform[0,0], sol.transform[1,1]) and sol.objective_score > self.max_opt_fraction*max_opt:
                        align.append(np.array(transform_2_xypsi(sol.transform)))
                        align_obj.append(sol.objective_score)
                    else:
                        continue
                return np.array(align)

    def get_putative_assoc(self, map1: ObjectMap, map2: ObjectMap):
        A_all = np.zeros((len(map1) * len(map2), 2)).astype(np.int64)
        A_i = 0
        for i in range(len(map1)):
            for j in range(len(map2)):
                A_all[A_i,:] = [i, j]
                A_i += 1

        to_delete = []
        for i, pair in enumerate(A_all):
            widths = [map1.widths[pair[0]], map2.widths[pair[1]]]
            heights = [map1.heights[pair[0]], map2.heights[pair[1]]]
            if max(widths) > min(widths) * self.wh_scale_diff:
                to_delete.append(i)
            elif max(heights) > min(heights) * self.wh_scale_diff:
                to_delete.append(i)
            elif max(widths) - min(widths) > self.h_diff:
                to_delete.append(i)
            elif max(heights) - min(heights) > self.h_diff:
                to_delete.append(i)
                
        A_put = np.delete(A_all, to_delete, axis=0)
        return A_put

def setup_frame_align_filter(
    window_len=5,
    rho=3.0,
    exploring_branching_factor=2,
    max_leaves_exp=10,
    max_leaves_main=20,
    prob_no_match=.01
):
    ts = 1.0
    A = np.array([
        [1., ts],
        [0., 1.]
    ])

    H = np.array([
        [1., 0.]
    ])
    Q = np.array([
        [ts**4/4, ts**3/2],
        [ts**3/2, ts**2]
    ])
    P0 = np.array([
        [1., 0.],
        [0., 10.]
    ])

    A = block_diag(A, A, A)
    H = block_diag(H, H, H)
    Q = block_diag(Q, Q, Q)
    P0 = block_diag(P0, P0, P0)

    Q[:4,:4] *= .1
    Q[4:,4:] *= .5*np.pi/180

    P0[:4,:4] *= .5
    P0[4:,4:] *= 2*np.pi/180

    n = A.shape[0]
    p = H.shape[0]
    
    z2x_func = lambda z : np.array([[z.item(0), 0., z.item(1), 0., z.item(2), 0.]]).T

    A = np.eye(3)
    H = np.eye(3)
    Q = np.diag([.1, .1, .5*np.pi/180])*SCALING
    P0 = np.diag([.5, .5, 2*np.pi/180])*SCALING
    z2x_func = lambda z : np.array([[z.item(0), z.item(1), z.item(2)]]).T
    
    KMD = False
    if KMD:
        Q[:4,:4] *= .5
        Q[4:,4:] *= 2.5*np.pi/180

        P0[:4,:4] *= 1.
        P0[4:,4:] *= 5*np.pi/180

        n = A.shape[0]
        p = H.shape[0]
        
        z2x_func = lambda z : np.array([[z.item(0), 0., z.item(1), 0., z.item(2), 0.]]).T

        A = np.eye(3)
        H = np.eye(3)
        Q = np.diag([.5, .5, 2.5*np.pi/180])
        P0 = np.diag([1, 1., 5*np.pi/180])
        z2x_func = lambda z : np.array([[z.item(0), z.item(1), z.item(2)]]).T

    # frame_align_filter_manager = FrameAlignFilter(P0=P0, A=A, H=H, Q=Q, window_len_mini_filter=window_len_mini_filter, window_len=window_len, z2x_func=z2x_func, max_branching_factor_mini_filter=2, prob_no_match=.01)
    create_exploring_tree = lambda xhat0 : TCAFFTree(
        xhat0=xhat0, P0=P0, A=A, H=H, Q=Q, window_len=window_len, prob_no_match=prob_no_match, 
        max_branching_factor=exploring_branching_factor, max_tree_leaves=max_leaves_exp)
    create_main_tree = lambda xhat0 : TCAFFTree(
        xhat0=xhat0, P0=P0, A=A, H=H, Q=Q, window_len=window_len, prob_no_match=prob_no_match, 
        max_branching_factor=4, max_tree_leaves=max_leaves_main)
    frame_align_filter = TCAFF(z2x=z2x_func, K=window_len, create_exploring_tree=create_exploring_tree, create_main_tree=create_main_tree, rho=rho)
    return frame_align_filter

def setup_measurement_model():
    R = np.array([[1.]])*SCALING
    R = block_diag(R, R, R)
    R[:2,:2] *= .5
    R[2:,2:] *= 2*np.pi/180
    KMD = False
    if KMD:
        R[:2,:2] *= 1
        R[2:,2:] *= 5*np.pi/180
    # R[:2,:2] *= .2
    # R[2:,2:] *= .5*np.pi/180
    return R