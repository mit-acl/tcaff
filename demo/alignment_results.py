from dataclasses import dataclass, field
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from tcaff.tcaff.se2_align_plot import se2_align_plot
from tcaff.tcaff.tcaff_manager import TCAFFManager
from tcaff.utils.transform import transform_2_xypsi, xypsi_2_transform

@dataclass
class AlignmentResults():
    
    times: List[float] = field(default_factory=list) 
    fa: List[List[List[float]]] = field(default_factory=list) 
    gt: List[List[float]] = field(default_factory=list) 
    est: List[List[float]] = field(default_factory=list) 

    def __post_init__(self):
        self.n = len(self.times)
        ms = [len(fa) for fa in self.fa]
        if len(ms) > 0:
            self.m = max(ms)
        else:
            self.m = 0
        assert len(self.fa) == self.n and len(self.gt) == self.n and len(self.est) == self.n, "All lists must be same length"

        self.no_align = (np.zeros(3)*np.nan).tolist()

        self.errors_translation = []
        self.errors_rotation = []

    def update_from_tcaff_manager(self, t: float, tcaff_manager: TCAFFManager, gt: List[float]=None):
        fa = [z.reshape(-1) for z in tcaff_manager.latest_zs]
        est = transform_2_xypsi(tcaff_manager.T)
        if np.any(np.isnan(tcaff_manager.T)):
            est = None
        self.update(t, fa, gt, est)

    def update(self, t: float, fa: List[List[float]]=None, gt: List[float]=None, est: List[float]=None):        
        if fa is not None and len(fa) > self.m:
            self.m = len(fa)
        if fa is None:
            fa = []
        if gt is None:
            gt = self.no_align
        if est is None:
            est = self.no_align
        
        self.times.append(t)
        self.fa.append(fa)
        self.gt.append(gt)
        self.est.append(est)

        T_diff = np.linalg.inv(xypsi_2_transform(*gt)) @ xypsi_2_transform(*est)
        if np.any(np.isnan(T_diff)):
            diff_x, diff_y, diff_th = np.nan, np.nan, np.nan
        else:
            diff_x, diff_y, diff_th = transform_2_xypsi(T_diff)
        self.errors_translation.append(np.linalg.norm([diff_x, diff_y]))
        self.errors_rotation.append(np.abs(diff_th))

        self.n += 1

    def get_Tij_gt(self, t, pd_gt_0, pd_gt_1, pd_est_0, pd_est_1, xytheta=True):
        T0_gt = pd_gt_0.T_WB(t)
        T1_gt = pd_gt_1.T_WB(t)
        T0_est = pd_est_0.T_WB(t)
        T1_est = pd_est_1.T_WB(t)
        if T0_gt is None or T0_est is None or \
            T1_gt is None or T1_est is None:
            return None
        T = np.linalg.inv(T0_gt @ np.linalg.inv(T0_est)) @ T1_gt @ np.linalg.inv(T1_est)
        if xytheta:
            return transform_2_xypsi(T)
        else:
            return T

    def plot(self, zero_time=True, line_kwargs={}, marker_kwargs={}, figax=None):
        if self.m == 0:
            self.m = 1
        for i in range(len(self.fa)):
            if len(self.fa[i]) < self.m:
                self.fa[i] += [self.no_align]*(self.m - len(self.fa[i]))

        times = np.array(self.times)
        if zero_time:
            times -= times[0]
        # import ipdb; ipdb.set_trace()
        fa = np.array(self.fa).reshape((self.n, self.m, 3))
        gt = np.array(self.gt)
        est = np.array(self.est)

        return se2_align_plot(times, fa, gt, est, figax=figax, line_kwargs=line_kwargs, marker_kwargs=marker_kwargs)
