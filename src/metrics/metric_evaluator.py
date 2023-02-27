import numpy as np
import motmetrics as mm
from copy import deepcopy

from utils.transform import transform

class MetricEvaluator():

    def __init__(self, max_d):
        self.max_d = max_d
        self.acc = mm.MOTAccumulator(auto_id=True)
        
    @property
    def mota(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['mota'], name='acc')
        return summary.mota.iloc[0]
    
    @property
    def motp(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['motp'], name='acc')
        return summary.motp.iloc[0]
    
    @property
    def avg_fp_per_frame(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['num_false_positives', 'num_frames'], name='acc')
        return summary.num_false_positives.iloc[0] / summary.num_frames.iloc[1]
    
    @property
    def avg_fn_per_frame(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['num_misses', 'num_frames'], name='acc')
        return summary.num_misses.iloc[0] / summary.num_frames.iloc[1]
    
    @property
    def avg_switch_per_frame(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['num_switches', 'num_frames'], name='acc')
        return summary.num_switches.iloc[0] / summary.num_frames.iloc[1]
    
    def get_metric(self, metric_name):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=[metric_name], name='acc')
        return summary[metric_name].iloc[0]
    
    def update(self, gt_dict, hyp_dict, T_true, T_bel):
        # put ground truth into camera frame
        gt_dict = deepcopy(gt_dict)
        for gt_id in gt_dict:
            gt_dict[gt_id] = transform(T_bel @ np.linalg.inv(T_true), gt_dict[gt_id])
            
        gt_id_list, gt_pt_matrix = self._matrix_list_form(gt_dict)
        hyp_id_list, hyp_pt_matrix = self._matrix_list_form(hyp_dict)
        
        # remove camera name from hyp_id_list
        for i in range(len(hyp_id_list)):
            hyp_id_list[i] = hyp_id_list[i][1]
            
        distances = mm.distances.norm2squared_matrix(gt_pt_matrix, hyp_pt_matrix, max_d2=self.max_d**2)
        
        self.acc.update(gt_id_list, hyp_id_list, distances)
        
    def display_results(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['num_frames', 'mota', 'motp'], name='acc')
        print(summary)
       
    def _matrix_list_form(self, in_dict):
        out_list = []
        out_matrix = np.array([[]])
        for item_id, pt in in_dict.items():
            out_list.append(item_id)
            if out_matrix.shape == (1,0):
                out_matrix = pt.reshape(1,2)
            else:
                out_matrix = np.concatenate([out_matrix, pt.reshape(1,2)], axis=0)
        # if out_matrix == None:
        #     out_matrix = np.array([[]])
        return out_list, out_matrix
    
def get_avg_metric(metric, mes, divide_by_frames=False):
    num_cams=len(mes)
    m_avg = 0
    for me in mes:
        if divide_by_frames:
            m_val = me.get_metric(metric) / (me.get_metric('num_frames') * num_cams)
        else:
            m_val = me.get_metric(metric) / num_cams
        m_avg += m_val
    return m_avg

def print_metric_results(mes, inconsistencies, agents, mota_only=True):
    mota = get_avg_metric('mota', mes)
    motp = get_avg_metric('motp', mes)
    fp = get_avg_metric('num_false_positives', mes, divide_by_frames=True)
    fn = get_avg_metric('num_misses', mes, divide_by_frames=True)
    switch = get_avg_metric('num_switches', mes, divide_by_frames=True)
    precision = get_avg_metric('precision', mes)
    recall = get_avg_metric('recall', mes)
    total_num_tracks = sum([len(a.tracker_mapping) / len(agents) for a in agents]) / len(agents)
    incon_per_track = inconsistencies / total_num_tracks if total_num_tracks else 0.0

    print(f'mota: {mota}')
    if mota_only: return
    print(f'motp: {motp}')
    print(f'fp: {fp}')
    print(f'fn: {fn}')
    print(f'switch: {switch}')
    print(f'inconsistencies: {inconsistencies}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'num_tracks: {total_num_tracks}')
    print(f'incon_per_track: {incon_per_track}')