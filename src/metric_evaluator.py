import numpy as np
import motmetrics as mm

class MetricEvaluator():

    def __init__(self, max_d=.5, noise_rot=np.eye(2), noise_tran=np.zeros((2,1))):
        self.max_d = max_d
        self.R_noise = noise_rot
        self.t_noise = noise_tran
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
    
    def update(self, gt_dict, hyp_dict):
        # put ground truth into camera frame
        for gt_id in gt_dict:
            gt_pos = np.array(gt_dict[gt_id]).reshape((-1,1))
            gt_pos_og_shape = gt_dict[gt_id].shape
            gt_dict[gt_id] = (self.R_noise @ gt_pos + self.t_noise).reshape(gt_pos_og_shape)
            
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