import numpy as np
import motmetrics as mm

class MetricEvaluator():

    def __init__(self, max_d=2*37.5):
        self.max_d = max_d
        self.acc = mm.MOTAccumulator(auto_id=True)
    
    def update(self, gt_dict, hyp_dict):
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