import numpy as np
from copy import deepcopy

# MAIN_TREE_OBJ_REQ = 6.5 # doesn't work for 4 8 currently
# MAIN_TREE_OBJ_REQ = 5.18
# MAIN_TREE_OBJ_REQ = 5.18
STOP_EXPLORING_TREES_AFTER_MAIN_TREE = False

class TCAFF():
    
    def __init__(
        self,
        z2x,
        K,
        create_exploring_tree,
        create_main_tree,
        rho=3.,
        steps_before_main_tree_deletion=20,
        main_tree_obj_req=5.0
    ):        
        self.z2x = z2x
        self.k = 0 # depth
        self.K = K # window_len
        self.rho = rho
        self.z_bank = []
        self.R_bank = []
        self.exploring_trees = []
        self.main_tree = None
        self.create_exploring_tree = create_exploring_tree
        self.create_main_tree = create_main_tree
        self.steps_with_no_meas = 0
        self.steps_before_main_tree_deletion = steps_before_main_tree_deletion
        self.main_tree_obj_req = main_tree_obj_req
        
    def update(self, zs, Rs):
        # if len(zs) == 0 and len(self.exploring_trees) == 0:
        #     self.reset_main_tree()
        #     self.k = 0
        #     return        
        
        zs_cp = deepcopy(zs)
        Rs_cp = deepcopy(Rs)
        to_delete = set()
        # TODO: Change this
        # Deletes measurements that seem to be duplicates
        # for i in range(len(zs_cp)):
        #     for j in range(i+1, len(zs_cp)):
        #         if i != j and (zs[i] - zs[j]).T @ inv(Rs[i] + Rs[j]) @ (zs[i] - zs[j]) < .1:
        #             to_delete.add(j)
        try:
            for i in sorted(to_delete, reverse=True):
                del zs_cp[i]
                del Rs_cp[i]
        except:
            import ipdb; ipdb.set_trace()
            
        # print(f"before: {len(zs)} after: {len(zs)}")
            
        self.z_bank.append(deepcopy(zs_cp))
        self.R_bank.append(deepcopy(Rs_cp))
        
        if self.k == 0 or self.k >= self.K:
            self.exploring_trees = []
            for z in self.z_bank[0]:
                self.exploring_trees.append(self.create_exploring_tree(self.z2x(z)))
            if self.k >= self.K:
                self.z_bank = self.z_bank[1:]
                self.R_bank = self.R_bank[1:]
                        
        self.k += 1
        if self.k <= self.K:
            for tree in self.exploring_trees:
                tree.update(zs, Rs)
        else:
            # print(f"{self.k}: {self.steps_with_no_meas}")

            if self.steps_with_no_meas > self.steps_before_main_tree_deletion:
                self.reset_main_tree()
                self.steps_with_no_meas = 0
            if self.main_tree is None:
                for tree in self.exploring_trees:
                    for zs_from_bank, Rs_from_bank in zip(self.z_bank, self.R_bank):
                        tree.update(zs_from_bank, Rs_from_bank)
            if self.main_tree_condition():
                self.steps_with_no_meas = 0
                # if self.main_tree is None:
                self.setup_main_tree()
                # elif not self.shares_measurements(self.main_tree, self.get_optimal_tree()):
                        # print('uh oh')
                        # print(self.main_tree.optimal.cumulative_objective())
                        # print([node.z for node in self.get_optimal_tree().get_optimal_ancestral_line()])
                        # print(self.get_optimal_tree().optimal.cumulative_objective())
                    # pass
                    # self.setup_main_tree()
                    # self.main_tree = None
                    # self.k = 0
                    # self.z_bank = []
                    # self.R_bank = []
                                
            if self.main_tree is not None:
                self.main_tree.update(zs, Rs)
                # if np.all([node.z is None for node in self.main_tree.get_optimal_ancestral_line()]):
                #     self.main_tree = None
                        
    def reset_main_tree(self):
        if STOP_EXPLORING_TREES_AFTER_MAIN_TREE:
            return
        self.main_tree = None
        
    def main_tree_condition(self):
        if self.main_tree is not None and STOP_EXPLORING_TREES_AFTER_MAIN_TREE:
            return False
        if self.main_tree is not None:
            if np.all([node.z is None for node in self.main_tree.get_optimal_ancestral_line()]):
                self.steps_with_no_meas += 1
            else:
                self.steps_with_no_meas = 0
            # if obj_opt < MAIN_TREE_OBJ_REQ and self.steps_with_no_meas > self.steps_before_main_tree_deletion:
            #     self.steps_with_no_meas = 0
            #     return True
            return False
        leaf_opt = self.get_optimal_leaf()
        if leaf_opt is None:
            return False
        obj_opt = leaf_opt.cumulative_objective()
            # return False
            # return obj_opt < -2 and self.main_tree.optimal.cumulative_objective() > 5
        # tree_opt = self.get_optimal_tree()
        # obj_all = [opt.cumulative_objective() for opt in [tree.optimal for tree in self.exploring_trees]]
        # obj_all.remove(obj_opt)
        # if len(obj_all) == 0:
        #     num_meas = [node.z for node in tree_opt.get_optimal_ancestral_line() if node.z is not None]
        #     num_non_meas = [node.z for node in tree_opt.get_optimal_ancestral_line() if node.z is None]
        #     return len(num_meas) > len(num_non_meas)
            
        return obj_opt < self.main_tree_obj_req
        # return self.rho * obj_opt < np.min(obj_all) and obj_opt < MAIN_TREE_OBJ_REQ
        
        obj_all = [opt.cumulative_objective() for opt in [tree.optimal for tree in self.exploring_trees]]

        for obj, tree in zip(obj_all, self.exploring_trees):
            if tree == tree_opt:
                continue
            if not self.rho * obj_opt < obj:
                if not self.shares_measurements(tree_opt, tree):
                    return False
        # all close to optimal share measurements
        return True
            
    def shares_measurements(self, tree1, tree2):
        """this doesn't consider time of measures"""
        line1 = tree1.get_optimal_ancestral_line()
        line2 = tree2.get_optimal_ancestral_line()
        for ni in line1:
            for nj in line2:
                if ni.z is not None and nj.z is not None and np.allclose(ni.z, nj.z):
                # if ni.z is not None and nj.z is not None and (ni.z - nj.z).T @ inv(ni.R + nj.R) @ (ni.z - nj.z) < MERGE_MEAS_MD_TOL:
                    return True
        return False
        
    def setup_main_tree(self):
        xhat0 = self.get_optimal_tree().hyp_root.xhat
        self.main_tree = self.create_main_tree(xhat0)
        for zs, Rs in zip(self.z_bank[:-1], self.R_bank[:-1]):
            self.main_tree.update(zs, Rs)
        self.steps_with_no_meas = 0
        
        
    def get_optimal_leaf(self):
        if len(self.exploring_trees) == 0:
            return None
        opt_all = [tree.optimal for tree in self.exploring_trees]
        obj_all = [opt.cumulative_objective() for opt in opt_all]
        return opt_all[np.argmin(obj_all)]
            
    def get_optimal_tree(self):
        if len(self.exploring_trees) == 0:
            return None
        opt_all = [tree.optimal for tree in self.exploring_trees]
        obj_all = [opt.cumulative_objective() for opt in opt_all]
        return self.exploring_trees[np.argmin(obj_all)]
    
    def exploring_trees_plotting_lines(self, t_offset=0):
        ts = []
        xhats = []
        for tree in self.exploring_trees:
            new_ts, new_xhats = tree.plotting_lines(t_offset)
            ts += new_ts
            xhats += new_xhats
        return ts, xhats