import numpy as np
from copy import deepcopy
from motlee.tcaff.node import Node

SPEEDUP_TRICK = True
DIFF_TOL = np.array([2., 2., .7])


class Tree():
    
    def __init__(self, xhat0, P0, A, H, Q, window_len, prob_no_match=.1, 
                 max_branching_factor=4, max_tree_leaves=np.inf, wrap_theta_idx_2=True):
        self.hyp_root = Node(xhat0, P0)
        self.hyp_root.add_model(A, H, Q)
        self.hyp_leaves = [self.hyp_root]
        self.optimal = self.hyp_root
        
        self.window_len = window_len
        self.tree_depth = 1
        self.max_branching_factor = max_branching_factor
        self.prob_no_match_nl = -np.log(prob_no_match)
        self.max_tree_leaves = max_tree_leaves
        self.wrap_theta_idx_2 = wrap_theta_idx_2
        
    def update(self, zs, Rs):
        # extend
        self.extend(zs, Rs)
        
        # find optimal
        min_obj = np.inf
        min_node = None
        for node in self.hyp_leaves:
            node_obj = node.cumulative_objective()
            # assert node.objective_sum == node.cumulative_objective()
            if node_obj < min_obj:
                min_obj = node_obj
                min_node = node
        
        self.optimal = min_node
        
        # prune
        self.prune()
                  
            
        return self.optimal.xhat
        
        
    def extend(self, zs, Rs):
        """extends tree of hypotheses by finding measurements within a Mahalanobis distance
        tolerance away from current hypotheses

        Args:
            zs (list of numpy.arrays, shape(p,1)): measurements
            Rs (list of numpy.arrays, shape(p,p)): measurement covariances
        """
        hyp_leaves_old = self.hyp_leaves
        self.hyp_leaves = []

        # find potential new hypotheses
        for node in hyp_leaves_old:
            nlmls = []
            if SPEEDUP_TRICK:
                zs = deepcopy(zs)
                Rs = deepcopy(Rs)
                delete = []
                for i, (z, R) in enumerate(zip(zs, Rs)):
                    Hxhat = node.H @ node.xhat
                    if self.wrap_theta_idx_2:
                        while Hxhat.item(2) - z.item(2) > np.pi:
                            z[2] += 2*np.pi
                        while Hxhat.item(2) - z.item(2) < -np.pi:
                            z[2] -= 2*np.pi
                    diff = np.abs(Hxhat - z)
                    if diff.item(0) > DIFF_TOL[0] or diff.item(1) > DIFF_TOL[0] or diff.item(2) > DIFF_TOL[0]:
                        delete.append(i)
                    else:
                        nlmls.append(node.nlml(z, R))
                        
                # for i in range(len(delete) - 1, -1, -1):
                zs = np.delete(zs, delete, axis=0)
                Rs = np.delete(Rs, delete, axis=0)
                        
                        
            else:
                for z, R in zip(zs, Rs):
                    if self.wrap_theta_idx_2:
                        while Hxhat.item(2) - z.item(2) > np.pi:
                            z[2] += 2*np.pi
                        while Hxhat.item(2) - z.item(2) < -np.pi:
                            z[2] -= 2*np.pi
                    nlmls.append(node.nlml(z, R))
            
            z_sorted = [z for _, z in sorted(zip(nlmls, zs), key=lambda zipped: zipped[0])]
            R_sorted = [R for _, R in sorted(zip(nlmls, Rs), key=lambda zipped: zipped[0])]
            nlml_sorted = sorted(nlmls)
            # if there are objective values that are less probable than P(no match)
            # if np.where(np.array(nlml_sorted) > self.prob_no_match_nl)[0].shape[0] > 0:
            #     # then take the minimum between the max branching factor and the number of nodes lower than the threshold
            #     num_children = min(self.max_branching_factor, np.where(np.array(nlml_sorted) > self.prob_no_match_nl)[0][0])
            # else: 
            num_children = self.max_branching_factor
            
            self.hyp_leaves += \
                node.create_children(z_sorted[:num_children], 
                                     R_sorted[:num_children],
                                     nlml_sorted[:num_children])
            self.hyp_leaves.append(node.create_measurementless_child(objective=self.prob_no_match_nl))
        
        self.tree_depth += 1
        
    def prune(self):
        # move window
        if self.tree_depth > self.window_len:
            node = self.optimal
            while node not in self.hyp_root.children:
                node = node.parent
            new_root = node
        
            leaves_to_prune = []
            for node in self.hyp_leaves:
                parent_node = node.parent
                while parent_node != new_root and parent_node != self.hyp_root:
                    parent_node = parent_node.parent
                if parent_node == new_root:
                    continue
                else:
                    assert parent_node == self.hyp_root
                    leaves_to_prune.append(node)
            
            for node in leaves_to_prune:
                self.hyp_leaves.remove(node)
                
            old_root = self.hyp_root
            self.hyp_root = new_root
            self.hyp_root.parent = None
            for node in self.tree_as_list():
                node.objective_sum -= old_root.objective
                if old_root.z is not None:
                    node.measurements_in_hyp -= 1

            self.tree_depth -= 1
                
        # prune to max tree leaves
        objectives = [node.cumulative_objective() for node in self.hyp_leaves]
        sorted_leaves = [leaf for _, leaf in sorted(zip(objectives, self.hyp_leaves))]
        leaves_to_prune = sorted_leaves[self.max_tree_leaves:]
        for node in leaves_to_prune:
            self.hyp_leaves.remove(node)
        
    def tree_as_list(self):
        tree_as_list = []
        unvisited = [self.hyp_root]
        
        # repeat until unvisited is empty
        while unvisited:
            for child in unvisited[0].children:
                unvisited.append(child)
            tree_as_list.append(unvisited[0])
            unvisited.pop(0)
                
        return tree_as_list
    
    def get_optimal_ancestral_line(self):
        line = [self.optimal]
        node = self.optimal
        while node != self.hyp_root:
            node = node.parent
            line.insert(0, node)
        line.insert(0, self.hyp_root)
        return line
    
    def plotting_lines(self, t_offset):
        xhats = []
        ts = []
        nodes = [(self.hyp_root, 0)] # each tuple is a node and depth
        while nodes:
            curr_node, curr_depth = nodes.pop(0)
            for child in curr_node.children:
                ts.append([t_offset + curr_depth, t_offset + curr_depth + 1])
                xhats.append([curr_node.xhat.reshape(-1), child.xhat.reshape(-1)])
                nodes.append([child, curr_depth + 1])
        return ts, xhats
    
    def __str__(self):
        return_str = ''
        layer = [self.hyp_root]
        next_layer = []

        while layer or next_layer:
            if not layer:
                layer = next_layer
                next_layer = []
                return_str += '\n'
            if not layer:
                break
            return_str += f'{layer[0]} '
            next_layer += layer[0].children
            layer.pop(0)
            
        return return_str