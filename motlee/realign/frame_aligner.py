import numpy as np
from numpy.linalg import inv
from enum import Enum, auto

import open3d as o3d
if o3d.__DEVICE_API__ == 'cuda':
    import open3d.cuda.pybind.t.pipelines.registration as treg
else:
    import open3d.cpu.pybind.t.pipelines.registration as treg
import clipperpy

from motlee.utils.transform import T2d_2_T3d
from motlee.realign.wls import wls, wls_residual

class AssocMethod(Enum):
    ICP = auto()
    ICP_STRONG_CORRES = auto()
    CLIPPER = auto()
    CLIPPER_MULT_SOL = auto()
    CLIPPER_SPARSE = auto()
        
class FrameAlignSolution():
    """
    Simple struct-like class for holding information returned by a frame alignment attempt.
    """
    
    def __init__(
        self,
        success,
        transform=None,
        num_objs_associated=None,
        transform_residual=None,
        objective_score=None,
        associated_objs=None
    ):
        self.success = success
        self.transform = transform
        self.num_objs_associated = num_objs_associated
        self.transform_residual = transform_residual
        self.objective_score = objective_score
        self.associated_objs = associated_objs
        
    def __str__(self):
        if not self.success:
            return "Success: false\nFrame align failed\n"
        ret = f"Success: {self.success}\n"
        if self.transform is not None:
            ret += f"Transform: \n{np.round(self.transform, 3)}\n"
        if self.num_objs_associated is not None:
            ret += f"Num objects used: {self.num_objs_associated}\n"
        if self.transform_residual is not None:
            ret += f"Transform residual: {self.transform_residual}\n"
        if self.objective_score is not None:
            ret += f"Objective score: {self.objective_score}\n"
        if self.associated_objs is not None:
            ret += f"Associated objects: {self.associated_objs.T}"
        return ret

class FrameAligner():
    """
    Class for aligning static object landmarks and dynamic object detections.
    """
    
    def __init__(
        self,
        method,
        num_objs_req=0,
        icp_max_dist=None,
        clipper_sigma=None,
        clipper_epsilon=None,
        clipper_mult_downweight=None, 
        clipper_mult_repeats=None,
        clipper_sparse_proportion=None,
        clipper_sparse_max=None
    ):
        """FrameAligner constructor

        Args:
            method (_type_): _description_
            num_objs_req (int, optional): _description_. Defaults to 0.
            icp_max_dist (_type_, optional): _description_. Defaults to None.
            clipper_sigma (_type_, optional): _description_. Defaults to None.
            clipper_epsilon (_type_, optional): _description_. Defaults to None.
            clipper_mult_downweight (_type_, optional): _description_. Defaults to None.
            clipper_mult_repeats (_type_, optional): _description_. Defaults to None.
            
        """
        if method == AssocMethod.ICP \
            or method == AssocMethod.ICP_STRONG_CORRES:
            assert icp_max_dist is not None, \
                "icp_max_dist arg is required for ICP method"
        elif method == AssocMethod.CLIPPER or \
            method == AssocMethod.CLIPPER_MULT_SOL or \
            method == AssocMethod.CLIPPER_SPARSE:
            assert clipper_epsilon is not None and clipper_epsilon is not None, \
                "clipper params need to be set"
            if method == AssocMethod.CLIPPER_MULT_SOL:
                assert clipper_mult_downweight is not None and clipper_mult_repeats is not None, \
                    "clipper params need to be set"
            elif method == AssocMethod.CLIPPER_SPARSE:
                assert clipper_sparse_proportion is not None and clipper_sparse_max is not None, \
                    "clipper params need to be set"
    
        self.method = method       
        self.num_objs_req = num_objs_req
        self.sigma=clipper_sigma
        self.epsilon=clipper_epsilon
        self.max_dist = icp_max_dist
        self.clipper_mult_downweight = clipper_mult_downweight
        self.clipper_mult_repeats = clipper_mult_repeats
        self.clipper_sparse_proportion = clipper_sparse_proportion
        self.clipper_sparse_max = clipper_sparse_max
        
    def align_objects(self, static_objects=None, static_ages=None, static_put_assoc=None,
                      dynamic_objects=None, dynamic_weights=None, T_init_guess=None):
        """performs alignment on objects by associating objects and then running Aruns with weights
        based on the age of the object.

        Args:
            static_objects (list(numpy.array, shape(n,2|3)), optional): list of first and second 
                numpy array representing object point locations
            static_ages (list(numpy.array, shape(n,)), optional): timesteps since objs1 were seen 
                if weighted algorithm is to be used. Defaults to None.
            static_put_assoc (numpy.array, shape(m,2), type(int), optional): array representing 
                indices of potential associations. Indices in i-th column  and j-th row 
                corresponds to a match involving the j-th object of the i-th list from 
                static_objects. Defaults to None.
            dynamic_objects (list(numpy.array, shape(n,2|3)), optional): list of first and second 
                numpy array representing object point locations
            dynamic_weights (list(numpy.array, shape(n,)), optional): weights for dynamic object 
                correspondences
            T_init_guess (numpy.array, shape(4,4), optional): Initial guess transform. Defaults to None.

        Returns:
            numpy.array, shape(4,4): 3D transform that aligns objs2 with objs2
            float: residual of alignment
            int: number of corresponding objects used for alignment
        """
        if static_objects is not None:
            inliers, static_corres, static_weights, static_assoc_scores = self.associate_static_objects(static_objects, static_ages, T_init_guess, static_put_assoc)
            solutions = []
                
            for i, (ins, sc, sw, ss) in enumerate(zip(inliers, static_corres, static_weights, static_assoc_scores)):
                if dynamic_objects is None:
                    solutions.append(self.compose_solution(sc[0], sc[1], sw))
                else:
                    static_weight_sum = np.sum(sw)
                    dynamic_weight_sum = np.sum(dynamic_weights)
                    # sw /= static_weight_sum / sw.shape[0]
                    # dynamic_weights /= dynamic_weight_sum / dynamic_weights.shape[0]
                    sw /= static_weight_sum
                    dynamic_weights /= dynamic_weight_sum
                    dynamic_weights *= .05
                    # print(f'number of dynamic: {dynamic_weights.shape}')
                    weights_all = np.concatenate([sw.reshape(-1), dynamic_weights.reshape(-1)])
                    # print(f"weights_all: {np.sum(weights_all)}")
                    # print(f"dynamic_weights: {np.sum(dynamic_weights)}")
                    # print(f"static_weights: {np.sum(sw)}")
                    objs1_all = np.vstack([sc[0], dynamic_objects[0]])
                    objs2_all = np.vstack([sc[1], dynamic_objects[1]])
                    solutions.append(self.compose_solution(objs1_all, objs2_all, weights_all))
                if solutions[-1].success:
                    solutions[-1].associated_objs = np.copy(ins)
                    solutions[-1].objective_score = ss
        elif dynamic_objects is not None:
            solutions = [self.compose_solution(dynamic_objects[0], dynamic_objects[1], dynamic_weights)]
        else:
            assert False, "no dynamic or static objects for alignment"
            
        if self.method == AssocMethod.CLIPPER_MULT_SOL:
            return solutions
        else:
            assert len(solutions) == 1, solutions
            return solutions[0]
        
    def associate_static_objects(self, static_objects, static_ages, T_init_guess, putative_assoc):
        objs1, objs2 = static_objects
        ages1, ages2 = static_ages if static_ages is not None else (np.ones(objs1.shape[0]), np.ones(objs2.shape[0]))
        if 0 in objs1.shape or 0 in objs2.shape:
            return None, [[np.array([]), np.array([])]], [np.array([])], [None if not self.method != AssocMethod.CLIPPER_MULT_SOL else 0.]
        if self.method == AssocMethod.ICP or \
            self.method == AssocMethod.ICP_STRONG_CORRES:
            assert T_init_guess is not None, "initial guess is required for ICP registration method"
            assert putative_assoc is None, "putative associations cannot be used for ICP"
            objs1_reordered, objs2_del, weights1 = self.icp_data_association(objs1, objs2, T_init_guess, ages1, ages2)
            objs2_reordered, objs1_del, weights2 = self.icp_data_association(objs2, objs1, inv(T_init_guess), ages2, ages1)
            if objs1_reordered is None or objs2_reordered is None or objs1_del is None or objs2_del is None:
                return FrameAlignSolution(success=False)
            objs1_corres = np.concatenate([objs1_reordered, objs1_del], axis=0)
            objs2_corres = np.concatenate([objs2_del, objs2_reordered], axis=0)
            weights_corres = np.concatenate([weights1, weights2], axis=0)

            to_delete = []
            for i in range(objs1_corres.shape[0]):
                no_other_pair = True
                for j in range(objs2_corres.shape[0]):
                    if i == j: continue
                    if np.allclose(objs1_corres[i,:], objs1_corres[j,:]) and np.allclose(objs2_corres[i,:], objs2_corres[j,:]):
                        no_other_pair = False
                        if j > i:
                            to_delete.append(j)
                if no_other_pair and self.method == AssocMethod.ICP_STRONG_CORRES:
                    to_delete.append(i)

            objs1_corres = np.delete(objs1_corres, to_delete, axis=0)
            objs2_corres = np.delete(objs2_corres, to_delete, axis=0)
            weights_corres = np.delete(weights_corres, to_delete, axis=0)
            Ain = None # TODO: find inliniers
            
        elif self.method == AssocMethod.CLIPPER or \
            self.method == AssocMethod.CLIPPER_MULT_SOL or \
            self.method == AssocMethod.CLIPPER_SPARSE:
            objs1 = np.array([o for o in objs1.tolist()])
            objs2 = np.array([o for o in objs2.tolist()])
            if self.method == AssocMethod.CLIPPER or \
                self.method == AssocMethod.CLIPPER_SPARSE:
                Ain = self.clipper_data_association(objs1, objs2, putative_assoc)
                objs1_corres, objs2_corres, ages1_corres, ages2_corres = \
                    self.clipper_inlier_associations(Ain, objs1, objs2, ages1, ages2)
                weights_corres = 1/(.01 + ages1_corres * ages2_corres)
            elif self.method == AssocMethod.CLIPPER_MULT_SOL:
                Ains, scores = self.clipper_mult_sols(objs1, objs2, putative_assoc)
                objs1_corres, objs2_corres, weights_corres = [], [], []
                for Ain in Ains:
                    o1c, o2c, a1c, a2c = \
                        self.clipper_inlier_associations(Ain, objs1, objs2, ages1, ages2)
                    objs1_corres.append(o1c)
                    objs2_corres.append(o2c)
                    weights_corres.append(1/(.01 + a1c * a2c))
                # opt_idx = np.argmax(scores)
                # Ain = Ains[opt_idx]
        
        if self.method == AssocMethod.CLIPPER_MULT_SOL:
            return Ains, [(o1c, o2c) for o1c, o2c in zip(objs1_corres, objs2_corres)], weights_corres, scores
        else:
            # TODO: get Ain for ICP
            return [Ain], [(objs1_corres, objs2_corres)], [weights_corres], [None]    
        
    def compose_solution(self, objs1_corres, objs2_corres, weights_corres):

        num_objs = len(objs1_corres)
        if num_objs < self.num_objs_req:
            return FrameAlignSolution(success=False)

        T_new = wls(objs1_corres, objs2_corres, weights_corres)
        if T_new.shape[0] == 3 and T_new.shape[1] == 3:
            T_new = T2d_2_T3d(T_new)
        residual = wls_residual(objs1_corres, objs2_corres, weights_corres, T_new)
        # return T_new, residual, num_objs, c1_out, c2_out
        return FrameAlignSolution(success=True, transform=T_new, 
                                  transform_residual=residual, num_objs_associated=num_objs)

    def detections2pointcloud(detections, org_by_tracks):
        dets_cp = []
        if org_by_tracks:
            pass
        else:
            for frame in detections:
                for detection in frame:
                    if detection is not None:
                        dets_cp.append(np.concatenate([detection, [[0]]], axis=0).reshape(-1))
            dets_np = np.array(dets_cp)
        return o3d.t.geometry.PointCloud(dets_np)

    def run_icp(self, detections1, detections2, initial_guess):
        trans_init = initial_guess
        # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
        estimation = treg.TransformationEstimationPointToPoint()
        # Search distance for Nearest Neighbour Search [Hybrid-Search is used].
        max_correspondence_distance = self.max_dist
        # Initial alignment or source to target transform.
        init_source_to_target = trans_init
        # Convergence-Criteria for Vanilla ICP
        criteria = treg.ICPConvergenceCriteria(relative_fitness=0.0000001,
                                            relative_rmse=0.0000001,
                                            max_iteration=30)
        # Down-sampling voxel-size. If voxel_size < 0, original scale is used.
        voxel_size = -1
        reg_point_to_point = treg.icp(detections2, detections1, max_correspondence_distance,
                                init_source_to_target, estimation, criteria,
                                voxel_size)
        return reg_point_to_point

    def clipper_data_association(self, detections1, detections2, putative_assoc=None):
        """
        Parameters
        ----------
        detections1 : (n,3) np.array
        detections2 : (m,3) np.array

        Return
        ------
        Ain : (p,2) np.array (int) - inlier set. First column contains indices from detections1, 
            second contains corresponding indices from detections2
        """
        iparams = clipperpy.invariants.EuclideanDistanceParams()
        iparams.sigma = self.sigma
        iparams.epsilon = self.epsilon
        invariant = clipperpy.invariants.EuclideanDistance(iparams)

        params = clipperpy.Params()
        clipper = clipperpy.CLIPPER(invariant, params)

        n = len(detections1)
        m = len(detections2)
        
        if putative_assoc is None:
            A = clipperpy.utils.create_all_to_all(n, m)
        else:
            A = putative_assoc.astype(np.int32)
        
        if self.method == AssocMethod.CLIPPER_SPARSE:
            num_assoc = min(int(self.clipper_sparse_proportion * A.shape[0]), self.clipper_sparse_max)
            A = A[np.random.choice(A.shape[0], num_assoc, replace=False)]

        clipper.score_pairwise_consistency(detections1.T, detections2.T, A)
        clipper.solve()
        Ain = clipper.get_selected_associations()
        
        return Ain

    def clipper_inlier_associations(self, Ain, pts1, pts2, weights1=None, weights2=None):
        """
        Returns two lists of ordered pts that have been associated with each other

        Args:
            Ain (np.array(p,2, int)): inlier set from CLIPPER
            pts1 (np.array(n,2|3): first set of points
            pts2 (np.array(m,2|3)): second set of points
            weights1 (np.array(n,)): weights
            weights2 (np.array(m,)): weights

        Returns:
            pts1_corres (np.array(p,2|3)): reordered points from pts1 corresponding to pts2_corres
            pts2_corres (np.array(p,2|3)): reordered points from pts2 corresponding to pts1_corres
        """
        assert (weights1 is None and weights2 is None) or \
            (weights1 is not None and weights2 is not None)
        if Ain.shape[0] == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        pts1_corres = np.zeros((Ain.shape[0], pts1.shape[1]))
        pts2_corres = np.zeros((Ain.shape[0], pts2.shape[1]))
        weights1_corres = np.zeros((Ain.shape[0]))
        weights2_corres = np.zeros((Ain.shape[0]))
        
        for i in range(Ain.shape[0]):
            pts1_corres[i,:] = pts1[Ain[i,0]]
            pts2_corres[i,:] = pts2[Ain[i,1]]
            if weights1 is not None and weights2 is not None:
                weights1_corres[i] = weights1[Ain[i,0]]
                weights2_corres[i] = weights2[Ain[i,1]]
        return pts1_corres, pts2_corres, weights1_corres, weights2_corres

    def icp_data_association(self, objs1, objs2, T_current, ages1=None, ages2=None):
        if ages1 is None or ages2 is None:
            ages1 = np.ones((len(objs1), 1))
            ages2 = np.ones((len(objs2), 1))
        if ages1.ndim == 1: ages1 = ages1.reshape((-1, 1))
        if ages2.ndim == 1: ages2 = ages2.reshape((-1, 1))
        objs1_ptcld = o3d.t.geometry.PointCloud(objs1 if objs1.shape[1] == 3 else np.hstack([objs1, np.zeros((objs1.shape[0], 1))]))
        objs2_ptcld = o3d.t.geometry.PointCloud(objs2 if objs2.shape[1] == 3 else np.hstack([objs2, np.zeros((objs2.shape[0], 1))]))
        correspondence_set2 = self.run_icp(objs1_ptcld, objs2_ptcld, T_current).correspondences_.numpy()
            
        objs1_reordered = np.zeros(objs2.shape)
        ages1_reordered = np.zeros((objs2.shape[0], 1))
        for i in range(objs2.shape[0]):
            if correspondence_set2[i] == -1: continue
            try:
                objs1_reordered[i, :] = objs1[correspondence_set2[i], :]
            except:
                import ipdb; ipdb.set_trace()
            ages1_reordered[i, 0] = ages1[correspondence_set2[i], 0]
        no_correspond_idx = [i for i,x in enumerate(correspondence_set2.reshape(-1).tolist()) if x==-1]
        objs2 = np.delete(objs2, no_correspond_idx, axis=0)
        objs1_reordered = np.delete(objs1_reordered, no_correspond_idx, axis=0)
        ages2 = np.delete(ages2, no_correspond_idx, axis=0)
        ages1_reordered = np.delete(ages1_reordered, no_correspond_idx, axis=0)
            
        # weights = 1/(1 + ages2 * ages1_reordered)
        weights = 1/(.01 + ages2 * ages1_reordered)
        
        # return objs1_new, objs2_new, weights
        return objs1_reordered, objs2, weights

    def realign_static_downweight(self, detections1, detections2, downweight_nodes=None, putative_assoc=None):
        """
        Parameters
        ----------
        detections1 : (n,3) np.array
        detections2 : (m,3) np.array
        downweight_associations : indices of (p_in,2) associations to downweight. First column
            corresponds to detections1 indices and second corresponds to detections2
        downweight  : float [0.0, 1.0] amount to multiply associations by

        Return
        ------
        Ain   : (p_out,2) np.array (int) - inlier set. First column contains indices from detections1, 
            second contains corresponding indices from detections2
        score : float, u'Mu / (u'u)
        """
        iparams = clipperpy.invariants.EuclideanDistanceParams()
        iparams.sigma = self.sigma
        iparams.epsilon = self.epsilon
        invariant = clipperpy.invariants.EuclideanDistance(iparams)

        params = clipperpy.Params()
        clipper = clipperpy.CLIPPER(invariant, params)

        n = len(detections1)
        m = len(detections2)
        if putative_assoc is None:
            A = clipperpy.utils.create_all_to_all(n, m)
        else:
            A = putative_assoc.astype(np.int32)

        clipper.score_pairwise_consistency(detections1.T, detections2.T, A)
        M_orig = clipper.get_affinity_matrix()
        if downweight_nodes is not None:
            M = M_orig.copy()
            C = clipper.get_constraint_matrix()
            # for idx, downweight in downweight_nodes.items():
            for idx, downweight in downweight_nodes:
                # M[idx,:] *= downweight
                # M[:,idx] *= downweight
                row_indices, col_indices = np.meshgrid(idx, idx, indexing='ij')
                if len(row_indices) != 0 and len(col_indices) != 0:
                    M[row_indices,col_indices] *= downweight

            clipper = clipperpy.CLIPPER(invariant, params)
            clipper.set_matrix_data(M=M, C=C)
                    
        clipper.solve()
        Ain = np.zeros((len(clipper.get_solution().nodes), 2)).astype(np.int64)
        for i in range(len(clipper.get_solution().nodes)):
            Ain[i,:] = A[clipper.get_solution().nodes[i],:]
        # Ain = clipper2.get_selected_associations()
        
        u_sol = clipper.get_solution().u.copy()
        for i in range(u_sol.shape[0]):
            u_sol[i] = u_sol[i] if i in clipper.get_solution().nodes else 0.0
        score = u_sol.T @ M_orig @ u_sol / (u_sol.T @ u_sol)
        if len(clipper.get_solution().nodes) == 0:
            score = 0
        
        return Ain, score, clipper.get_solution().nodes
        
    def clipper_mult_sols(self, pts1, pts2, putative_assoc=None):
        """repeatedly reweights and reruns CLIPPER to search for close-to-optimal solutions

        Args:
            pts1 (numpy.array, shape(n,2 or 3)): First set of points for association
            pts2 (numpy.array, shape(m,2 or 3)): Seconds set of points for association
            downweight (float [0.0, 1.0), optional): Amount to downweight CLIPPER on each iteration. Defaults to .9.
            num_repeats (int, optional): Number of iterations to rerun CLIPPER. Defaults to 10.

        Returns:
            num_repeats[(p_out,2) np.array (int)] : inlier set. First column contains indices from detections1, 
            second contains corresponding indices from detections2
            num_repeats list of scores
        """
        all_scores = []
        all_pairs = []
        
        for i in range(self.clipper_mult_repeats):
            if i == 0:
                pairs, score, nodes = self.realign_static_downweight(pts1, pts2, putative_assoc=putative_assoc)
                downweight_nodes = [(nodes, self.clipper_mult_downweight)]
            else:
                pairs, score, nodes = self.realign_static_downweight(pts1, pts2, putative_assoc=putative_assoc, downweight_nodes=downweight_nodes)
                downweight_nodes.append((nodes, self.clipper_mult_downweight))
                # for node in nodes:
                #     if node not in downweight_nodes:
                #         downweight_nodes[node] = 1.0
                #     downweight_nodes[node] *= self.clipper_mult_downweight
            all_scores.append(score)
            all_pairs.append(pairs)
            # objs0_corres, objs1_corres = get_inlier_associations(pts1, pts2, pairs)

        return all_pairs, all_scores