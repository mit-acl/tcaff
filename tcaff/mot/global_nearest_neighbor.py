import numpy as np
from scipy.optimize import linear_sum_assignment

def global_nearest_neighbor(xbar_list, P_list, H_list, z_list, R_list, tau, alpha, use_nlml=True):
    '''
    xbar_list: list of state predictions (nx1 numpy arrays)
    P_list: list of state estimation covariances (nxn numpy arrays)
    H_list: list of measurement matrices (pxn numpy arrays)
    z_list: list of z_list (px1 numpy arrays)
    R_list: list of meaurement covariances (pxp numpy arrays)
    tau: Mahanobis distance tolerance for a measurement being associated with an object
    alpha: association parameter
    returns: list of state indices and measurement indicies that should be associated together
    returns: list of unassociated measurement indices
    '''
    assert len(xbar_list) == len(P_list) and len(xbar_list) == len(H_list)
    assert len(z_list) == len(R_list)
    num_states = len(xbar_list)
    num_meas = len(z_list)

    geometry_scores = np.zeros((num_states, num_meas))
    large_num = 1000
    for i in range(num_states):
        Hx = (H_list[i] @ xbar_list[i])
        for j, (z, R) in enumerate(zip(z_list, R_list)):
            V = P_list[i][:2,:2] + R
            # Mahalanobis distance
            d = np.sqrt((z - Hx).T @ np.linalg.inv(V) @ (z - Hx)).item(0)
            
            # Geometry similarity value
            if not d < tau:
                s_d = large_num
            elif use_nlml:
                s_d = 1/alpha*(2*np.log(2*np.pi) + d**2 + np.log(np.linalg.det(V)))
            else:
                s_d = 1/alpha*d
            if np.isnan(s_d):
                s_d = large_num
            geometry_scores[i,j] = s_d

    # augment cost to add option for no associations
    hungarian_cost = np.concatenate([
        np.concatenate([geometry_scores, np.ones(geometry_scores.shape)], axis=1),
        np.ones((geometry_scores.shape[0], 2*geometry_scores.shape[1]))], axis=0)
    row_ind, col_ind = linear_sum_assignment(hungarian_cost)

    state_meas_pairs = []
    unassociated = []
    for x_idx, z_idx in zip(row_ind, col_ind):
        # state and measurement associated together
        if x_idx < num_states and z_idx < len(z_list):
            assert geometry_scores[x_idx,z_idx] < 1
            state_meas_pairs.append((x_idx, z_idx))
        # unassociated measurement
        elif z_idx < len(z_list):
            unassociated.append(z_idx)
        # unassociated state or augmented part of matrix
        else:
            continue
        
    # if there are no tracks, hungarian matrix will be empty, handle separately
    if num_states == 0:
        for z_idx in range(len(z_list)):
            unassociated.append(z_idx)

    return state_meas_pairs, unassociated