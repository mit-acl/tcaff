import numpy as np

def transform(T, vec):
    unshaped_vec = vec.reshape(-1)
    resized_vec = np.concatenate(
        [unshaped_vec, np.zeros((T.shape[0] - 1 - unshaped_vec.shape[0]))]).reshape(-1)
    resized_vec = np.concatenate(
        [resized_vec, np.ones((T.shape[0] - resized_vec.shape[0]))]).reshape((-1, 1))
    transformed = T @ resized_vec
    return transformed.reshape(-1)[:unshaped_vec.shape[0]].reshape(vec.shape) 

def pixel2groundplane(K, T_WC, pixel_coord):
    p_c = np.concatenate([pixel_coord[0:2].reshape(-1), [1]]).reshape((3, 1))
    p_w = transform(T_WC, np.linalg.inv(K) @ p_c)
    alpha = -p_w.item(2) / (T_WC[2, 3] - p_w.item(2))
    return p_w + (T_WC[:3, 3].reshape((3, 1)) - p_w) * alpha

def pt_is_seeable(K, T_WC, width, height, pt):
    #
    # draw out FOV lines
    #
    #              C
    #             / \
    #            p1  p0
    #
    p0 = np.array([0, height, 1]).reshape((3, 1))
    p1 = np.array([width, height, 1]).reshape((3, 1))
    p03d = pixel2groundplane(K, T_WC, p0)
    p13d = pixel2groundplane(K, T_WC, p1)
    t = T_WC[:3, 3]
    
    l0_slope = (p03d.item(1) - t.item(1)) / (p03d.item(0) - t.item(0))
    l0_intrcpt = p03d.item(1) - l0_slope*p03d.item(0)
    l1_slope = (p13d.item(1) - t.item(1)) / (p13d.item(0) - t.item(0))
    l1_intrcpt = p13d.item(1) - l1_slope*p13d.item(0)
    
    # if p03d is to the right of c, cone must be less than
    if p03d.item(0) > t.item(0):
        l0_check = lambda x : x.item(1) <= l0_slope * x.item(0) + l0_intrcpt
    else:
        l0_check = lambda x : x.item(1) >= l0_slope * x.item(0) + l0_intrcpt
        
    # if p13d is to the right of c, cone must be greater than
    if p13d.item(0) > t.item(0):
        l1_check = lambda x : x.item(1) >= l1_slope * x.item(0) + l1_intrcpt
    else:
        l1_check = lambda x : x.item(1) <= l1_slope * x.item(0) + l1_intrcpt
        
    return l0_check(pt) and l1_check(pt)
