import numpy as np

from tcaff.utils.transform import transform

def is_viewable(pos, T_WC):
    cx = 425.5404052734375
    cy = 400.3540954589844
    fx = 285.72650146484375
    fy = 285.77301025390625
    K = np.array([[fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]])
    D = np.array([-0.006605582777410746, 0.04240882024168968, -0.04068116843700409, 0.007674722000956535])
    width = 800
    height = 848
    
    return pt_is_seeable(K, T_WC, width, height, pos)

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

def pixel2groundplane(K, T_WC, pixel_coord):
    p_c = np.concatenate([pixel_coord[0:2].reshape(-1), [1]]).reshape((3, 1))
    p_w = transform(T_WC, np.linalg.inv(K) @ p_c)
    alpha = -p_w.item(2) / (T_WC[2, 3] - p_w.item(2))
    return p_w + (T_WC[:3, 3].reshape((3, 1)) - p_w) * alpha

def get_3d_coordinates(K, d, x, y):
    """Written by ChatGPT and fixed by me

    Args:
        K ((3,3) np.array): camera intrinsic matrix
        d (float): depth (distance along z-axis)
        x (float): pixel x coordinate
        y (float): pixel y coordinate

    Returns:
        (3,) np.array: x, y, z 3D point coordinates in camera RDF coordinates
    """
    # Step 1: Convert pixel coordinates to normalized device coordinates (NDC)
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    
    NDC_x = (x - cx) / fx
    NDC_y = (y - cy) / fy
    
    # Step 2: Get the 3D world coordinates (W)
    W_x = NDC_x * d
    W_y = NDC_y * d
    W_z = d
    return np.array([W_x, W_y, W_z])


def depth_mask_2_centroid(depth, mask, center_pixel, K, depth_scale=1e-3):
    """Processes a depth image and binary mask to output a 3D centroid measurement of the segmented object

    Args:
        depth ((h,w) uint16 np.array):      depth image array
        mask ((h,w) binary np.array):   segment mask
        center_pixel (2-element int tuple): x, y from top left corner
        K ((3,3) np.array):                 camera intrinsic matrix
        depth_scale (float):                amount to multiply depth image by

    Returns:
        (3,) np.array: centroid 3d position
    """    
    depth_masked = depth.copy().astype(np.float64)
    depth_masked[mask==0] = np.nan
    depth_masked[depth_masked < 1.] = np.nan

    depth_med = np.nanmedian(depth_masked.reshape(-1))
    
    centroid = get_3d_coordinates(K, depth_med*depth_scale, center_pixel[0], center_pixel[1])
    
    return centroid

def mask_depth_2_width_height(depth, mask, K):
    # find bbox corners
    nonzero_indices = np.transpose(np.nonzero(mask))
    x0 = np.min(nonzero_indices[:,1])
    x1 = np.max(nonzero_indices[:,1])
    y0 = np.min(nonzero_indices[:,0])
    y1 = np.max(nonzero_indices[:,0])
    
    # get 3D points of 2D bounding box using median depth
    pt0 = get_3d_coordinates(K, depth, x0, y0)
    pt1 = get_3d_coordinates(K, depth, x1, y1)

    # calculate and return width and height
    width = pt1.item(0) - pt0.item(0)
    height = pt1.item(1) - pt0.item(1)
    return width, height
    