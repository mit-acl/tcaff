import numpy as np

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
    