import numpy as np
import torch
from scipy.ndimage import label, center_of_mass


def generate_prompts_from_mask(mask: np.ndarray, num_points: int = 3):
    """Extract representative positive point prompts and bounding box from a binary ground-truth/support mask.
    
    Args:
        mask: 2D binary numpy array (H, W) where 1 indicates foreground target class
        num_points: Maximum number of positive point prompts to extract
        
    Returns:
        points: (N, 2) array of coordinates (x, y)
        labels: (N,) array of point labels (1 for positive prompt)
        box: (4,) array [x_min, y_min, x_max, y_max] or None if mask is empty
    """
    if mask.sum() == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32), None
    
    # Bounding box
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = float(np.min(x_indices)), float(np.max(x_indices))
    y_min, y_max = float(np.min(y_indices)), float(np.max(y_indices))
    box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
    
    # Point prompts via connected components centers
    labeled_mask, num_features = label(mask)
    points = []
    
    for feat_id in range(1, min(num_features + 1, num_points + 1)):
        cy, cx = center_of_mass(labeled_mask == feat_id)
        # Verify centroid is inside the mask; if not, pick the closest foreground pixel
        if mask[int(cy), int(cx)] > 0:
            points.append([cx, cy])
        else:
            feat_ys, feat_xs = np.where(labeled_mask == feat_id)
            idx = len(feat_ys) // 2
            points.append([feat_xs[idx], feat_ys[idx]])
            
    # If fewer components than num_points, randomly sample foreground points
    while len(points) < num_points and len(x_indices) > 0:
        rand_idx = np.random.randint(0, len(x_indices))
        points.append([x_indices[rand_idx], y_indices[rand_idx]])
        
    points = np.array(points[:num_points], dtype=np.float32)
    labels = np.ones(len(points), dtype=np.int32)
    
    return points, labels, box


def generate_ndvi_guided_prompts(
    ndvi: np.ndarray, 
    threshold: float = 0.35, 
    num_positive: int = 3, 
    num_negative: int = 2
):
    """Automatically generate positive and negative point prompts using NDVI thresholding.
    
    Args:
        ndvi: 2D numpy array (H, W) in range [-1, 1]
        threshold: Minimum NDVI value to consider as vigorous vegetation
        num_positive: Number of positive points to sample from high NDVI regions
        num_negative: Number of negative points to sample from low NDVI regions
        
    Returns:
        points: (N, 2) array of coordinates (x, y)
        labels: (N,) array of labels (1 = positive foreground, 0 = negative background)
    """
    pos_mask = (ndvi >= threshold).astype(np.uint8)
    neg_mask = (ndvi < 0.1).astype(np.uint8)
    
    pos_points, pos_labels, _ = generate_prompts_from_mask(pos_mask, num_points=num_positive)
    neg_points, _, _ = generate_prompts_from_mask(neg_mask, num_points=num_negative)
    neg_labels = np.zeros(len(neg_points), dtype=np.int32)
    
    if len(pos_points) > 0 and len(neg_points) > 0:
        points = np.vstack([pos_points, neg_points])
        labels = np.concatenate([pos_labels, neg_labels])
    elif len(pos_points) > 0:
        points, labels = pos_points, pos_labels
    elif len(neg_points) > 0:
        points, labels = neg_points, neg_labels
    else:
        points = np.zeros((0, 2), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int32)
        
    return points, labels
