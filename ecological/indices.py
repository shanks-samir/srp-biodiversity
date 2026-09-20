import torch
import torch.nn.functional as F
import numpy as np


def compute_ndvi_numpy(nir: np.ndarray, red: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Compute Normalized Difference Vegetation Index (NDVI) using NumPy.
    
    Args:
        nir: Near-Infrared channel as float32 in range [0, 1] or [0, 255]
        red: Red channel as float32 in range [0, 1] or [0, 255]
        eps: Small constant to avoid division by zero
        
    Returns:
        ndvi: NDVI map in range [-1, 1] as float32
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    numerator = nir - red
    denominator = nir + red + eps
    ndvi = numerator / denominator
    return np.clip(ndvi, -1.0, 1.0)


def compute_ndvi_torch(nir: torch.Tensor, red: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute Normalized Difference Vegetation Index (NDVI) using PyTorch tensor.
    
    Args:
        nir: Tensor of shape (B, 1, H, W) or (B, H, W)
        red: Tensor of shape (B, 1, H, W) or (B, H, W)
        eps: Small constant to prevent zero division
        
    Returns:
        ndvi: Tensor in range [-1.0, 1.0] of same spatial dimensions
    """
    numerator = nir - red
    denominator = nir + red + eps
    ndvi = numerator / denominator
    return torch.clamp(ndvi, -1.0, 1.0)


def compute_shannon_entropy_torch(
    raster: torch.Tensor, 
    kernel_size: int = 7, 
    num_bins: int = 16, 
    eps: float = 1e-7
) -> torch.Tensor:
    """Compute local Shannon Diversity Index H over a sliding window for habitat heterogeneity.
    
    H(p) = - sum(p_i * log2(p_i)) over normalized binned values.
    
    Args:
        raster: Single-channel tensor (B, 1, H, W), normalized in [0, 1] (e.g. (NDVI + 1)/2)
        kernel_size: Sliding window size w (must be odd, default 7)
        num_bins: Number of histogram bins to measure local entropy
        eps: Smoothing epsilon
        
    Returns:
        entropy_map: (B, 1, H, W) local Shannon diversity index map
    """
    B, C, H, W = raster.shape
    assert C == 1, "Raster must be single-channel"
    pad = kernel_size // 2
    
    # Quantize raster into bin indices [0, num_bins - 1]
    raster_clamped = torch.clamp(raster, 0.0, 1.0 - 1e-5)
    bin_indices = (raster_clamped * num_bins).long()  # (B, 1, H, W)
    
    # One-hot encode the bins: (B, num_bins, H, W)
    one_hot = F.one_hot(bin_indices.squeeze(1), num_classes=num_bins).permute(0, 3, 1, 2).float()
    
    # Uniform smoothing filter across window to compute local bin probabilities
    weight = torch.ones((num_bins, 1, kernel_size, kernel_size), device=raster.device) / (kernel_size * kernel_size)
    
    # Depthwise conv2d to compute local frequencies
    local_probs = F.conv2d(one_hot, weight, padding=pad, groups=num_bins)  # (B, num_bins, H, W)
    local_probs = torch.clamp(local_probs, min=eps)
    
    # Compute Shannon Entropy: H = - sum(p * log2(p))
    entropy = -torch.sum(local_probs * torch.log2(local_probs), dim=1, keepdim=True)  # (B, 1, H, W)
    
    # Normalize by max possible entropy log2(num_bins) so output is in [0, 1]
    max_entropy = np.log2(num_bins)
    entropy_normalized = entropy / max_entropy
    return torch.clamp(entropy_normalized, 0.0, 1.0)
