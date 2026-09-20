import os
from glob import glob
from typing import Optional, List, Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from ecological.indices import compute_ndvi_numpy, compute_shannon_entropy_torch

# Try importing rasterio for GIS GeoTIFF support, fallback to PIL / tifffile
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class EUMultispectralUAVDataset(Dataset):
    """Flexible Dataset Loader for EU Multispectral UAV Imagery across Portugal, Germany, UK, Austria.
    
    Supports:
      1. Multi-band GeoTIFF orthomosaics (e.g. 5-band MicaSense: Blue, Green, Red, RedEdge, NIR)
      2. 4-band or 3-band UAV tiles
      3. Automatic sliding-window patch extraction for large orthomosaics
    """

    def __init__(
        self,
        data_dir: str,
        country: Optional[str] = None,
        patch_size: int = 512,
        stride: int = 384,
        red_band_idx: int = 2,   # 0-indexed (typical MicaSense: B=0, G=1, R=2, RE=3, NIR=4)
        nir_band_idx: int = 4,   # 0-indexed
        compute_shannon: bool = True,
    ):
        """
        Args:
            data_dir: Path to country or all EU UAV site data
            country: Filter by country folder ('portugal', 'germany', 'uk', 'austria') if structured
            patch_size: Size of patch to crop for SAM model input
            stride: Stride for sliding window tiling of large orthomosaics
            red_band_idx: Index of Red band in multi-band raster
            nir_band_idx: Index of NIR band in multi-band raster
            compute_shannon: Whether to calculate Shannon diversity index
        """
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.stride = stride
        self.red_band_idx = red_band_idx
        self.nir_band_idx = nir_band_idx
        self.compute_shannon = compute_shannon

        # Discover image paths
        target_dir = os.path.join(data_dir, country) if country and os.path.isdir(os.path.join(data_dir, country)) else data_dir
        
        # Look for multi-band GeoTIFF files or RGB tiles
        self.raster_files = sorted(
            glob(os.path.join(target_dir, "**/*.tif"), recursive=True) +
            glob(os.path.join(target_dir, "**/*.tiff"), recursive=True)
        )
        # Filter out mask/label files from image list
        self.image_files = [f for f in self.raster_files if not any(k in os.path.basename(f).lower() for k in ["mask", "label", "gt", "annot"])]
        
        # Create patch index list [(file_idx, y_offset, x_offset)]
        self.patches: List[Tuple[int, int, int]] = []
        self._index_patches()

    def _index_patches(self):
        """Index spatial patches across all UAV raster scenes."""
        for file_idx, fpath in enumerate(self.image_files):
            try:
                if HAS_RASTERIO:
                    with rasterio.open(fpath) as src:
                        h, w = src.height, src.width
                else:
                    with Image.open(fpath) as img:
                        w, h = img.size

                if h <= self.patch_size and w <= self.patch_size:
                    self.patches.append((file_idx, 0, 0))
                else:
                    for y in range(0, max(1, h - self.patch_size + 1), self.stride):
                        for x in range(0, max(1, w - self.patch_size + 1), self.stride):
                            self.patches.append((file_idx, y, x))
            except Exception as e:
                # If cannot open file, fallback to single entry
                self.patches.append((file_idx, 0, 0))

    def __len__(self):
        return len(self.patches)

    def _read_raster_window(self, fpath: str, y: int, x: int, size: int) -> np.ndarray:
        """Read multi-band patch from raster."""
        if HAS_RASTERIO:
            with rasterio.open(fpath) as src:
                window = rasterio.windows.Window(x, y, min(size, src.width - x), min(size, src.height - y))
                data = src.read(window=window)  # (C, H, W)
                # Pad to (C, size, size) if near boundaries
                c, h, w = data.shape
                if h < size or w < size:
                    padded = np.zeros((c, size, size), dtype=data.dtype)
                    padded[:, :h, :w] = data
                    data = padded
                return data
        else:
            img = Image.open(fpath)
            # Crop window
            cropped = img.crop((x, y, x + size, y + size))
            arr = np.array(cropped)
            if arr.ndim == 2:
                return arr[np.newaxis, ...]
            elif arr.ndim == 3:
                return arr.transpose(2, 0, 1)
            return arr

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, y_off, x_off = self.patches[idx]
        fpath = self.image_files[file_idx]

        raw_patch = self._read_raster_window(fpath, y_off, x_off, self.patch_size)  # (C, H, W)
        num_channels = raw_patch.shape[0]

        # Normalization to [0, 1] float
        raw_float = raw_patch.astype(np.float32)
        if raw_float.max() > 1.0:
            raw_float = raw_float / (65535.0 if raw_float.max() > 255.0 else 255.0)

        # Extract RGB bands for SAM
        if num_channels >= 3:
            # Assumed order: R=red_band_idx, G=1, B=0 (or first 3 channels)
            r_idx = min(self.red_band_idx, num_channels - 1)
            g_idx = 1 if num_channels > 1 else 0
            b_idx = 0
            rgb = np.stack([raw_float[r_idx], raw_float[g_idx], raw_float[b_idx]], axis=0)
        else:
            # Grayscale expanded to 3 channels
            rgb = np.repeat(raw_float[:1], 3, axis=0)

        # Extract or compute NDVI if NIR band exists
        if num_channels > self.nir_band_idx and num_channels > self.red_band_idx:
            nir = raw_float[self.nir_band_idx]
            red = raw_float[self.red_band_idx]
            ndvi = compute_ndvi_numpy(nir, red)
        else:
            # Default zero NDVI if NIR is not present on this UAV site
            ndvi = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)

        rgb_tensor = torch.from_numpy(rgb).float()
        ndvi_tensor = torch.from_numpy(ndvi).unsqueeze(0).float()

        # Compute Shannon Diversity
        if self.compute_shannon and ndvi.max() > ndvi.min():
            ndvi_norm = (ndvi_tensor + 1.0) / 2.0
            shannon_tensor = compute_shannon_entropy_torch(ndvi_norm.unsqueeze(0), kernel_size=7).squeeze(0)
        else:
            shannon_tensor = torch.zeros_like(ndvi_tensor)

        # Look for corresponding label mask if exists
        lbl_fpath = fpath.replace(".tif", "_label.tif").replace(".tiff", "_label.tiff")
        if os.path.exists(lbl_fpath):
            mask_data = self._read_raster_window(lbl_fpath, y_off, x_off, self.patch_size)
            mask_tensor = torch.from_numpy(mask_data[0]).long()
        else:
            mask_tensor = torch.zeros((self.patch_size, self.patch_size), dtype=torch.long)

        return {
            "image": rgb_tensor,                # (3, H, W)
            "ndvi": ndvi_tensor,                # (1, H, W)
            "shannon": shannon_tensor,          # (1, H, W)
            "mask": mask_tensor,                # (H, W)
            "meta": {"file": fpath, "y": y_off, "x": x_off}
        }
