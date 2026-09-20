from typing import Optional, List, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from torchgeo.datasets import Potsdam2D
    HAS_TORCHGEO = True
except ImportError:
    HAS_TORCHGEO = False

from ecological.indices import compute_ndvi_torch, compute_shannon_entropy_torch


# Official TorchGeo Potsdam2D Class Definitions
TORCHGEO_POTSDAM_CLASSES = {
    0: "clutter",
    1: "impervious_surfaces", # Base
    2: "building",            # Base
    3: "low_vegetation",      # Novel 1
    4: "tree",                # Novel 2
    5: "car",                 # Base
}


# Standard benchmark hold-out validation tiles cited in remote sensing literature:
# Rottensteiner et al. (2012), https://doi.org/10.5194/isprsannals-I-3-293-2012
POTSDAM_VAL_TILES = ["7_8", "4_10", "2_11", "5_11"]


class TorchGeoPotsdamDataset(Dataset):
    """Wrapper around torchgeo.datasets.Potsdam2D providing:
      - 3-channel RGB for SAM encoder
      - NDVI channel computed from Red (Band 0) and NIR (Band 3)
      - Shannon diversity index map
      - Standardized ground truth label masks
      - Standard literature validation split: hold-out tiles 7_8, 4_10, 2_11, 5_11
      
    Reference:
      Rottensteiner et al., "The ISPRS benchmark on urban object classification and 3D building reconstruction."
      ISPRS Annals of Photogrammetry, Remote Sensing and Spatial Information Sciences I-3 (2012): 293-298.
      https://doi.org/10.5194/isprsannals-I-3-293-2012
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        crop_size: int = 512,
        compute_shannon: bool = True,
        novel_classes: Optional[List[int]] = None,
        use_literature_split: bool = True,
        checksum: bool = False,
    ):
        """
        Args:
            root: Directory containing '4_Ortho_RGBIR.zip' and '5_Labels_all.zip' (or extracted folders)
            split: 'train', 'val', or 'test'
            crop_size: Spatial tile size for SAM input (default: 512)
            compute_shannon: Whether to calculate local Shannon entropy
            novel_classes: Novel class IDs for few-shot evaluation ([3, 4] for low_veg & tree)
            use_literature_split: If True, uses hold-out tiles [7_8, 4_10, 2_11, 5_11] for val, and remaining for train
            checksum: Verify file integrity if zips are used
        """
        if not HAS_TORCHGEO:
            raise ImportError(
                "torchgeo is not installed. Please run: pip install torchgeo"
            )

        self.crop_size = crop_size
        self.compute_shannon = compute_shannon
        self.novel_classes = novel_classes or [3, 4]  # 3: low_veg, 4: tree in torchgeo

        # When using literature split, load the full training pool and filter
        tg_split = "train" if (use_literature_split and split in ["train", "val"]) else split

        # Initialize official torchgeo Potsdam2D dataset
        self.dataset = Potsdam2D(
            root=root,
            split=tg_split,
            checksum=checksum
        )

        # Apply literature split filtering if requested
        if use_literature_split and split in ["train", "val"]:
            filtered_files = []
            for f in self.dataset.files:
                img_path = f["image"] if isinstance(f, dict) else str(f)
                is_val_tile = any(f"_{t}_" in img_path for t in POTSDAM_VAL_TILES)
                if split == "val" and is_val_tile:
                    filtered_files.append(f)
                elif split == "train" and not is_val_tile:
                    filtered_files.append(f)
            self.dataset.files = filtered_files
            print(f"Potsdam Literature Split [{split.upper()}]: {len(self.dataset.files)} tiles (Hold-out: {POTSDAM_VAL_TILES})")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        # TorchGeo returns:
        # image: (4, H, W) where 0=R, 1=G, 2=B, 3=IR, normalized in [0, 255] or float
        # mask:  (H, W) or (1, H, W) with class indices [0..5]
        img_raw = sample["image"].float()
        mask_raw = sample["mask"].long()

        if mask_raw.ndim == 3:
            mask_raw = mask_raw.squeeze(0)

        # Normalize image to [0, 1]
        if img_raw.max() > 1.0:
            img_raw = img_raw / 255.0

        # Channels: R=0, G=1, B=2, NIR=3
        red = img_raw[0:1, :, :]
        green = img_raw[1:2, :, :]
        blue = img_raw[2:3, :, :]
        nir = img_raw[3:4, :, :]

        # RGB for SAM
        rgb = torch.cat([red, green, blue], dim=0)  # (3, H, W)

        # Compute NDVI = (NIR - Red) / (NIR + Red)
        ndvi = compute_ndvi_torch(nir, red)  # (1, H, W) in [-1, 1]

        # Resize to crop_size if specified
        if self.crop_size is not None and (rgb.shape[-1] != self.crop_size or rgb.shape[-2] != self.crop_size):
            rgb = F.interpolate(rgb.unsqueeze(0), size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False).squeeze(0)
            mask = F.interpolate(mask_raw.unsqueeze(0).unsqueeze(0).float(), size=(self.crop_size, self.crop_size), mode="nearest").squeeze(0).squeeze(0).long()
            ndvi = F.interpolate(ndvi.unsqueeze(0), size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False).squeeze(0)
        else:
            mask = mask_raw

        # Compute local Shannon diversity map
        if self.compute_shannon:
            ndvi_norm = (ndvi + 1.0) / 2.0
            shannon = compute_shannon_entropy_torch(ndvi_norm.unsqueeze(0), kernel_size=7).squeeze(0)
        else:
            shannon = torch.zeros_like(ndvi)

        return {
            "image": rgb,          # (3, H, W)
            "mask": mask,          # (H, W) class indices [0..5]
            "ndvi": ndvi,          # (1, H, W)
            "shannon": shannon,    # (1, H, W)
        }
