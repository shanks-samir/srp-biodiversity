import os
import glob
import json
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset


class PotsdamPatchDataset(Dataset):
    """High-speed Dataset loader for pre-cropped 512x512 native Potsdam patches.
    
    Loads precomputed .pt files containing:
      - 'image': (3, 512, 512) float in [0, 1]
      - 'mask': (512, 512) long with class IDs [0..5]
      - 'ndvi': (1, 512, 512) float in [-1, 1]
      - 'shannon': (1, 512, 512) float in [0, 1]
    """

    def __init__(self, patch_dir: str, class_index_file: Optional[str] = None):
        """
        Args:
            patch_dir: Directory containing .pt patch files (e.g. ./Potsdam/patches_512/train)
            class_index_file: Path to pre-generated JSON class index mapping {class_id: [filenames]}
        """
        self.patch_dir = patch_dir
        self.patch_files = sorted(glob.glob(os.path.join(patch_dir, "*.pt")))
        self.filename_to_idx = {os.path.basename(f): i for i, f in enumerate(self.patch_files)}

        # Load precomputed class index if available
        self.class_to_indices: Dict[int, List[int]] = {}
        if class_index_file and os.path.exists(class_index_file):
            with open(class_index_file, "r") as f:
                raw_index = json.load(f)
            for c_str, fnames in raw_index.items():
                c = int(c_str)
                self.class_to_indices[c] = [
                    self.filename_to_idx[fn] for fn in fnames if fn in self.filename_to_idx
                ]
            print(f"Loaded instant class index from {class_index_file} ({len(self.patch_files)} patches total).")
        else:
            # Check if index file exists in parent folder
            parent_idx = os.path.join(os.path.dirname(patch_dir), f"{os.path.basename(patch_dir)}_class_index.json")
            if os.path.exists(parent_idx):
                with open(parent_idx, "r") as f:
                    raw_index = json.load(f)
                for c_str, fnames in raw_index.items():
                    c = int(c_str)
                    self.class_to_indices[c] = [
                        self.filename_to_idx[fn] for fn in fnames if fn in self.filename_to_idx
                    ]
                print(f"Loaded instant class index from {parent_idx} ({len(self.patch_files)} patches total).")

    def __len__(self) -> int:
        return len(self.patch_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fpath = self.patch_files[idx]
        data = torch.load(fpath, weights_only=True)

        # Handle uint8 image normalization to [0, 1] float32
        img = data["image"]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        elif img.max() > 1.0:
            img = img.float() / 255.0
        else:
            img = img.float()

        mask = data["mask"].long()
        ndvi = data["ndvi"].float()
        shannon = data["shannon"].float()

        return {
            "image": img,          # (3, 512, 512) float32 in [0, 1]
            "mask": mask,          # (512, 512) long with class IDs [0..5]
            "ndvi": ndvi,          # (1, 512, 512) float32 in [-1, 1]
            "shannon": shannon,    # (1, 512, 512) float32 in [0, 1]
        }
