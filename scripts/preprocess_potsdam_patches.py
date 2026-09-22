import os
import sys
import glob
import argparse
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ecological.indices import compute_ndvi_numpy, compute_shannon_entropy_torch

# Standard benchmark hold-out validation tiles cited in remote sensing literature:
# Rottensteiner et al. (2012), https://doi.org/10.5194/isprsannals-I-3-293-2012
POTSDAM_VAL_TILES = ["7_8", "4_10", "2_11", "5_11"]

# TorchGeo Potsdam2D classes:
# 0: Clutter/Boundary, 1: Impervious, 2: Building, 3: Low Veg, 4: Tree, 5: Car
def rgb_to_class_mask(rgb: np.ndarray) -> np.ndarray:
    """Vectorized conversion of Potsdam RGB label array (H, W, 3) to class IDs [0..5]."""
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    code = (r << 16) | (g << 8) | b
    mask = np.zeros(code.shape, dtype=np.uint8)  # 0: clutter (red or unlabeled)

    mask[code == ((255 << 16) | (255 << 8) | 255)] = 1  # Impervious (white)
    mask[code == 255] = 2                               # Building (blue)
    mask[code == ((255 << 8) | 255)] = 3                # Low vegetation (cyan)
    mask[code == (255 << 8)] = 4                        # Tree (green)
    mask[code == ((255 << 16) | (255 << 8))] = 5        # Car (yellow)
    return mask


def read_tiff(path: str) -> np.ndarray:
    """Robust TIFF reader using tifffile, rasterio, or PIL."""
    try:
        import tifffile
        return tifffile.imread(path)
    except Exception:
        pass
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read()  # (C, H, W)
            return np.transpose(data, (1, 2, 0))  # to (H, W, C)
    except Exception:
        pass
    with Image.open(path) as img:
        return np.array(img)


def main():
    parser = argparse.ArgumentParser(description="Pre-crop Potsdam GeoTIFFs into 512x512 Native Patches")
    parser.add_argument("--data_dir", type=str, default="./Potsdam", help="Path to Potsdam directory")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size (default: 512)")
    parser.add_argument("--stride", type=int, default=384, help="Sliding window stride (default: 384)")
    parser.add_argument("--min_foreground_ratio", type=float, default=0.10, help="Min ratio of non-clutter pixels")
    parser.add_argument("--output_dir", type=str, default="./Potsdam/patches_512", help="Destination folder")
    parser.add_argument("--max_tiles", type=int, default=None, help="Limit number of tiles for quick test")
    args = parser.parse_args()

    train_out = os.path.join(args.output_dir, "train")
    val_out = os.path.join(args.output_dir, "val")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    # Find RGBIR images
    img_candidates = sorted(
        glob.glob(os.path.join(args.data_dir, "**/*RGBIR*.tif"), recursive=True) +
        glob.glob(os.path.join(args.data_dir, "**/*RGBIR*.tiff"), recursive=True)
    )
    if not img_candidates:
        print(f"Error: No RGBIR files found in {args.data_dir}")
        sys.exit(1)

    # Match each RGBIR tile to its corresponding label file
    tile_pairs = []
    for img_path in img_candidates:
        fname = os.path.basename(img_path)
        # Extract tile id e.g. "2_10" from "top_potsdam_2_10_RGBIR.tif"
        parts = fname.replace("top_potsdam_", "").replace("_RGBIR.tif", "").replace("_RGBIR.tiff", "").split("_")
        if len(parts) >= 2:
            tile_id = f"{parts[0]}_{parts[1]}"
        else:
            continue

        # Look for label file
        lbl_candidates = [
            os.path.join(args.data_dir, f"top_potsdam_{tile_id}_label.tif"),
            os.path.join(args.data_dir, f"top_potsdam_{tile_id}_label_noBoundary.tif"),
            os.path.join(args.data_dir, "5_Labels_all", f"top_potsdam_{tile_id}_label.tif"),
            os.path.join(args.data_dir, "5_Labels_all_noBoundary", f"top_potsdam_{tile_id}_label_noBoundary.tif")
        ]
        lbl_path = None
        for cand in lbl_candidates:
            if os.path.exists(cand):
                lbl_path = cand
                break

        if lbl_path:
            is_val = tile_id in POTSDAM_VAL_TILES
            split = "val" if is_val else "train"
            tile_pairs.append((tile_id, img_path, lbl_path, split))

    if args.max_tiles:
        tile_pairs = tile_pairs[:args.max_tiles]

    print(f"Found {len(tile_pairs)} verified (RGBIR, Label) tile pairs.")
    print(f"Train tiles: {sum(1 for t in tile_pairs if t[3] == 'train')}, Val tiles: {sum(1 for t in tile_pairs if t[3] == 'val')}")

    patch_counts = {"train": 0, "val": 0}
    class_indices = {
        "train": {c: [] for c in range(6)},
        "val": {c: [] for c in range(6)}
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for tile_id, img_path, lbl_path, split in tqdm(tile_pairs, desc="Processing Potsdam Tiles"):
        out_split_dir = os.path.join(args.output_dir, split)

        # Load RGBIR image (4 bands: R=0, G=1, B=2, NIR=3)
        raw_img = read_tiff(img_path)
        if raw_img.ndim == 3 and raw_img.shape[-1] >= 4:
            r_uint8 = raw_img[..., 0]
            g_uint8 = raw_img[..., 1]
            b_uint8 = raw_img[..., 2]
            nir_uint8 = raw_img[..., 3]
        elif raw_img.ndim == 3 and raw_img.shape[0] >= 4:
            r_uint8 = raw_img[0]
            g_uint8 = raw_img[1]
            b_uint8 = raw_img[2]
            nir_uint8 = raw_img[3]
        else:
            print(f"Warning: Unexpected channel count {raw_img.shape} for {img_path}. Skipping.")
            continue

        rgb_uint8 = np.stack([r_uint8, g_uint8, b_uint8], axis=0)  # (3, H, W) uint8

        # Load Label
        raw_lbl = read_tiff(lbl_path)
        if raw_lbl.ndim == 3 and raw_lbl.shape[0] == 3 and raw_lbl.shape[-1] != 3:
            raw_lbl = np.transpose(raw_lbl, (1, 2, 0))
        mask = rgb_to_class_mask(raw_lbl)  # (H, W) uint8 with values 0..5

        H, W = mask.shape

        # Sliding window over the full 6000x6000 tile
        for y in range(0, H - args.patch_size + 1, args.stride):
            for x in range(0, W - args.patch_size + 1, args.stride):
                patch_mask = mask[y:y+args.patch_size, x:x+args.patch_size]
                fg_ratio = np.count_nonzero(patch_mask > 0) / float(args.patch_size * args.patch_size)

                # Skip uninformative patches that are purely clutter/empty border
                if fg_ratio < args.min_foreground_ratio:
                    continue

                # Ensure contiguous memory copy so PyTorch does NOT serialize the entire 6000x6000 parent storage
                patch_rgb = np.ascontiguousarray(rgb_uint8[:, y:y+args.patch_size, x:x+args.patch_size])
                patch_r = patch_rgb[0].astype(np.float32) / 255.0
                patch_nir = nir_uint8[y:y+args.patch_size, x:x+args.patch_size].astype(np.float32) / 255.0

                # Compute NDVI at native resolution
                patch_ndvi = compute_ndvi_numpy(patch_nir, patch_r)  # (H, W) float32

                # Compute Shannon Entropy H at native resolution
                ndvi_norm = torch.from_numpy((patch_ndvi + 1.0) / 2.0).unsqueeze(0).unsqueeze(0).float()
                shannon_tensor = compute_shannon_entropy_torch(ndvi_norm, kernel_size=7).squeeze(0)  # (1, H, W)

                # Format patch dictionary with compact memory-isolated tensors (~2MB per patch)
                patch_filename = f"patch_{tile_id}_{y:05d}_{x:05d}.pt"
                patch_fpath = os.path.join(out_split_dir, patch_filename)

                patch_dict = {
                    "image": torch.from_numpy(patch_rgb).clone(),                           # (3, 512, 512) uint8
                    "mask": torch.from_numpy(np.ascontiguousarray(patch_mask)).clone(),     # (512, 512) uint8
                    "ndvi": torch.from_numpy(patch_ndvi).unsqueeze(0).clone().half(),       # (1, 512, 512) float16
                    "shannon": shannon_tensor.clone().half(),                              # (1, 512, 512) float16
                    "tile_id": tile_id,
                    "coord": (y, x)
                }
                torch.save(patch_dict, patch_fpath)
                patch_counts[split] += 1

                # Record present classes for instant class-index lookups
                unique_classes = np.unique(patch_mask)
                for c in unique_classes:
                    if int(c) in class_indices[split]:
                        # Require at least 50 foreground pixels of class c
                        if (patch_mask == c).sum() >= 50:
                            class_indices[split][int(c)].append(patch_filename)

    # Save class indices to JSON for instant O(1) episodic sampling
    for split in ["train", "val"]:
        idx_path = os.path.join(args.output_dir, f"{split}_class_index.json")
        with open(idx_path, "w") as f:
            json.dump(class_indices[split], f, indent=2)
        print(f"Saved {split} class index to {idx_path}")
        print(f"  Total {split} patches: {patch_counts[split]}")
        for c, files in class_indices[split].items():
            print(f"    Class {c}: {len(files)} patches")

    print(f"\n[SUCCESS] Preprocessing completed! Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
