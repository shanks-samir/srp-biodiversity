"""Script to analyze and plot NDVI and Shannon diversity distributions per class on ISPRS Potsdam.

Specifically checks whether 'tree' and 'impervious' overlap due to leaf-off acquisition conditions.
Evaluates on the standard published hold-out validation tiles:
  7_8, 4_10, 2_11, 5_11
"""

import os
import io
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

# ISPRS Potsdam standard class definitions
CLASS_NAMES = {
    0: "Clutter",
    1: "Impervious surfaces",
    2: "Building",
    3: "Low vegetation",
    4: "Tree",
    5: "Car"
}

CLASS_COLORS = {
    (255, 255, 255): 1,  # Impervious
    (0, 0, 255): 2,      # Building
    (0, 255, 255): 3,    # Low veg
    (0, 255, 0): 4,      # Tree
    (255, 255, 0): 5,    # Car
    (255, 0, 0): 0       # Clutter
}

VALIDATION_TILES = ["2_11", "4_10", "5_11", "7_8"]


def load_tile_from_zip(rgbir_zip_path: str, label_zip_path: str, tile_id: str):
    """Load 4-band RGBIR and 3-band Label mask directly from zip archives without full decompression."""
    rgbir_name = f"4_Ortho_RGBIR/top_potsdam_{tile_id}_RGBIR.tif"
    label_name = f"top_potsdam_{tile_id}_label.tif"

    with zipfile.ZipFile(rgbir_zip_path, 'r') as z_img:
        # Some archives don't have the leading folder prefix
        matching_img = [n for n in z_img.namelist() if f"top_potsdam_{tile_id}_RGBIR.tif" in n]
        if not matching_img:
            raise FileNotFoundError(f"Could not find RGBIR for tile {tile_id} in {rgbir_zip_path}")
        img_bytes = z_img.read(matching_img[0])

    with zipfile.ZipFile(label_zip_path, 'r') as z_lbl:
        matching_lbl = [n for n in z_lbl.namelist() if f"top_potsdam_{tile_id}_label.tif" in n]
        if not matching_lbl:
            raise FileNotFoundError(f"Could not find label for tile {tile_id} in {label_zip_path}")
        lbl_bytes = z_lbl.read(matching_lbl[0])

    # Read TIFF bytes
    if HAS_TIFFFILE:
        rgbir = tifffile.imread(io.BytesIO(img_bytes))  # (H, W, 4) or (4, H, W)
        label_rgb = tifffile.imread(io.BytesIO(lbl_bytes))
    else:
        # Fallback to PIL
        with Image.open(io.BytesIO(img_bytes)) as pil_img:
            rgbir = np.array(pil_img)
        with Image.open(io.BytesIO(lbl_bytes)) as pil_lbl:
            label_rgb = np.array(pil_lbl.convert("RGB"))

    if rgbir.ndim == 3 and rgbir.shape[0] == 4:
        rgbir = rgbir.transpose(1, 2, 0)

    # Channels: 0=Red, 1=Green, 2=Blue, 3=NIR
    red = rgbir[:, :, 0].astype(np.float32)
    nir = rgbir[:, :, 3].astype(np.float32)

    # Decode RGB label mask into class indices
    h, w, _ = label_rgb.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for color, cid in CLASS_COLORS.items():
        match = (label_rgb[:, :, 0] == color[0]) & \
                (label_rgb[:, :, 1] == color[1]) & \
                (label_rgb[:, :, 2] == color[2])
        class_mask[match] = cid

    return red, nir, class_mask


def compute_fast_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    return np.clip((nir - red) / (nir + red + eps), -1.0, 1.0)


def compute_fast_shannon(ndvi: np.ndarray, window_size: int = 9, num_bins: int = 16) -> np.ndarray:
    """Fast approximation of local Shannon entropy using uniform filtering on histogram bins."""
    from scipy.ndimage import uniform_filter
    ndvi_norm = np.clip((ndvi + 1.0) / 2.0, 0.0, 1.0 - 1e-5)
    bins = (ndvi_norm * num_bins).astype(int)

    shannon = np.zeros_like(ndvi, dtype=np.float32)
    for b in range(num_bins):
        indicator = (bins == b).astype(np.float32)
        prob = uniform_filter(indicator, size=window_size)
        prob = np.clip(prob, 1e-7, 1.0)
        shannon -= prob * np.log2(prob)

    max_entropy = np.log2(num_bins)
    return np.clip(shannon / max_entropy, 0.0, 1.0)


def main():
    potsdam_dir = os.path.join(os.path.dirname(__file__), "..", "Potsdam")
    rgbir_zip = os.path.join(potsdam_dir, "4_Ortho_RGBIR.zip")
    label_zip = os.path.join(potsdam_dir, "5_Labels_all.zip")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Potsdam Leaf-Off Condition & Ecology Channel Diagnostic ===")
    print(f"Sampling hold-out validation tiles: {VALIDATION_TILES}...")

    ndvi_by_class = {c: [] for c in CLASS_NAMES.keys()}
    shannon_by_class = {c: [] for c in CLASS_NAMES.keys()}

    for tile_id in VALIDATION_TILES:
        print(f"  --> Processing tile {tile_id}...")
        try:
            red, nir, mask = load_tile_from_zip(rgbir_zip, label_zip, tile_id)
            ndvi = compute_fast_ndvi(nir, red)
            shannon = compute_fast_shannon(ndvi)

            # Subsample 100k pixels per tile to keep memory and speed fast
            sample_rate = 20
            sub_ndvi = ndvi[::sample_rate, ::sample_rate].flatten()
            sub_shn = shannon[::sample_rate, ::sample_rate].flatten()
            sub_mask = mask[::sample_rate, ::sample_rate].flatten()

            for c in CLASS_NAMES.keys():
                c_idx = (sub_mask == c)
                if c_idx.any():
                    ndvi_by_class[c].extend(sub_ndvi[c_idx])
                    shannon_by_class[c].extend(sub_shn[c_idx])
        except Exception as e:
            print(f"      Warning: could not process tile {tile_id}: {e}")

    # Summary Statistics Table
    print("\n" + "=" * 70)
    print(f"{'Class Name':<22} | {'NDVI Mean (Std)':<18} | {'Shannon Mean (Std)':<18}")
    print("-" * 70)
    for c, name in CLASS_NAMES.items():
        vals_ndvi = np.array(ndvi_by_class[c])
        vals_shn = np.array(shannon_by_class[c])
        if len(vals_ndvi) > 0:
            ndvi_str = f"{vals_ndvi.mean():.3f} (±{vals_ndvi.std():.3f})"
            shn_str = f"{vals_shn.mean():.3f} (±{vals_shn.std():.3f})"
            print(f"{name:<22} | {ndvi_str:<18} | {shn_str:<18}")
    print("=" * 70)

    # Plot Histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_classes = [1, 3, 4]  # Impervious, Low Veg, Tree
    colors = {1: "#7f7f7f", 3: "#2ca02c", 4: "#8c564b"}

    # Subplot 1: NDVI
    for c in plot_classes:
        vals = np.array(ndvi_by_class[c])
        if len(vals) > 0:
            axes[0].hist(vals, bins=60, range=(-0.2, 0.8), density=True, alpha=0.5,
                         label=CLASS_NAMES[c], color=colors[c])
    axes[0].set_title("NDVI Distribution (Leaf-Off Vulnerability Check)")
    axes[0].set_xlabel("NDVI Value")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Subplot 2: Shannon Diversity
    for c in plot_classes:
        vals = np.array(shannon_by_class[c])
        if len(vals) > 0:
            axes[1].hist(vals, bins=60, range=(0.0, 1.0), density=True, alpha=0.5,
                         label=CLASS_NAMES[c], color=colors[c])
    axes[1].set_title("Local Shannon Diversity Distribution (Structural Heterogeneity)")
    axes[1].set_xlabel("Normalized Shannon Entropy H")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "potsdam_ecology_distribution.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved distribution plot to: {plot_path}")
    print("Check this plot to verify if Tree and Impervious overlap on NDVI!")


if __name__ == "__main__":
    main()
