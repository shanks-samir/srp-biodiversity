"""FLAIR (French Land cover from Aerospace ImageRy, IGN) dataset loader.

Verified against the FLAIR #1 specification:
  - IMG patches: 5 channels x 512 x 512, channel order **Blue, Green, Red, NIR, Elevation**.
    Channels 0-3 come from BD ORTHO HR and are uint8. Channel 4 is nDSM (DSM - DTM),
    i.e. above-ground height in metres, and is NOT uint8 -- treat it as float.
  - MSK patches: 1 channel x 512 x 512, uint8, class IDs in [1, 19]. There is no 0 class.
  - Ground sampling distance: 0.20 m.
  - Licence: Open Licence 2.0 / etalab-2.0 (attribution required).

The channel order is the single easiest thing to get wrong here: it is BGR, not RGB, and
NIR sits at index 3 rather than index 4 as on a 5-band MicaSense sensor.

Directory layout is discovered by recursive glob rather than assumed, because FLAIR #1,
FLAIR #2 and FLAIR-HUB nest domains and regions of interest differently. Any layout works
as long as image files are named IMG_*.tif and each has a sibling MSK_*.tif reachable by
substituting IMG->MSK in the filename and directory.
"""

import os
import re
import json
import glob
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ecological.indices import compute_ndvi_numpy, compute_shannon_entropy_torch

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


# Channel indices within a FLAIR IMG patch.
BLUE, GREEN, RED, NIR, ELEV = 0, 1, 2, 3, 4

GSD_METRES = 0.20

# The full 19-class nomenclature. IDs are 1-based as stored in the MSK rasters.
FLAIR_CLASSES: Dict[int, str] = {
    1: "building",
    2: "pervious_surface",
    3: "impervious_surface",
    4: "bare_soil",
    5: "water",
    6: "coniferous",
    7: "deciduous",
    8: "brushwood",
    9: "vineyard",
    10: "herbaceous_vegetation",
    11: "agricultural_land",
    12: "plowed_land",
    13: "swimming_pool",
    14: "snow",
    15: "clear_cut",
    16: "mixed",
    17: "ligneous",
    18: "greenhouse",
    19: "other",
}

# The published baselines train on classes 1-12 plus "other"; 13-19 are rare and are
# typically grouped into "other" or ignored. Use BASELINE_13 unless you specifically
# want the long tail.
BASELINE_13 = list(range(1, 13)) + [19]

# Classes whose separation is genuinely ecological rather than cartographic. These are
# the ones where NDVI and canopy height should carry information.
VEGETATION_CLASSES = [6, 7, 8, 9, 10, 11, 12, 15, 17]

# Rotating novel-class folds over the 13 baseline classes, following the PASCAL-5^i
# convention of holding out a disjoint subset per fold. Using folds rather than a single
# fixed novel pair is what makes a few-shot number defensible.
FOLDS: Dict[int, List[int]] = {
    0: [1, 5, 9, 12],
    1: [2, 6, 10, 19],
    2: [3, 7, 11],
    3: [4, 8, 12],
}


def _read_tif(path: str) -> np.ndarray:
    """Read a GeoTIFF as (C, H, W). Falls back across rasterio -> tifffile."""
    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            return src.read()
    if HAS_TIFFFILE:
        arr = tifffile.imread(path)
        if arr.ndim == 2:
            return arr[np.newaxis, ...]
        # tifffile may hand back (H, W, C); move channels first if that is the case.
        if arr.shape[-1] <= 8 and arr.shape[0] > 8:
            return np.transpose(arr, (2, 0, 1))
        return arr
    raise ImportError("Reading FLAIR patches needs either rasterio or tifffile installed.")


def mask_path_for(img_path: str) -> Optional[str]:
    """Derive the MSK path for an IMG path by substituting in both dir and filename."""
    directory, filename = os.path.split(img_path)
    msk_name = filename.replace("IMG", "MSK", 1)
    candidates = [
        os.path.join(directory, msk_name),
        os.path.join(directory.replace("/img", "/msk"), msk_name),
        os.path.join(directory.replace("/IMG", "/MSK"), msk_name),
        os.path.join(directory.replace("/aerial", "/labels"), msk_name),
        os.path.join(directory.replace("/AERIAL", "/LABELS"), msk_name),
        os.path.join(re.sub(r'(^|/|\\)aerial($|/|\\)', r'\1labels\2', directory), msk_name),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def domain_of(path: str) -> str:
    """Extract the FLAIR spatial-temporal domain (e.g. 'D004_2021') from a patch path.

    Domains are the cross-domain generalisation axis: they vary by department AND by
    acquisition date, so holding out whole domains tests both geographic and seasonal
    transfer. Falls back to the grandparent directory name when the pattern is absent.
    """
    match = re.search(r"(D\d{3}_\d{4})", path)
    if match:
        return match.group(1)
    parts = os.path.normpath(path).split(os.sep)
    return parts[-4] if len(parts) >= 4 else "unknown"


class FLAIRDataset(Dataset):
    """FLAIR patches with derived ecological channels and a selectable 3-channel composite.

    The `composite` argument is the central experimental knob. A frozen RGB foundation
    model has exactly three input slots, and which three signals occupy them is the
    question this project exists to answer. Every option below is the same underlying
    patch rendered differently -- nothing else in the pipeline changes.

      'rgb'        true colour. The control.
      'cir'        colour-infrared: NIR, Red, Green. The standard remote-sensing false
                   colour; vegetation renders bright red. Violates the pretraining colour
                   statistics, which is exactly what makes it worth measuring.
      'ndvi_dsm_h' fully derived: NDVI, normalised nDSM, local Shannon entropy. Discards
                   raw radiometry entirely in favour of ecological quantities.
      'rgb_ndvi'   Red, Green, NDVI -- keeps two true bands, swaps blue for vegetation index.
      'rgb_dsm'    Red, Green, normalised height. Tests structure alone against colour.

    All composites are returned as float32 in [0, 1] so they can be fed to any model
    expecting ordinary image input without further rescaling.
    """

    COMPOSITES = ("rgb", "cir", "ndvi_dsm_h", "rgb_ndvi", "rgb_dsm")

    def __init__(
        self,
        root: str,
        composite: str = "rgb",
        shannon_source: str = "ndvi",
        shannon_window: int = 7,
        shannon_bins: int = 16,
        elevation_clip_m: float = 30.0,
        domains: Optional[Sequence[str]] = None,
        class_index_file: Optional[str] = None,
        min_class_pixels: int = 50,
    ):
        """
        Args:
            root: directory containing FLAIR patches (searched recursively for IMG_*.tif)
            composite: which three channels to expose as `image`; see COMPOSITES
            shannon_source: 'ndvi' or 'elevation'. Computing local entropy over the nDSM
                measures actual structural roughness rather than inferring it from a
                chlorophyll signal, and unlike NDVI it survives leaf-off acquisition.
            shannon_window: sliding window size in pixels (7 px = 1.4 m at 0.2 m GSD)
            shannon_bins: histogram bins for the entropy calculation
            elevation_clip_m: nDSM values are clipped to [0, this] before normalising.
                30 m covers mature canopy and most buildings; higher values are noise.
            domains: restrict to these FLAIR domains (e.g. ['D004_2021']). None = all.
            class_index_file: optional precomputed {class_id: [patch paths]} JSON
            min_class_pixels: a patch counts as containing a class at this many pixels
        """
        if composite not in self.COMPOSITES:
            raise ValueError(f"composite must be one of {self.COMPOSITES}, got {composite!r}")
        if shannon_source not in ("ndvi", "elevation"):
            raise ValueError(f"shannon_source must be 'ndvi' or 'elevation', got {shannon_source!r}")

        self.root = root
        self.composite = composite
        self.shannon_source = shannon_source
        self.shannon_window = shannon_window
        self.shannon_bins = shannon_bins
        self.elevation_clip_m = elevation_clip_m
        self.min_class_pixels = min_class_pixels

        all_imgs = sorted(glob.glob(os.path.join(root, "**", "IMG_*.tif"), recursive=True))
        if not all_imgs:
            raise FileNotFoundError(
                f"No IMG_*.tif found under {root!r}. Point --root at the directory holding "
                f"the FLAIR patches, or run scripts/download_toy.sh to fetch the toy subset."
            )

        # Keep only patches whose mask is actually present, and honour the domain filter.
        self.samples: List[Tuple[str, str]] = []
        self.domains: List[str] = []
        wanted = set(domains) if domains else None
        for img in all_imgs:
            dom = domain_of(img)
            if wanted is not None and dom not in wanted:
                continue
            msk = mask_path_for(img)
            if msk is None:
                continue
            self.samples.append((img, msk))
            self.domains.append(dom)

        if not self.samples:
            raise FileNotFoundError(
                f"Found {len(all_imgs)} IMG files under {root!r} but none had a matching MSK "
                f"(or none matched domains={domains})."
            )

        self.class_to_indices: Dict[int, List[int]] = {}
        if class_index_file and os.path.exists(class_index_file):
            self._load_class_index(class_index_file)

    def _load_class_index(self, path: str) -> None:
        with open(path, "r") as handle:
            raw = json.load(handle)
        by_path = {img: i for i, (img, _) in enumerate(self.samples)}
        for class_str, paths in raw.items():
            idxs = [by_path[p] for p in paths if p in by_path]
            if idxs:
                self.class_to_indices[int(class_str)] = idxs
        print(f"Loaded class index for {len(self.class_to_indices)} classes from {path}")

    def build_class_index(self, save_to: Optional[str] = None) -> Dict[int, List[int]]:
        """Scan every mask once and record which patches contain which classes.

        Slow (one full pass) but done once. Episodic samplers then get O(1) lookups.
        """
        index: Dict[int, List[str]] = {c: [] for c in FLAIR_CLASSES}
        for i, (img, msk) in enumerate(self.samples):
            mask = _read_tif(msk)[0]
            present, counts = np.unique(mask, return_counts=True)
            for class_id, count in zip(present.tolist(), counts.tolist()):
                if class_id in index and count >= self.min_class_pixels:
                    index[class_id].append(img)
            if (i + 1) % 500 == 0:
                print(f"  indexed {i + 1}/{len(self.samples)} patches", flush=True)

        by_path = {img: i for i, (img, _) in enumerate(self.samples)}
        self.class_to_indices = {
            c: [by_path[p] for p in paths] for c, paths in index.items() if paths
        }
        if save_to:
            os.makedirs(os.path.dirname(os.path.abspath(save_to)), exist_ok=True)
            with open(save_to, "w") as handle:
                json.dump({str(c): p for c, p in index.items() if p}, handle, indent=2)
            print(f"Saved class index to {save_to}")
        return self.class_to_indices

    def __len__(self) -> int:
        return len(self.samples)

    def _make_composite(
        self, bgr_nir: np.ndarray, ndvi: np.ndarray, elev_norm: np.ndarray, shannon: np.ndarray
    ) -> np.ndarray:
        """Assemble the selected 3-channel composite, all channels float32 in [0, 1]."""
        red = bgr_nir[RED]
        green = bgr_nir[GREEN]
        blue = bgr_nir[BLUE]
        nir = bgr_nir[NIR]
        # NDVI is [-1, 1]; rescale so every composite channel shares the [0, 1] range.
        ndvi01 = (ndvi + 1.0) / 2.0

        if self.composite == "rgb":
            return np.stack([red, green, blue])
        if self.composite == "cir":
            return np.stack([nir, red, green])
        if self.composite == "ndvi_dsm_h":
            return np.stack([ndvi01, elev_norm, shannon])
        if self.composite == "rgb_ndvi":
            return np.stack([red, green, ndvi01])
        if self.composite == "rgb_dsm":
            return np.stack([red, green, elev_norm])
        raise AssertionError(f"unhandled composite {self.composite!r}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, msk_path = self.samples[idx]

        raw = _read_tif(img_path).astype(np.float32)
        if raw.shape[0] < 5:
            raise ValueError(
                f"{img_path} has {raw.shape[0]} channels; FLAIR patches must have 5 "
                f"(Blue, Green, Red, NIR, Elevation)."
            )

        # Channels 0-3 are 8-bit reflectance proxies. Scale to [0, 1] only if they are
        # still in DN range -- a pre-normalised copy of the data must not be scaled twice.
        optical = raw[:4]
        if optical.max() > 1.0:
            optical = optical / 255.0

        # Channel 4 is nDSM in metres, not a DN. Clip then normalise independently.
        elevation_m = raw[ELEV]
        elev_norm = np.clip(elevation_m, 0.0, self.elevation_clip_m) / self.elevation_clip_m

        bgr_nir = np.concatenate([optical, elevation_m[np.newaxis]], axis=0)

        ndvi = compute_ndvi_numpy(optical[NIR], optical[RED])

        source = (ndvi + 1.0) / 2.0 if self.shannon_source == "ndvi" else elev_norm
        shannon = compute_shannon_entropy_torch(
            torch.from_numpy(source).unsqueeze(0).unsqueeze(0).float(),
            kernel_size=self.shannon_window,
            num_bins=self.shannon_bins,
        ).squeeze(0).squeeze(0).numpy()

        composite = self._make_composite(bgr_nir, ndvi, elev_norm, shannon)

        mask = _read_tif(msk_path)[0].astype(np.int64)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(composite)).float(),   # (3, 512, 512)
            "mask": torch.from_numpy(mask),                                       # (512, 512) in [1, 19]
            "ndvi": torch.from_numpy(ndvi).unsqueeze(0).float(),                  # (1, 512, 512) [-1, 1]
            "shannon": torch.from_numpy(shannon).unsqueeze(0).float(),            # (1, 512, 512) [0, 1]
            "elevation": torch.from_numpy(elev_norm).unsqueeze(0).float(),        # (1, 512, 512) [0, 1]
            "rgb": torch.from_numpy(
                np.ascontiguousarray(np.stack([optical[RED], optical[GREEN], optical[BLUE]]))
            ).float(),                                                            # (3, 512, 512) always true colour
            "domain": self.domains[idx],
            "path": img_path,
        }


def split_by_domain(
    dataset: FLAIRDataset, holdout: Sequence[str]
) -> Tuple[List[int], List[int]]:
    """Partition indices into (train, heldout) by FLAIR domain.

    Holding out whole domains rather than random patches is the point of FLAIR: domains
    differ in geography and acquisition season, so this measures transfer rather than
    memorisation. Random patch splits leak, because neighbouring patches overlap in
    appearance and lighting.
    """
    holdout_set = set(holdout)
    train, test = [], []
    for i, dom in enumerate(dataset.domains):
        (test if dom in holdout_set else train).append(i)
    return train, test
