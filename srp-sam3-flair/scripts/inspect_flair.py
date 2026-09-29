"""Verify the FLAIR loader's assumptions against real files before trusting any result.

The loader hard-codes a channel order (Blue, Green, Red, NIR, Elevation) taken from the
FLAIR #1 specification rather than from the files on disk. That assumption is load-bearing
for every NDVI value this project will compute, and it is silently wrong if a release ever
reorders the bands. This script checks it empirically and prints everything needed to sanity
-check the ecological channels.

Run this first, every time you point the code at a new FLAIR download.

    python scripts/inspect_flair.py --root ./data
"""

import os
import sys
import argparse
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.flair import (  # noqa: E402
    FLAIRDataset, FLAIR_CLASSES, GSD_METRES, _read_tif, BLUE, GREEN, RED, NIR, ELEV,
)


def check_channel_order(dataset: FLAIRDataset, num_samples: int = 20) -> None:
    """Test the band-order assumption using physics rather than documentation.

    Vegetated pixels reflect strongly in near-infrared and absorb in red, so whichever
    band is genuinely NIR should have a markedly higher mean than the visible bands over
    any scene with plant cover. If band 3 is not the brightest on vegetated patches, the
    assumed order is wrong and every NDVI in this project is meaningless.
    """
    print("\n=== Channel order check ===")
    print("Assumption: 0=Blue, 1=Green, 2=Red, 3=NIR, 4=Elevation(nDSM, metres)\n")

    means = np.zeros(5, dtype=np.float64)
    n = min(num_samples, len(dataset))
    elev_all = []

    for i in range(n):
        img_path, _ = dataset.samples[i]
        raw = _read_tif(img_path).astype(np.float32)
        optical = raw[:4]
        if optical.max() > 1.0:
            optical = optical / 255.0
        means += np.array([optical[c].mean() for c in range(4)] + [0.0])
        elev_all.append(raw[ELEV])

    means[:4] /= n
    names = ["ch0 (assumed Blue)", "ch1 (assumed Green)", "ch2 (assumed Red)", "ch3 (assumed NIR)"]
    for name, value in zip(names, means[:4]):
        print(f"  {name:26} mean = {value:.4f}")

    nir_idx = int(np.argmax(means[:4]))
    if nir_idx == NIR:
        print("\n  PASS: channel 3 is the brightest, consistent with it being NIR.")
    else:
        print(f"\n  *** FAIL: channel {nir_idx} is brightest, not channel 3. ***")
        print("  The assumed band order is probably wrong. Do NOT trust any NDVI until this")
        print("  is resolved -- check the release notes for the FLAIR variant you downloaded.")

    elev = np.concatenate([e.ravel() for e in elev_all])
    print(f"\n  ch4 (assumed Elevation): min={elev.min():.2f}  max={elev.max():.2f}  "
          f"mean={elev.mean():.2f}  p99={np.percentile(elev, 99):.2f}")
    if elev.max() <= 1.001 and elev.min() >= -0.001:
        print("  NOTE: elevation looks pre-normalised to [0, 1], not metres. Set")
        print("        elevation_clip_m=1.0 or the height channel will be crushed.")
    elif elev.max() > 200:
        print("  NOTE: values above 200 m are almost certainly artefacts. The default")
        print("        elevation_clip_m=30 already handles this, but check the p99.")
    else:
        print("  Consistent with nDSM in metres (above-ground height).")


def check_ndvi_separation(dataset: FLAIRDataset, num_samples: int = 30) -> None:
    """Report per-class NDVI and height, which is what decides whether this dataset can
    actually test the ecological hypothesis.

    On the previous dataset (ISPRS Potsdam, flown leaf-off) tree NDVI collapsed to ~0.207
    and overlapped bare soil, which meant the NDVI ablation was answered by the data rather
    than by the method. The check below is the equivalent diagnostic for FLAIR. Vegetation
    classes should sit clearly above built classes; if they do not, look at which domains
    were sampled before blaming the method.
    """
    print("\n=== Per-class NDVI and height ===")
    ndvi_by_class, elev_by_class = {}, {}

    n = min(num_samples, len(dataset))
    for i in range(n):
        sample = dataset[i]
        ndvi = sample["ndvi"].squeeze(0).numpy()
        elev = sample["elevation"].squeeze(0).numpy()
        mask = sample["mask"].numpy()
        # Subsample: every 8th pixel is plenty for a distribution and 64x cheaper.
        ndvi, elev, mask = ndvi[::8, ::8], elev[::8, ::8], mask[::8, ::8]
        for class_id in np.unique(mask):
            sel = mask == class_id
            ndvi_by_class.setdefault(int(class_id), []).append(ndvi[sel])
            elev_by_class.setdefault(int(class_id), []).append(elev[sel])

    print(f"  (from {n} patches)\n")
    print(f"  {'class':<24} {'pixels':>9}  {'NDVI mean':>10} {'sd':>7}   {'height':>8}")
    print("  " + "-" * 66)
    for class_id in sorted(ndvi_by_class):
        vals = np.concatenate(ndvi_by_class[class_id])
        hts = np.concatenate(elev_by_class[class_id])
        name = FLAIR_CLASSES.get(class_id, f"unknown({class_id})")
        print(f"  {name:<24} {len(vals):>9,}  {vals.mean():>10.3f} {vals.std():>7.3f}   {hts.mean():>8.3f}")

    veg = [c for c in (6, 7, 8, 10) if c in ndvi_by_class]
    built = [c for c in (1, 3) if c in ndvi_by_class]
    if veg and built:
        veg_mean = np.concatenate([np.concatenate(ndvi_by_class[c]) for c in veg]).mean()
        built_mean = np.concatenate([np.concatenate(ndvi_by_class[c]) for c in built]).mean()
        print(f"\n  vegetation NDVI {veg_mean:.3f} vs built {built_mean:.3f} "
              f"(separation {veg_mean - built_mean:+.3f})")
        if veg_mean - built_mean < 0.15:
            print("  *** Weak separation. Check whether these domains are leaf-off before")
            print("      concluding anything about the method. ***")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a FLAIR download and the loader's assumptions")
    parser.add_argument("--root", type=str, default="./data", help="Directory holding FLAIR patches")
    parser.add_argument("--samples", type=int, default=30, help="Patches to sample for statistics")
    parser.add_argument("--composite", type=str, default="rgb", help="Composite to instantiate")
    args = parser.parse_args()

    dataset = FLAIRDataset(root=args.root, composite=args.composite)

    print("=== Dataset ===")
    print(f"  root            {os.path.abspath(args.root)}")
    print(f"  patches         {len(dataset):,}")
    print(f"  GSD             {GSD_METRES} m  (a {dataset.shannon_window}px Shannon window "
          f"spans {dataset.shannon_window * GSD_METRES:.1f} m)")

    domains = Counter(dataset.domains)
    print(f"  domains         {len(domains)}")
    for dom, count in sorted(domains.items())[:12]:
        print(f"                    {dom}: {count:,} patches")
    if len(domains) > 12:
        print(f"                    ... and {len(domains) - 12} more")

    sample = dataset[0]
    print("\n=== Sample 0 tensors ===")
    for key in ("image", "rgb", "mask", "ndvi", "shannon", "elevation"):
        tensor = sample[key]
        print(f"  {key:<10} {str(tuple(tensor.shape)):<18} {str(tensor.dtype):<14} "
              f"[{tensor.float().min():.3f}, {tensor.float().max():.3f}]")
    print(f"  domain     {sample['domain']}")

    check_channel_order(dataset, args.samples)
    check_ndvi_separation(dataset, args.samples)

    print("\n=== Composites ===")
    for name in FLAIRDataset.COMPOSITES:
        probe = FLAIRDataset(root=args.root, composite=name)
        img = probe[0]["image"]
        print(f"  {name:<12} {str(tuple(img.shape)):<18} "
              f"[{img.min():.3f}, {img.max():.3f}]  per-channel means "
              f"{[round(float(img[c].mean()), 3) for c in range(3)]}")

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
