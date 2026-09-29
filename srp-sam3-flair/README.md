# Ecological priors for few-shot land-cover segmentation with SAM 3

Few-shot semantic segmentation of vegetation and habitat classes from multispectral aerial
imagery, using a frozen foundation model and ecological channels derived from near-infrared
and canopy height.

Student Research Project 2025/26 · Group 7 · Universität Hildesheim, ISMLL.

This is a clean restart. The predecessor (`ViRefSAM`, SAM v1 + ISPRS Potsdam) is preserved
separately; see [Why a new repository](#why-a-new-repository).

---

## The question

A frozen, RGB-only foundation model has exactly **three input channels**. Multispectral
aerial imagery offers five: blue, green, red, near-infrared, and above-ground height.

**Which three should occupy those slots, and does feeding derived ecological quantities
instead of raw colour improve few-shot segmentation of vegetation classes?**

That framing matters because it makes the input composite the *only* moving part. With a
training-free pipeline there is nothing else to confound the measurement — no learning rate,
no episode count, no adapter initialisation. The ablation is as clean as it can be.

Candidate composites, all implemented in [`datasets/flair.py`](datasets/flair.py):

| Composite | Channels | Rationale |
|---|---|---|
| `rgb` | Red, Green, Blue | Control. What the model was pretrained on. |
| `cir` | NIR, Red, Green | Standard remote-sensing false colour; vegetation renders bright red. Violates pretraining colour statistics, which is precisely what makes it worth measuring. |
| `ndvi_dsm_h` | NDVI, height, Shannon *H* | Discards raw radiometry entirely for ecological quantities. |
| `rgb_ndvi` | Red, Green, NDVI | Keeps two true bands, swaps blue for a vegetation index. |
| `rgb_dsm` | Red, Green, height | Structure against colour, no spectral vegetation signal. |

---

## Dataset: FLAIR (IGN, France)

[FLAIR](https://ignf.github.io/FLAIR/) — French Land cover from Aerospace ImageRy.

- **512 × 512 patches at 0.20 m** GSD, 20.4 billion annotated pixels
- **5 channels: Blue, Green, Red, NIR, Elevation** — note the order; it is BGR-first, and
  NIR is at index **3**, not 4 as on a 5-band MicaSense sensor
- Channel 4 is **nDSM** (DSM − DTM): above-ground height in metres, *not* an 8-bit DN
- **19 classes**, IDs 1–19, no class 0. Vegetation classes include coniferous, deciduous,
  brushwood, vineyard, herbaceous vegetation, agricultural land, plowed land, clear cut
- **55 spatial-temporal domains** across metropolitan France, varying by department *and*
  acquisition date — so holding out domains tests geographic **and seasonal** transfer
- Licence: **Open Licence 2.0 / etalab-2.0**. Free to use and redistribute with attribution
  to IGN; compatible with CC-BY

### Why FLAIR over ISPRS Potsdam

Potsdam was flown **leaf-off**. Measured on the hold-out tiles, tree NDVI averages ≈ 0.207
and overlaps low vegetation almost completely. An NDVI ablation there is answered by the
dataset rather than by the method — there is no vegetation-vigour signal to find.

FLAIR fixes three things at once: season becomes a *variable* rather than a fixed handicap;
19 classes permit proper fold-based few-shot cross-validation instead of a two-class mean;
and the classes are ecological (coniferous vs deciduous) rather than cartographic
(tree vs low vegetation).

The nDSM is a bonus worth stating plainly. The predecessor computed Shannon entropy over
**NDVI** as an indirect proxy for canopy roughness. FLAIR supplies actual per-pixel height,
so entropy can be computed over **structure directly** — closer to the landscape-metric
sense of Shannon diversity, and it survives leaf-off acquisition because bare branches still
have height variance. Set `shannon_source="elevation"`.

---

## Quick start

```bash
pip install -r requirements.txt

# ~750 MB toy subset — enough to verify the loader without the full download
bash scripts/download_toy.sh ./data

# ALWAYS run this against a new download. It checks the band-order assumption
# empirically and reports per-class NDVI and height.
python scripts/inspect_flair.py --root ./data

# Logic tests. TIFF I/O is stubbed, so these run with no data and no raster library.
python tests/test_flair.py
```

`inspect_flair.py` is not optional ceremony. The loader hard-codes a band order taken from
the FLAIR specification rather than read from the files, and that assumption is load-bearing
for every NDVI value the project will produce. The script tests it with physics: on any
vegetated scene the true NIR band must be markedly brighter than the visible bands, so if
channel 3 is not the brightest, the order is wrong and nothing downstream can be trusted.

---

## Usage

```python
from datasets import FLAIRDataset, split_by_domain

ds = FLAIRDataset(
    root="./data",
    composite="cir",              # the experimental knob
    shannon_source="elevation",   # entropy over canopy height, not NDVI
    shannon_window=7,             # 7 px = 1.4 m at 0.20 m GSD
)

sample = ds[0]
sample["image"]      # (3, 512, 512) float32 in [0,1] — the selected composite
sample["rgb"]        # (3, 512, 512) always true colour, for figures
sample["mask"]       # (512, 512) int64, class IDs 1..19
sample["ndvi"]       # (1, 512, 512) [-1, 1]
sample["shannon"]    # (1, 512, 512) [0, 1]
sample["elevation"]  # (1, 512, 512) [0, 1], nDSM clipped to 30 m
sample["domain"]     # e.g. "D004_2021"

# Hold out whole domains, never random patches — neighbouring patches share
# appearance and lighting, so a random split leaks.
train_idx, test_idx = split_by_domain(ds, holdout=["D007_2020"])
```

A one-off class index makes episodic sampling O(1):

```python
ds.build_class_index(save_to="class_index.json")
ds_fast = FLAIRDataset(root="./data", class_index_file="class_index.json")
```

---

## Fixed from the predecessor

**Shannon entropy border artefact.** `compute_shannon_entropy_torch` convolved the one-hot
bin stack using zero padding. Windows straddling a patch edge therefore saw phantom pixels
belonging to no bin, the local probabilities stopped summing to 1, and the entropy formula
read that deficit as diversity.

On a perfectly uniform raster the interior correctly returns ~9 × 10⁻⁶ while **every border
pixel returns ≈ 0.13** — a ring of fabricated heterogeneity around every patch, of width
`kernel_size // 2`. It affects ~2% of a 512 px patch at a 7 px window and ~8% at 21 px, and
since large orthomosaics are tiled into overlapping patches, those rings land in the middle
of real scenes at arbitrary positions.

Now uses `padding_mode="replicate"`, encoding the correct assumption that the landscape
continues past the patch boundary. Pass `padding_mode="zeros"` to reproduce the old numbers.

> This bug is still present in the predecessor repository and affects every Shannon map it
> has produced.

---

## Why a new repository

The predecessor is **SAM v1 ViT-B on ISPRS Potsdam**, trained episodically: masked average
pooling → prototype → MLP → four synthetic tokens → frozen SAM mask decoder. Best novel-class
validation mIoU 0.056.

The diagnosis was that the frozen decoder had never been trained to interpret MLP-generated
tokens, so all gradient had to squeeze through it into a ~400 K-parameter MLP. SAM 3
(released 19 Nov 2025) makes that architecture redundant: its **Promptable Concept
Segmentation** accepts image exemplars natively, through an exemplar encoder and fusion
encoder trained end-to-end for exactly this. Tsai et al. (arXiv 2604.05433) report
state-of-the-art few-shot segmentation from a **fully frozen** SAM 3 with no training at all.

Different foundation model, different dataset, different training regime — effectively no
shared surface beyond the ecological index code, which is carried over here with the border
fix applied.

---

## Layout

```
datasets/flair.py         FLAIR loader; composites, domain splits, class index
ecological/indices.py     NDVI and local Shannon entropy (border artefact fixed)
models/                   SAM 3 wrapper — not yet written
scripts/download_toy.sh   fetch the ~750 MB toy subset
scripts/inspect_flair.py  verify band order and report per-class NDVI/height
tests/test_flair.py       loader logic tests; TIFF I/O stubbed, runs without data
```

---

## Status

Loader, ecological channels and verification tooling are in place and tested. The SAM 3
integration is not yet written.

Not yet verified against real FLAIR files — the assumptions in `inspect_flair.py` exist
precisely because they have not been checked against a download.

---

## Citation

FLAIR is © IGN under Open Licence 2.0. Cite the FLAIR papers
([#1](https://arxiv.org/abs/2211.12979), [#2](https://arxiv.org/abs/2305.14467),
[country-scale](https://arxiv.org/abs/2310.13336)) in any write-up that uses the data.
