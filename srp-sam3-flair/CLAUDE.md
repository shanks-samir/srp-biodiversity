# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements.txt

bash scripts/download_toy.sh ./data        # ~750 MB FLAIR-HUB toy subset
python scripts/inspect_flair.py --root ./data
python tests/test_flair.py                 # plain asserts in __main__, not pytest; runs all, no selection
```

Run scripts from the repo root — they insert the project root into `sys.path` themselves.

`tests/test_flair.py` stubs `datasets.flair._read_tif`, so it needs **no data and no raster
library**. It runs on any machine. Keep it that way: a test that needs a 750 MB download is a
test nobody runs.

## The point of the project

A frozen RGB-only foundation model has three input slots; multispectral aerial imagery has
five channels. **Which three, and do derived ecological quantities beat raw colour for
few-shot vegetation segmentation?**

`FLAIRDataset(composite=...)` is that experiment. Because the intended pipeline is
training-free SAM 3, the composite is the *only* moving part — no learning rate, no episode
count, nothing else to confound it. Preserve that property. If you find yourself adding a
trainable component, stop and reconsider: it costs the project its cleanest measurement.

## FLAIR facts that are easy to get wrong

- **Channel order is B, G, R, NIR, Elevation.** BGR-first, and NIR is index **3**, not 4.
  Constants `BLUE, GREEN, RED, NIR, ELEV` exist in `datasets/flair.py` — use them, never
  integer literals.
- **Channel 4 is nDSM in metres, not an 8-bit DN.** It must not be divided by 255 along with
  the optical bands. It is clipped to `elevation_clip_m` (default 30 m) and normalised
  separately. `test_elevation_scaling` guards this.
- **Mask IDs are 1–19. There is no class 0.** Anything assuming 0-based or background-at-0
  will silently mislabel `building`.
- The band order is taken from the FLAIR *specification*, not read from the files. It is
  load-bearing for every NDVI value. `scripts/inspect_flair.py` verifies it empirically
  (on a vegetated scene the true NIR band must be the brightest) — run it against any new
  download before trusting a result.

## Splitting

Split by **domain**, never randomly. Use `split_by_domain`. FLAIR domains (`D004_2021`) vary
by department *and* acquisition date, so holding them out tests geographic and seasonal
transfer. Neighbouring patches share appearance and lighting, so a random patch split leaks.

For few-shot, use the rotating `FOLDS` rather than a fixed novel pair — a mIoU averaged over
two classes is high-variance and easy to attack. `VEGETATION_CLASSES` is available for an
ecologically-motivated split, but note it recreates a hard base→novel distribution gap
(cartographic → ecological) that confounds "does few-shot work" with "does that transfer
work".

## Shannon entropy

`compute_shannon_entropy_torch` defaults to `padding_mode="replicate"`. **Do not change this
back to zero padding.** Zero padding makes windows at the patch edge see phantom pixels
belonging to no bin, so probabilities stop summing to 1 and the entropy formula reads the
deficit as diversity — a ring of fabricated heterogeneity ~0.13 wide on a *uniform* input,
scaling with window size. `padding_mode="zeros"` remains only for reproducing the
predecessor's numbers.

`shannon_source="elevation"` computes entropy over canopy height rather than NDVI. That is
closer to the landscape-metric meaning of Shannon diversity and survives leaf-off
acquisition, since bare branches still have height variance. Worth preferring, and worth
ablating against `"ndvi"`.

Window size is in pixels at 0.20 m GSD — a 7 px window spans 1.4 m. Sweep it; the
predecessor never did, and its Shannon channel failed to separate any classes.

## Context

Predecessor: `../SRP/` (Phase 1) and `../SRP II.zip/` (Phase 2) — SAM v1 ViT-B on ISPRS
Potsdam, episodic training, best novel mIoU 0.056. `../SRP/SRP_MASTER_SUMMARY.md` has the
full review of why that stalled. The short version: the frozen SAM mask decoder was never
trained to interpret MLP-generated prompt tokens, and SAM 3's native exemplar prompting makes
that whole apparatus unnecessary.

Potsdam was flown leaf-off (tree NDVI ≈ 0.207, overlapping bare soil), so it could not test
the ecological hypothesis at all. That is the main reason for moving to FLAIR.

Do not copy code from the predecessor without re-reading it. `ecological/indices.py` was
carried over and then fixed; other files there carry known defects documented in
`../SRP/CLAUDE.md`.
