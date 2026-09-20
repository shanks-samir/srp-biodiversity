# Few-Shot Semantic Segmentation for Biodiversity Assessment Using SAM and Multispectral UAV Imagery

This repository contains the implementation of the few-shot semantic segmentation pipeline for habitat and biodiversity assessment using the **Segment Anything Model (SAM)**, visual contextual prompting (**ViRefSAM**), and ecological channels (**NDVI** and **local Shannon Diversity Index**).

Designed for evaluation on **ISPRS Potsdam** (IRRG) and cross-country transfer to the **EU Multispectral UAV** dataset (Portugal, Germany, UK, Austria).

---

## 1. Architecture Overview

```
                      +-----------------------------+
                      |   Support Set (K images)    |
                      |   + K Target Class Masks    |
                      +--------------+--------------+
                                     |
                                     v
                       [Visual Context Prompt Encoder]
                                     |
                                     | Context Tokens
                                     v
+----------------+          +-----------------+          +---------------------+
| Query RGB Tile | -------> |   Frozen SAM    | -------> |  SAM Mask Decoder   |
+----------------+          |  Image Encoder  |          +----------+----------+
                            +--------+--------+                     |
                                     |                              |
                                     v (Query Features)             |
                            +-----------------+                     |
                            |   Ecological    |                     |
   Query NDVI    ---------> | Dynamic Target  |                     |
   Query Shannon ---------> |    Alignment    | --------------------+
                            |     Adapter     |   (Adapted Features)
                            +-----------------+
                                                                    |
                                                                    v
                                                          Predicted Query Mask
```

1. **Frozen Foundation Model**: The core SAM vision encoder remains frozen, preserving its powerful zero-shot general vision representations.
2. **Visual Contextual Prompt Encoder**: Extracts class-specific visual semantics from the $K$ support examples ($K \in \{1, 5\}$) to automatically prompt the mask decoder (no manual point-clicking required).
3. **Dynamic Target Alignment / Ecological Adapter**: Fuses computed vegetation vigour (NDVI) and spatial heterogeneity (Shannon Diversity Index) into the query feature space via lightweight residual bottleneck blocks.

---
 
 ## 2. ISPRS Potsdam Leaf-Off Condition & Ecology Channels

> [!WARNING]
> **Seasonal Acquisition Constraint (Leaf-Off Deciduous Trees)**:
> ISPRS Potsdam was captured during **leaf-off** conditions. Because deciduous trees have bare canopies, their NDVI signature is suppressed and can overlap heavily with impervious surfaces or bare soil.
>
> **Experimental Strategy**:
> 1. **Diagnostic check**: Run `python scripts/analyze_potsdam_ndvi.py` to inspect the empirical NDVI distribution of `tree` vs. `impervious` vs. `low_vegetation`.
> 2. **Shannon Heterogeneity**: Rely on the **local Shannon Diversity Index** ($H$) on Potsdam. Bare canopies maintain complex branch texture and high spatial entropy, which separates them from smooth impervious ground regardless of chlorophyll levels.
> 3. **Validation Split**: We use the standard benchmark hold-out validation tiles cited in the remote sensing literature:
>    - **Validation tiles (4)**: `7_8`, `4_10`, `2_11`, `5_11`
>    - **Training tiles (20)**: Remaining labelled tiles
>    - *Reference*: [Rottensteiner et al., 2012](https://doi.org/10.5194/isprsannals-I-3-293-2012).
> 4. **Multispectral UAV Fallback**: The EU UAV dataset (collected during growing seasons) provides the primary validation for the true NDVI vegetation vigour signal.

---

## 3. EU Multispectral UAV Dataset Format

Based on surveys across European ecological research sites (Portugal, Germany, UK, Austria), UAV multispectral data typically follows one of two formats:

1. **Standard Multi-band GeoTIFF Stacks**:
   - 5-band sensor (e.g. MicaSense RedEdge / Altum):
     - `Band 1`: Blue (475 nm)
     - `Band 2`: Green (560 nm)
     - `Band 3`: Red (668 nm)
     - `Band 4`: Red Edge (717 nm)
     - `Band 5`: Near-Infrared / NIR (840 nm)
   - 4-band sensor (e.g. DJI Phantom 4 Multispectral): Green, Red, Red Edge, NIR.
2. **Single-band TIFF series**:
   - Files named with suffixes (e.g., `_B.tif`, `_G.tif`, `_R.tif`, `_NIR.tif`).

The dataloader in [`datasets/eu_uav.py`](file:///Users/samirpaudyal/Downloads/SRP/datasets/eu_uav.py) is built to automatically handle both formats, performing dynamic tiling ($512 \times 512$ patches) across large orthomosaics and computing NDVI on the fly.

---

## 3. Directory Structure

```
.
├── datasets/
│   ├── potsdam.py              # ISPRS Potsdam IRRG + label loader
│   ├── eu_uav.py               # Flexible EU Multispectral UAV loader
│   └── episode_sampler.py      # K-shot support/query episodic sampler
├── ecological/
│   ├── indices.py              # PyTorch/NumPy NDVI and Shannon Entropy
│   └── prompt_heuristics.py    # Automatic point/box prompt extraction
├── models/
│   └── viref_sam/
│       ├── context_encoder.py  # Visual contextual prompt encoder
│       ├── ecological_adapter.py # Dynamic Target Alignment Adapter
│       └── model.py            # End-to-end ViRefSAM wrapper
├── scripts/
│   ├── train_virefsam.py       # Episodic few-shot training on Potsdam
│   ├── evaluate.py             # Evaluation across K-shot novel classes
│   └── submit_cluster_job.sh   # SLURM batch job script for University cluster
├── tests/
│   └── test_pipeline.py        # Architecture & index unit tests
└── requirements.txt
```

---

## 4. Running on the University Cluster

### Step 1: Verify Dataset & Install Requirements
```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git

# Verify that the Potsdam data archives are properly recognized
python scripts/verify_potsdam_data.py
```

### Step 2: Submit SLURM Batch Job
The submission script is pre-configured with `POTSDAM_DATA_DIR="./Potsdam"`. Adjust your cluster partition inside [`scripts/submit_cluster_job.sh`](file:///Users/samirpaudyal/Downloads/SRP/scripts/submit_cluster_job.sh), then submit:
```bash
sbatch scripts/submit_cluster_job.sh
```

### Step 3: Run Interactive Training (Optional)
```bash
python scripts/train_virefsam.py \
    --data_dir ./Potsdam \
    --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --k_shot 5 \
    --num_episodes 1000 \
    --use_ndvi \
    --use_shannon
```

### Step 4: Evaluate on EU UAV Dataset
```bash
python scripts/evaluate.py \
    --dataset_type eu_uav \
    --data_dir /path/to/eu_uav_data \
    --checkpoint ./outputs/best_virefsam_model.pth \
    --k_shot 5
```
