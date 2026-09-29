#!/bin/bash
#SBATCH --job-name=srp_sam3_flair
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=shresthap@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ---------------------------------------------------------------------------
# University cluster job. Submit with:  sbatch run.sh
#
# There is NO TRAINING in this pipeline. SAM 3 few-shot segmentation is
# training-free -- the model stays frozen and the only thing that varies is
# which three channels are fed to it. So this job prepares the data and then
# runs an evaluation sweep. It is GPU-bound only in stage 5.
#
# Stages 1-4 need no GPU at all. If the queue for GPU nodes is long, run those
# on a CPU partition first; stage 4 in particular is the slow one.
# ---------------------------------------------------------------------------
set -euo pipefail

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
mkdir -p logs outputs

# Adjust to match your cluster's module system / environment name.
# module load cuda/12.1
source activate srp_env

FLAIR_DATA_DIR="${FLAIR_DATA_DIR:-./data}"
CLASS_INDEX="${CLASS_INDEX:-./outputs/flair_class_index.json}"
COMPOSITES="${COMPOSITES:-rgb cir ndvi_dsm_h rgb_ndvi rgb_dsm}"
SHANNON_SOURCE="${SHANNON_SOURCE:-elevation}"
K_SHOT="${K_SHOT:-5}"
HOLDOUT_DOMAINS="${HOLDOUT_DOMAINS:-}"

# ---------------------------------------------------------------------------
# 1. Data present?
# ---------------------------------------------------------------------------
# Compute nodes frequently have no outbound network, so the download is NOT done
# here. Fetch it on a login node first:  bash scripts/download_toy.sh ./data
if [ ! -d "$FLAIR_DATA_DIR" ] || [ -z "$(find "$FLAIR_DATA_DIR" -name 'IMG_*.tif' -print -quit 2>/dev/null)" ]; then
    echo "ERROR: no FLAIR patches found under $FLAIR_DATA_DIR"
    echo "Fetch them on a login node first:"
    echo "    bash scripts/download_toy.sh $FLAIR_DATA_DIR       # ~750 MB toy subset"
    echo "    # or download FLAIR #1 from https://ignf.github.io/FLAIR/"
    echo "Then resubmit, or set FLAIR_DATA_DIR to an existing copy."
    exit 1
fi
echo "Found $(find "$FLAIR_DATA_DIR" -name 'IMG_*.tif' | wc -l | tr -d ' ') FLAIR patches in $FLAIR_DATA_DIR"

# ---------------------------------------------------------------------------
# 2. Loader logic tests (fast, no data, no GPU)
# ---------------------------------------------------------------------------
echo
echo "=== Stage 2: loader tests ==="
srun --ntasks=1 python tests/test_flair.py

# ---------------------------------------------------------------------------
# 3. Verify the band-order assumption against the actual download
# ---------------------------------------------------------------------------
# This is not ceremony. The loader hard-codes B,G,R,NIR,Elevation from the FLAIR
# spec rather than reading it from the files, and that assumption determines every
# NDVI value the project produces. The script checks it with physics: on a
# vegetated scene the true NIR band must be the brightest.
#
# READ THE OUTPUT. If it prints a FAIL on the channel-order check, stop -- every
# downstream number is meaningless until it is resolved.
echo
echo "=== Stage 3: verify band order and per-class NDVI ==="
srun --ntasks=1 python scripts/inspect_flair.py --root "$FLAIR_DATA_DIR" --samples 50

# ---------------------------------------------------------------------------
# 4. Build the class index (slow: one full pass over every mask)
# ---------------------------------------------------------------------------
# Episodic sampling needs to know which patches contain which of the 19 classes.
# Done once and cached; skipped if the JSON already exists.
echo
echo "=== Stage 4: class index ==="
if [ -f "$CLASS_INDEX" ]; then
    echo "Class index already at $CLASS_INDEX -- skipping."
else
    srun --ntasks=1 python -c "
import sys; sys.path.insert(0, '.')
from datasets.flair import FLAIRDataset
ds = FLAIRDataset(root='$FLAIR_DATA_DIR')
print(f'Indexing {len(ds)} patches across 19 classes...', flush=True)
idx = ds.build_class_index(save_to='$CLASS_INDEX')
for c, patches in sorted(idx.items()):
    print(f'  class {c:2d}: {len(patches):6d} patches')
"
fi

# ---------------------------------------------------------------------------
# 5. Few-shot evaluation sweep (GPU)
# ---------------------------------------------------------------------------
# The experiment: hold the model frozen, vary only which three channels it sees.
echo
echo "=== Stage 5: composite sweep, ${K_SHOT}-shot ==="
if [ ! -f scripts/evaluate_sam3.py ]; then
    echo "SKIPPED: scripts/evaluate_sam3.py does not exist yet."
    echo "The SAM 3 wrapper is not written. Stages 1-4 have done the expensive"
    echo "preparation, so once the wrapper lands this job resumes cheaply."
    echo
    echo "Job finished (data prepared) at $(date)"
    exit 0
fi

for COMPOSITE in $COMPOSITES; do
    echo
    echo "--- composite: $COMPOSITE ---"
    srun --ntasks=1 python scripts/evaluate_sam3.py \
        --root "$FLAIR_DATA_DIR" \
        --class_index "$CLASS_INDEX" \
        --composite "$COMPOSITE" \
        --shannon_source "$SHANNON_SOURCE" \
        --k_shot "$K_SHOT" \
        ${HOLDOUT_DOMAINS:+--holdout_domains $HOLDOUT_DOMAINS} \
        --output_dir ./outputs
done

echo
echo "=== Summary ==="
srun --ntasks=1 python -c "
import json, glob, os
rows = []
for f in sorted(glob.glob('./outputs/results_*.json')):
    with open(f) as fh: rows.append(json.load(fh))
if not rows:
    print('No results written.'); raise SystemExit
print(f\"{'composite':<14}{'k':>3}{'mIoU':>9}{'std':>8}\")
for r in sorted(rows, key=lambda r: -r.get('miou', 0)):
    print(f\"{r.get('composite','?'):<14}{r.get('k_shot','?'):>3}{r.get('miou',0):>9.4f}{r.get('std',0):>8.4f}\")
"

echo
echo "Job completed at $(date)"
