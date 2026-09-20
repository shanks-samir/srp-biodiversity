#!/bin/bash
#SBATCH --job-name=srp_fewshot_sam
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu           # Adjust partition name to your cluster (e.g. gpu, gpu_long)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1              # Request 1 GPU (e.g. V100, A100, RTX6000)
#SBATCH --time=12:00:00

# -------------------------------------------------------------
# University Cluster Environment Setup
# -------------------------------------------------------------
echo "Starting Few-Shot SAM Job on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Load modules (uncomment/adjust according to your cluster's module system)
# module load cuda/12.1
# module load python/3.10

# Activate conda environment
# source activate srp_env

# Ensure project root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create directories
mkdir -p logs outputs checkpoints

# Download standard SAM ViT-Base checkpoint if not already present
CHECKPOINT_PATH="./checkpoints/sam_vit_b_01ec64.pth"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Downloading SAM ViT-Base pretrained checkpoint..."
    wget -q -O "$CHECKPOINT_PATH" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
fi

# -------------------------------------------------------------
# 1. Run Leaf-Off & Ecology Distribution Diagnostic
# -------------------------------------------------------------
POTSDAM_DATA_DIR="${POTSDAM_DATA_DIR:-./Potsdam}"

echo "Running Leaf-Off Condition & NDVI/Shannon distribution check..."
python scripts/analyze_potsdam_ndvi.py

# -------------------------------------------------------------
# 2. Run 5-Shot Training on ISPRS Potsdam (with NDVI + Shannon)
# -------------------------------------------------------------

echo "Beginning 5-Shot Training on Potsdam with NDVI and Shannon Diversity..."
python scripts/train_virefsam.py \
    --data_dir "$POTSDAM_DATA_DIR" \
    --sam_checkpoint "$CHECKPOINT_PATH" \
    --model_type vit_b \
    --k_shot 5 \
    --num_episodes 1000 \
    --lr 1e-4 \
    --use_ndvi \
    --use_shannon \
    --output_dir ./outputs

# -------------------------------------------------------------
# 2. Run Evaluation (1-Shot & 5-Shot)
# -------------------------------------------------------------
echo "Evaluating 5-Shot Performance..."
python scripts/evaluate.py \
    --dataset_type potsdam \
    --data_dir "$POTSDAM_DATA_DIR" \
    --checkpoint ./outputs/best_virefsam_model.pth \
    --sam_checkpoint "$CHECKPOINT_PATH" \
    --k_shot 5 \
    --num_episodes 100

echo "Evaluating 1-Shot Performance..."
python scripts/evaluate.py \
    --dataset_type potsdam \
    --data_dir "$POTSDAM_DATA_DIR" \
    --checkpoint ./outputs/best_virefsam_model.pth \
    --sam_checkpoint "$CHECKPOINT_PATH" \
    --k_shot 1 \
    --num_episodes 100

echo "Job completed successfully at $(date)"
