#!/bin/bash
#SBATCH --job-name=srp_fewshot_sam
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=shrestha@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# Ensure project root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Download standard SAM ViT-Base checkpoint if not already present
CHECKPOINT_PATH="./checkpoints/sam_vit_b_01ec64.pth"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Downloading SAM ViT-Base pretrained checkpoint..."
    mkdir -p ./checkpoints
    wget -q -O "$CHECKPOINT_PATH" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
fi

# Activate project conda environment
source activate srp_env

# 1. Run the leaf-off / NDVI diagnostic
echo "Running Leaf-Off Condition & NDVI/Shannon distribution check..."
srun python scripts/analyze_potsdam_ndvi.py

# 2. Run 5-shot training with ViRefSAM on Potsdam
echo "Beginning 5-Shot Training on Potsdam with NDVI and Shannon Diversity..."
srun python scripts/train_virefsam.py \
    --data_dir ./Potsdam \
    --sam_checkpoint "$CHECKPOINT_PATH" \
    --model_type vit_b \
    --k_shot 5 \
    --num_episodes 1000 \
    --lr 1e-4 \
    --use_ndvi \
    --use_shannon \
    --output_dir ./outputs

# 3. Run evaluation on hold-out validation tiles
echo "Evaluating 5-Shot Performance..."
srun python scripts/evaluate.py \
    --dataset_type potsdam \
    --data_dir ./Potsdam \
    --checkpoint ./outputs/best_virefsam_model.pth \
    --sam_checkpoint "$CHECKPOINT_PATH" \
    --k_shot 5 \
    --num_episodes 100
