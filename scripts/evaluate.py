import sys
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from segment_anything import sam_model_registry
from datasets.potsdam import TorchGeoPotsdamDataset
from datasets.eu_uav import EUMultispectralUAVDataset
from datasets.episode_sampler import FewShotEpisodeSampler
from models.viref_sam.model import ViRefSAM
from scripts.train_virefsam import compute_iou


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViRefSAM Few-Shot Model")
    parser.add_argument("--dataset_type", type=str, choices=["potsdam", "eu_uav"], default="potsdam")
    parser.add_argument("--data_dir", type=str, default="./Potsdam", help="Path to dataset directory (default: ./Potsdam)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained model checkpoint path")
    parser.add_argument("--sam_checkpoint", type=str, default="./checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--k_shot", type=int, default=5, help="1 or 5 shot")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--use_ndvi", action="store_true", default=True)
    parser.add_argument("--use_shannon", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device}...")

    # Load SAM and ViRefSAM
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint if os.path.exists(args.sam_checkpoint) else None)
    model = ViRefSAM(
        sam_model=sam,
        num_prompt_tokens=4,
        adapter_bottleneck=64,
        use_ndvi=args.use_ndvi,
        use_shannon=args.use_shannon,
        freeze_sam=True
    ).to(device)

    # Load trained weights
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}. Running with initialized adapter.")

    model.eval()

    # Setup dataset
    if args.dataset_type == "potsdam":
        target_classes = [3, 4]  # TorchGeo: 3=Low vegetation, 4=Tree
        dataset = TorchGeoPotsdamDataset(
            root=args.data_dir,
            split="val",
            crop_size=512,
            compute_shannon=args.use_shannon,
            use_literature_split=True
        )
        class_names = {3: "Low Vegetation", 4: "Tree"}
    else:
        target_classes = [1]     # Habitat foreground class in UAV dataset
        dataset = EUMultispectralUAVDataset(data_dir=args.data_dir, patch_size=512, compute_shannon=args.use_shannon)
        class_names = {1: "Target Habitat"}

    sampler = FewShotEpisodeSampler(dataset, classes=target_classes, k_shot=args.k_shot, q_queries=1)

    class_scores = {c: [] for c in target_classes}

    print(f"\n--- Evaluating {args.k_shot}-Shot Performance over {args.num_episodes} Episodes ---")
    with torch.no_grad():
        for _ in tqdm(range(args.num_episodes)):
            for c in target_classes:
                episode = sampler.sample_episode(target_class=c)
                s_imgs = episode["support_images"].to(device)
                s_masks = episode["support_masks"].to(device)
                q_imgs = episode["query_images"].to(device)
                q_masks = episode["query_masks"].to(device)
                s_ndvi = episode["support_ndvi"].to(device) if args.use_ndvi else None
                q_ndvi = episode["query_ndvi"].to(device) if args.use_ndvi else None
                s_shn = episode["support_shannon"].to(device) if args.use_shannon else None
                q_shn = episode["query_shannon"].to(device) if args.use_shannon else None

                pred_logits = model.forward_few_shot(
                    s_imgs, s_masks, q_imgs,
                    s_ndvi, q_ndvi, s_shn, q_shn
                )

                iou = compute_iou(pred_logits, q_masks)
                class_scores[c].append(iou)

    print("\n================ Results ================")
    overall_ious = []
    for c in target_classes:
        mean_c_iou = np.mean(class_scores[c])
        overall_ious.append(mean_c_iou)
        print(f"Class: {class_names.get(c, str(c))} | {args.k_shot}-shot mIoU: {mean_c_iou:.4f}")

    total_mIoU = np.mean(overall_ious)
    print(f"-----------------------------------------")
    print(f"Overall Novel Class Mean IoU ({args.k_shot}-shot): {total_mIoU:.4f}")
    print("=========================================\n")


if __name__ == "__main__":
    main()
