import sys
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from segment_anything import sam_model_registry
from datasets.potsdam import TorchGeoPotsdamDataset
from datasets.episode_sampler import FewShotEpisodeSampler
from models.viref_sam.model import ViRefSAM
from scripts.train_virefsam import compute_iou


def main():
    parser = argparse.ArgumentParser(description="Visualize Few-Shot SAM Predictions")
    parser.add_argument("--data_dir", type=str, default="./Potsdam")
    parser.add_argument("--checkpoint", type=str, default="./outputs/best_virefsam_model.pth")
    parser.add_argument("--sam_checkpoint", type=str, default="./checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint if os.path.exists(args.sam_checkpoint) else None)
    model = ViRefSAM(sam_model=sam, num_prompt_tokens=4, adapter_bottleneck=64, use_ndvi=True, use_shannon=True).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {args.checkpoint}")
    model.eval()

    # Load Validation Set
    dataset = TorchGeoPotsdamDataset(root=args.data_dir, split="val", crop_size=512, compute_shannon=True, use_literature_split=True)
    classes = {3: "Low Vegetation", 4: "Tree"}

    sampler = FewShotEpisodeSampler(dataset, classes=[3, 4], k_shot=args.k_shot, q_queries=1)

    print("Generating qualitative visualization figures...")
    with torch.no_grad():
        for target_class, class_name in classes.items():
            episode = sampler.sample_episode(target_class=target_class)
            s_imgs = episode["support_images"].to(device)
            s_masks = episode["support_masks"].to(device)
            q_imgs = episode["query_images"].to(device)
            q_masks = episode["query_masks"].to(device)
            s_ndvi = episode["support_ndvi"].to(device)
            q_ndvi = episode["query_ndvi"].to(device)
            s_shn = episode["support_shannon"].to(device)
            q_shn = episode["query_shannon"].to(device)

            pred_logits = model.forward_few_shot(s_imgs, s_masks, q_imgs, s_ndvi, q_ndvi, s_shn, q_shn)
            pred_prob = torch.sigmoid(pred_logits).squeeze().cpu().numpy()
            pred_bin = (pred_prob > 0.5).astype(np.float32)

            gt_mask = q_masks.squeeze().cpu().numpy()
            rgb_img = q_imgs.squeeze().permute(1, 2, 0).cpu().numpy()
            ndvi_img = q_ndvi.squeeze().cpu().numpy()
            shn_img = q_shn.squeeze().cpu().numpy()

            iou = compute_iou(pred_logits, q_masks)

            # Create Figure
            fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

            axes[0].imshow(np.clip(rgb_img, 0, 1))
            axes[0].set_title(f"Query RGB ({class_name})")
            axes[0].axis("off")

            axes[1].imshow(ndvi_img, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
            axes[1].set_title("Computed NDVI")
            axes[1].axis("off")

            axes[2].imshow(shn_img, cmap="magma", vmin=0, vmax=0.6)
            axes[2].set_title("Shannon Diversity (H)")
            axes[2].axis("off")

            axes[3].imshow(gt_mask, cmap="gray")
            axes[3].set_title(f"Ground Truth: {class_name}")
            axes[3].axis("off")

            axes[4].imshow(pred_bin, cmap="viridis")
            axes[4].set_title(f"ViRefSAM Prediction (IoU: {iou:.3f})")
            axes[4].axis("off")

            plt.tight_layout()
            out_file = os.path.join(args.output_dir, f"qualitative_{class_name.lower().replace(' ', '_')}.png")
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved visualization to {out_file} (IoU: {iou:.4f})")


if __name__ == "__main__":
    main()
