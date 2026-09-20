import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from segment_anything import sam_model_registry
from datasets.potsdam import TorchGeoPotsdamDataset
from datasets.episode_sampler import FewShotEpisodeSampler
from models.viref_sam.model import ViRefSAM


def dice_loss(pred_logits: torch.Tensor, target_mask: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Compute binary Dice Loss."""
    pred_prob = torch.sigmoid(pred_logits)
    intersection = (pred_prob * target_mask).sum(dim=(2, 3))
    union = pred_prob.sum(dim=(2, 3)) + target_mask.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def compute_iou(pred_logits: torch.Tensor, target_mask: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Intersection over Union (IoU)."""
    pred_bin = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred_bin * target_mask).sum().item()
    union = (pred_bin + target_mask).clamp(0, 1).sum().item()
    if union == 0:
        return 1.0
    return intersection / union


def main():
    parser = argparse.ArgumentParser(description="Train ViRefSAM for Few-Shot Remote Sensing")
    parser.add_argument("--data_dir", type=str, default="./Potsdam", help="Path to ISPRS Potsdam dataset (default: ./Potsdam)")
    parser.add_argument("--sam_checkpoint", type=str, default="./checkpoints/sam_vit_b_01ec64.pth", help="SAM checkpoint path")
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--num_episodes", type=int, default=500, help="Total training episodes")
    parser.add_argument("--val_episodes", type=int, default=50, help="Validation episodes")
    parser.add_argument("--k_shot", type=int, default=5, help="Number of support examples")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for adapter and prompt encoder")
    parser.add_argument("--use_ndvi", action="store_true", default=True, help="Fuse NDVI channel")
    parser.add_argument("--use_shannon", action="store_true", default=True, help="Fuse Shannon diversity channel")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # 1. Load Pretrained SAM
    print(f"Loading SAM model ({args.model_type}) from {args.sam_checkpoint}...")
    if not os.path.exists(args.sam_checkpoint):
        print(f"Warning: Checkpoint not found at {args.sam_checkpoint}. Downloading or placing it is required on cluster.")
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint if os.path.exists(args.sam_checkpoint) else None)
    sam.to(device)

    # 2. Initialize ViRefSAM wrapper
    model = ViRefSAM(
        sam_model=sam,
        num_prompt_tokens=4,
        adapter_bottleneck=64,
        use_ndvi=args.use_ndvi,
        use_shannon=args.use_shannon,
        freeze_sam=True
    ).to(device)

    # Only train the new modules (Adapter and Context Prompt Encoder)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_episodes, eta_min=1e-6)

    # 3. Setup Dataset and Episode Samplers using TorchGeo
    # Official TorchGeo Potsdam2D classes:
    # 0: Clutter, 1: Impervious, 2: Building, 3: Low Veg (Novel 1), 4: Tree (Novel 2), 5: Car
    base_classes = [1, 2, 5]
    novel_classes = [3, 4]

    # Literature split: hold-out tiles 7_8, 4_10, 2_11, 5_11 for validation, remaining 20 for training
    train_dataset = TorchGeoPotsdamDataset(root=args.data_dir, split="train", crop_size=512, compute_shannon=args.use_shannon, use_literature_split=True)
    val_dataset = TorchGeoPotsdamDataset(root=args.data_dir, split="val", crop_size=512, compute_shannon=args.use_shannon, use_literature_split=True)

    train_sampler = FewShotEpisodeSampler(train_dataset, classes=base_classes, k_shot=args.k_shot, q_queries=1)
    val_sampler = FewShotEpisodeSampler(val_dataset, classes=novel_classes, k_shot=args.k_shot, q_queries=1)

    print("Starting few-shot episodic training...")
    best_val_iou = 0.0

    for episode_idx in range(1, args.num_episodes + 1):
        model.train()
        episode = train_sampler.sample_episode()

        support_imgs = episode["support_images"].to(device)
        support_masks = episode["support_masks"].to(device)
        query_imgs = episode["query_images"].to(device)
        query_masks = episode["query_masks"].to(device)
        support_ndvi = episode["support_ndvi"].to(device) if args.use_ndvi else None
        query_ndvi = episode["query_ndvi"].to(device) if args.use_ndvi else None
        support_shannon = episode["support_shannon"].to(device) if args.use_shannon else None
        query_shannon = episode["query_shannon"].to(device) if args.use_shannon else None

        optimizer.zero_grad()
        pred_logits = model.forward_few_shot(
            support_imgs, support_masks, query_imgs,
            support_ndvi, query_ndvi, support_shannon, query_shannon
        )

        bce = F.binary_cross_entropy_with_logits(pred_logits, query_masks)
        dice = dice_loss(pred_logits, query_masks)
        total_loss = bce + dice

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if episode_idx % 20 == 0 or episode_idx == args.num_episodes:
            # Run validation on novel classes
            model.eval()
            val_ious = []
            with torch.no_grad():
                for _ in range(args.val_episodes):
                    val_ep = val_sampler.sample_episode()
                    s_imgs = val_ep["support_images"].to(device)
                    s_masks = val_ep["support_masks"].to(device)
                    q_imgs = val_ep["query_images"].to(device)
                    q_masks = val_ep["query_masks"].to(device)
                    s_ndvi = val_ep["support_ndvi"].to(device) if args.use_ndvi else None
                    q_ndvi = val_ep["query_ndvi"].to(device) if args.use_ndvi else None
                    s_shn = val_ep["support_shannon"].to(device) if args.use_shannon else None
                    q_shn = val_ep["query_shannon"].to(device) if args.use_shannon else None

                    val_pred = model.forward_few_shot(
                        s_imgs, s_masks, q_imgs,
                        s_ndvi, q_ndvi, s_shn, q_shn
                    )
                    iou = compute_iou(val_pred, q_masks)
                    val_ious.append(iou)

            mean_val_iou = sum(val_ious) / len(val_ious)
            print(f"Episode {episode_idx}/{args.num_episodes} | Train Loss: {total_loss.item():.4f} | Novel Classes Val mIoU: {mean_val_iou:.4f}")

            if mean_val_iou > best_val_iou:
                best_val_iou = mean_val_iou
                ckpt_path = os.path.join(args.output_dir, "best_virefsam_model.pth")
                torch.save({
                    "episode": episode_idx,
                    "model_state_dict": model.state_dict(),
                    "best_val_iou": best_val_iou,
                }, ckpt_path)
                print(f"--> Saved new best checkpoint to {ckpt_path} (mIoU: {best_val_iou:.4f})")

    print(f"Training complete! Best Novel Class mIoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()
