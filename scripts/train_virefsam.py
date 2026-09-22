import sys
import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from segment_anything import sam_model_registry
from datasets.potsdam import TorchGeoPotsdamDataset
from datasets.potsdam_patched import PotsdamPatchDataset
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
    parser.add_argument("--data_dir", type=str, default="./Potsdam", help="Path to ISPRS Potsdam dataset")
    parser.add_argument("--patch_dir", type=str, default=None, help="Path to pre-cropped 512x512 patches")
    parser.add_argument("--sam_checkpoint", type=str, default="./checkpoints/sam_vit_b_01ec64.pth", help="SAM checkpoint path")
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--num_episodes", type=int, default=2000, help="Total training episodes")
    parser.add_argument("--val_episodes", type=int, default=200, help="Validation episodes")
    parser.add_argument("--val_freq", type=int, default=50, help="Validation frequency in episodes")
    parser.add_argument("--k_shot", type=int, default=5, help="Number of support examples")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for adapter and prompt encoder")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient norm clipping")
    parser.add_argument("--use_ndvi", action="store_true", default=True, help="Fuse NDVI channel")
    parser.add_argument("--use_shannon", action="store_true", default=True, help="Fuse Shannon diversity channel")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # 1. Load Pretrained SAM
    if not os.path.exists(args.sam_checkpoint):
        print(f"SAM checkpoint not found at {args.sam_checkpoint}. Downloading standard vit_b weights...")
        os.makedirs(os.path.dirname(args.sam_checkpoint) or ".", exist_ok=True)
        try:
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            urllib.request.urlretrieve(url, args.sam_checkpoint)
            print(f"Downloaded SAM weights to {args.sam_checkpoint}")
        except Exception as e:
            print(f"Could not auto-download SAM weights ({e}). Initializing without preloaded weights.")

    print(f"Loading SAM model ({args.model_type}) from {args.sam_checkpoint}...")
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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"ViRefSAM initialized. Trainable parameters: {num_trainable:,} (Backbone frozen)")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_episodes, eta_min=1e-6)

    start_episode = 1
    best_val_iou = 0.0

    # Resume from checkpoint if requested
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        start_episode = ckpt.get("episode", 0) + 1
        best_val_iou = ckpt.get("best_val_iou", 0.0)
        print(f"Resumed from {args.resume} at Episode {start_episode} (Previous Best Val mIoU: {best_val_iou:.4f})")

    # 3. Setup Dataset and Episode Samplers
    # Base classes: 1: Impervious, 2: Building, 5: Car
    # Novel classes: 3: Low Veg, 4: Tree
    base_classes = [1, 2, 5]
    novel_classes = [3, 4]

    # Auto-detect pre-cropped native patches directory
    auto_patch_dir = args.patch_dir or os.path.join(args.data_dir, "patches_512")
    using_patches = os.path.isdir(os.path.join(auto_patch_dir, "train"))

    if using_patches:
        print(f"Using pre-cropped native 512x512 patches from {auto_patch_dir}")
        train_dataset = PotsdamPatchDataset(patch_dir=os.path.join(auto_patch_dir, "train"))
        val_dataset = PotsdamPatchDataset(patch_dir=os.path.join(auto_patch_dir, "val"))
    else:
        print(f"Note: Pre-cropped patches not found at {auto_patch_dir}. Falling back to on-the-fly tile resizing.")
        print(f"Tip: Run 'python scripts/preprocess_potsdam_patches.py' to accelerate training by 100x.")
        train_dataset = TorchGeoPotsdamDataset(root=args.data_dir, split="train", crop_size=512, compute_shannon=args.use_shannon, use_literature_split=True)
        val_dataset = TorchGeoPotsdamDataset(root=args.data_dir, split="val", crop_size=512, compute_shannon=args.use_shannon, use_literature_split=True)

    train_sampler = FewShotEpisodeSampler(train_dataset, classes=base_classes, k_shot=args.k_shot, q_queries=1)
    val_sampler = FewShotEpisodeSampler(val_dataset, classes=novel_classes, k_shot=args.k_shot, q_queries=1)

    print(f"\nStarting few-shot episodic training ({args.num_episodes} episodes)...", flush=True)

    # History tracking
    training_history = {
        "episodes": [],
        "train_loss": [],
        "val_checkpoints": []
    }

    for episode_idx in range(start_episode, args.num_episodes + 1):
        model.train()
        episode = train_sampler.sample_episode()

        target_class = episode["target_class"]
        is_veg = target_class in [3, 4]

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
            support_ndvi, query_ndvi, support_shannon, query_shannon,
            is_vegetation_class=is_veg
        )

        bce = F.binary_cross_entropy_with_logits(pred_logits, query_masks)
        dice = dice_loss(pred_logits, query_masks)
        total_loss = bce + dice

        total_loss.backward()

        # Gradient clipping for training stability
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)

        optimizer.step()
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else args.lr

        # Log every episode
        if episode_idx % 10 == 0 or episode_idx == 1:
            print(f"[Episode {episode_idx:04d}/{args.num_episodes}] Target Class: {target_class} | Loss: {total_loss.item():.4f} (BCE: {bce.item():.4f}, Dice: {dice.item():.4f}) | LR: {current_lr:.2e}", flush=True)

        training_history["episodes"].append(episode_idx)
        training_history["train_loss"].append(round(total_loss.item(), 4))

        # Periodic Validation on Novel Classes
        if episode_idx % args.val_freq == 0 or episode_idx == args.num_episodes:
            model.eval()
            # Reset validation RNG for reproducible, deterministic evaluations
            val_sampler.reset_rng(seed=42)

            class_ious = {c: [] for c in novel_classes}
            episodes_per_class = max(1, args.val_episodes // len(novel_classes))

            with torch.no_grad():
                for c in novel_classes:
                    is_veg_val = c in [3, 4]
                    for _ in range(episodes_per_class):
                        val_ep = val_sampler.sample_episode(target_class=c)
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
                            s_ndvi, q_ndvi, s_shn, q_shn,
                            is_vegetation_class=is_veg_val
                        )
                        iou = compute_iou(val_pred, q_masks)
                        class_ious[c].append(iou)

            mean_low_veg_iou = sum(class_ious[3]) / len(class_ious[3]) if class_ious[3] else 0.0
            mean_tree_iou = sum(class_ious[4]) / len(class_ious[4]) if class_ious[4] else 0.0
            mean_val_iou = (mean_low_veg_iou + mean_tree_iou) / 2.0

            print(f"\n========================================================", flush=True)
            print(f"--> [EVALUATION @ Episode {episode_idx}/{args.num_episodes}]", flush=True)
            print(f"    Low Vegetation (3) IoU: {mean_low_veg_iou:.4f}", flush=True)
            print(f"    Tree (4) IoU:           {mean_tree_iou:.4f}", flush=True)
            print(f"    Overall Novel mIoU:     {mean_val_iou:.4f}", flush=True)
            print(f"========================================================\n", flush=True)

            val_record = {
                "episode": episode_idx,
                "low_veg_iou": round(mean_low_veg_iou, 4),
                "tree_iou": round(mean_tree_iou, 4),
                "mean_val_iou": round(mean_val_iou, 4)
            }
            training_history["val_checkpoints"].append(val_record)

            # Save best checkpoint
            if mean_val_iou > best_val_iou:
                best_val_iou = mean_val_iou
                ckpt_path = os.path.join(args.output_dir, "best_virefsam_model.pth")
                torch.save({
                    "episode": episode_idx,
                    "model_state_dict": model.state_dict(),
                    "best_val_iou": best_val_iou,
                    "low_veg_iou": mean_low_veg_iou,
                    "tree_iou": mean_tree_iou,
                    "args": vars(args)
                }, ckpt_path)
                print(f"--> Saved new best checkpoint to {ckpt_path} (mIoU: {best_val_iou:.4f})\n", flush=True)

            # Save latest checkpoint
            latest_path = os.path.join(args.output_dir, "latest_virefsam_model.pth")
            torch.save({
                "episode": episode_idx,
                "model_state_dict": model.state_dict(),
                "val_iou": mean_val_iou,
                "args": vars(args)
            }, latest_path)

            # Update history JSON
            history_path = os.path.join(args.output_dir, "training_history.json")
            with open(history_path, "w") as f:
                json.dump(training_history, f, indent=2)

    print(f"Training complete! Best Novel Class mIoU: {best_val_iou:.4f}", flush=True)


if __name__ == "__main__":
    main()
