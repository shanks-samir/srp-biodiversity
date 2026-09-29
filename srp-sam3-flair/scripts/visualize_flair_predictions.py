"""Qualitative Visualization Script for Few-Shot FLAIR Predictions.

Generates publication-quality visual comparisons:
  [Support + Prompt Box] | [Query RGB] | [Query Modality/NDVI/Height] | [Similarity Map] | [Ground Truth] | [Predicted Mask (IoU)] | [Error Map]

Outputs PNG figures for presentations, defense slides, and reports.
"""

import os
import sys
import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Headless cluster backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.flair import FLAIRDataset, FLAIR_CLASSES, BASELINE_13, VEGETATION_CLASSES
from models.prompting import select_prompt_box
from scripts.evaluate_sam3 import SAM3Wrapper, compute_iou


def create_qualitative_figure(
    support_rgb: np.ndarray,
    support_box: Tuple[int, int, int, int],
    query_rgb: np.ndarray,
    query_modality: np.ndarray,
    modality_name: str,
    sim_map: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    class_name: str,
    iou: float,
    out_path: str,
):
    """Render a 6-panel qualitative evaluation figure."""
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.2))

    # 1. Support Image + Prompt Box
    axes[0].imshow(np.clip(support_rgb, 0, 1))
    bx, by, bw, bh = support_box
    rect = patches.Rectangle((bx, by), bw, bh, linewidth=2.5, edgecolor="#00FF00", facecolor="none")
    axes[0].add_patch(rect)
    axes[0].set_title(f"Support Exemplar\n({class_name})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Query RGB
    axes[1].imshow(np.clip(query_rgb, 0, 1))
    axes[1].set_title("Query RGB Scene", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. Query Modality (CIR / NDVI / Height)
    if query_modality.ndim == 3 and query_modality.shape[0] == 3:
        mod_show = np.transpose(query_modality, (1, 2, 0))
        axes[2].imshow(np.clip(mod_show, 0, 1))
    elif query_modality.ndim == 3 and query_modality.shape[-1] == 3:
        axes[2].imshow(np.clip(query_modality, 0, 1))
    else:
        # 1-channel heatmap (e.g. NDVI or Height)
        axes[2].imshow(query_modality.squeeze(), cmap="viridis")
    axes[2].set_title(f"Query {modality_name}", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # 4. Dense Cosine Similarity Map
    sim_plot = axes[3].imshow(sim_map, cmap="magma")
    fig.colorbar(sim_plot, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].set_title("Cosine Similarity Map\n(ViT Feature Affinity)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    # 5. Ground Truth Mask
    axes[4].imshow(np.clip(query_rgb * 0.4, 0, 1))
    gt_overlay = np.zeros((*gt_mask.shape, 4))
    gt_overlay[gt_mask == 1] = [0.0, 1.0, 0.0, 0.7]  # Green overlay
    axes[4].imshow(gt_overlay)
    axes[4].set_title(f"Ground Truth\n({class_name})", fontsize=11, fontweight="bold")
    axes[4].axis("off")

    # 6. Predicted Mask (with IoU)
    axes[5].imshow(np.clip(query_rgb * 0.4, 0, 1))
    pred_overlay = np.zeros((*pred_mask.shape, 4))
    # True Positives: Green
    pred_overlay[(pred_mask == 1) & (gt_mask == 1)] = [0.0, 1.0, 0.0, 0.75]
    # False Positives: Red
    pred_overlay[(pred_mask == 1) & (gt_mask == 0)] = [1.0, 0.0, 0.0, 0.75]
    # False Negatives: Cyan
    pred_overlay[(pred_mask == 0) & (gt_mask == 1)] = [0.0, 0.7, 1.0, 0.75]
    axes[5].imshow(pred_overlay)
    
    status_color = "#008800" if iou >= 0.30 else "#CC6600"
    axes[5].set_title(f"Prediction (IoU: {iou:.3f})\n[Green=TP, Red=FP, Cyan=FN]", 
                      fontsize=11, fontweight="bold", color=status_color)
    axes[5].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved qualitative figure: {out_path} (IoU: {iou:.4f})")


def generate_benchmark_chart(results_cir_path: str, results_eco_path: str, results_rgb_path: str, out_path: str):
    """Generate a grouped bar chart comparing all 12 classes across the 3 composites."""
    import json
    if not (os.path.exists(results_cir_path) and os.path.exists(results_eco_path) and os.path.exists(results_rgb_path)):
        print("Skipping benchmark summary chart: one or more JSON result files not found.")
        return

    with open(results_cir_path) as f:
        data_cir = json.load(f)["class_ious"]
    with open(results_eco_path) as f:
        data_eco = json.load(f)["class_ious"]
    with open(results_rgb_path) as f:
        data_rgb = json.load(f)["class_ious"]

    classes = list(data_rgb.keys())
    rgb_vals = [data_rgb.get(c, 0.0) * 100 for c in classes]
    cir_vals = [data_cir.get(c, 0.0) * 100 for c in classes]
    eco_vals = [data_eco.get(c, 0.0) * 100 for c in classes]

    x = np.arange(len(classes))
    width = 0.26

    fig, ax = plt.subplots(figsize=(16, 6.5))
    ax.bar(x - width, rgb_vals, width, label="RGB Control (True Color)", color="#4A90E2", edgecolor="black", alpha=0.9)
    ax.bar(x, cir_vals, width, label="CIR (NIR, Red, Green)", color="#E94E77", edgecolor="black", alpha=0.9)
    ax.bar(x + width, eco_vals, width, label="Pure Ecological Prior (NDVI, nDSM, H)", color="#2ECC71", edgecolor="black", alpha=0.9)

    ax.set_ylabel("1-Shot IoU (%)", fontsize=13, fontweight="bold")
    ax.set_title("FLAIR #1 Few-Shot Benchmark: Composite Ablation Across 12 Ecological Classes", fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in classes], rotation=35, ha="right", fontsize=11)
    ax.legend(fontsize=12, loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 70)

    # Highlight average lines
    ax.axhline(y=np.mean(rgb_vals), color="#2A70C2", linestyle=":", linewidth=1.5, label=f"RGB Mean: {np.mean(rgb_vals):.1f}%")
    ax.axhline(y=np.mean(cir_vals), color="#C92E57", linestyle=":", linewidth=1.5, label=f"CIR Mean: {np.mean(cir_vals):.1f}%")
    ax.axhline(y=np.mean(eco_vals), color="#0EAC51", linestyle=":", linewidth=1.5, label=f"Eco Mean: {np.mean(eco_vals):.1f}%")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved grouped benchmark chart: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Qualitative Figures for FLAIR Few-Shot Predictions")
    parser.add_argument("--root", type=str, default="./data", help="FLAIR dataset root")
    parser.add_argument("--composite", type=str, default="cir", choices=["rgb", "cir", "ndvi_dsm_h"], help="Composite to visualize")
    parser.add_argument("--sam_checkpoint", type=str, default=None, help="SAM ViT checkpoint path")
    parser.add_argument("--output_dir", type=str, default="./outputs/vis", help="Output directory for visual figures")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of qualitative figures per class")
    parser.add_argument("--classes", type=int, nargs="*", default=[6, 7, 8, 9, 10, 5], help="Class IDs to visualize (default: Coniferous, Deciduous, Brushwood, Vineyard, Herbaceous, Water)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    dataset = FLAIRDataset(root=args.root, composite=args.composite)
    cached_index = "./outputs/flair_class_index.json"
    if os.path.exists(cached_index):
        dataset._load_class_index(cached_index)
    else:
        dataset.build_class_index(save_to=cached_index)

    # Load model
    model = SAM3Wrapper(backend="auto", checkpoint=args.sam_checkpoint)

    target_classes = args.classes
    print(f"Generating qualitative figures for classes: {[FLAIR_CLASSES.get(c, c) for c in target_classes]}...")

    with torch.no_grad():
        for c in target_classes:
            c_name = FLAIR_CLASSES.get(c, str(c))
            candidates = dataset.class_to_indices.get(c, [])
            if len(candidates) < 2:
                print(f"Skipping {c_name}: insufficient candidates.")
                continue

            saved_for_class = 0
            attempts = 0
            while saved_for_class < args.num_samples and attempts < 30:
                attempts += 1
                q_idx = int(rng.choice(candidates))
                s_pool = [i for i in candidates if i != q_idx]
                if not s_pool:
                    continue
                s_idx = int(rng.choice(s_pool))

                s_sample = dataset[s_idx]
                q_sample = dataset[q_idx]

                s_mask_c = (s_sample["mask"].numpy() == c)
                q_mask_c = (q_sample["mask"].numpy() == c)

                # Require noticeable target area for informative qualitative figures
                if s_mask_c.sum() < 200 or q_mask_c.sum() < 200:
                    continue

                # Select prompt box
                s_ndvi = s_sample["ndvi"].squeeze(0).numpy() if "ndvi" in s_sample else None
                s_shn = s_sample["shannon"].squeeze(0).numpy() if "shannon" in s_sample else None
                box = select_prompt_box(s_mask_c, strategy="ecological", ndvi=s_ndvi, shannon=s_shn, rng=rng)
                if box is None:
                    continue

                s_img = s_sample["image"].numpy()
                q_img = q_sample["image"].numpy()

                # Run model prediction
                orig_size = (q_mask_c.shape[0], q_mask_c.shape[1])
                pred_bin = model.predict(
                    support_box_xywh=box.box_xywh,
                    orig_size=orig_size,
                    support_image=s_img,
                    query_image=q_img,
                    support_mask=s_mask_c,
                )

                iou, _, _ = compute_iou(pred_bin, q_mask_c)

                # Extract modalities for plotting
                s_rgb = s_sample["rgb"].permute(1, 2, 0).numpy() if hasattr(s_sample["rgb"], "permute") else s_sample["rgb"]
                q_rgb = q_sample["rgb"].permute(1, 2, 0).numpy() if hasattr(q_sample["rgb"], "permute") else q_sample["rgb"]

                if args.composite == "cir":
                    mod_img = q_img
                    mod_name = "CIR False-Color (NIR, R, G)"
                elif args.composite == "ndvi_dsm_h":
                    mod_img = q_sample["elevation"].numpy() if "elevation" in q_sample else q_img[1]
                    mod_name = "Canopy Height nDSM"
                else:
                    mod_img = q_sample["ndvi"].numpy() if "ndvi" in q_sample else q_img
                    mod_name = "NDVI Chlorophyll"

                # Retrieve similarity map for visualization
                if hasattr(model, "last_sim_map") and model.last_sim_map is not None:
                    sim_map = model.last_sim_map
                elif model.backend == "sam":
                    # Get the similarity map directly from predictor
                    s_hwc = (np.transpose(s_img, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
                    q_hwc = (np.transpose(q_img, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
                    model.predictor.set_image(s_hwc)
                    feat_s = model.predictor.get_image_embedding()
                    mask_s_t = torch.from_numpy(s_mask_c).float().to(model.device)[None, None, ...]
                    mask_down = F.interpolate(mask_s_t, size=feat_s.shape[-2:], mode="nearest")
                    target_feat = (feat_s * mask_down).sum(dim=(2, 3)) / (mask_down.sum() + 1e-6)
                    target_embed = F.normalize(target_feat, p=2, dim=-1)

                    model.predictor.set_image(q_hwc)
                    feat_q = model.predictor.get_image_embedding()
                    feat_q_norm = F.normalize(feat_q, p=2, dim=1)
                    sim = torch.einsum("bc,bchw->bhw", target_embed, feat_q_norm)
                    sim_map = F.interpolate(sim.unsqueeze(1), size=orig_size, mode="bilinear", align_corners=False).squeeze().cpu().numpy()
                else:
                    sim_map = np.zeros(orig_size, dtype=np.float32)

                out_filename = os.path.join(
                    args.output_dir,
                    f"qualitative_{c_name}_{args.composite}_sample{saved_for_class + 1}.png"
                )
                create_qualitative_figure(
                    support_rgb=s_rgb,
                    support_box=box.box_xywh,
                    query_rgb=q_rgb,
                    query_modality=mod_img,
                    modality_name=mod_name,
                    sim_map=sim_map,
                    gt_mask=q_mask_c,
                    pred_mask=pred_bin,
                    class_name=c_name,
                    iou=iou,
                    out_path=out_filename,
                )
                saved_for_class += 1

    # Also try generating the grouped benchmark bar chart if JSON outputs exist
    generate_benchmark_chart(
        results_cir_path="./outputs/results_cir_1shot.json",
        results_eco_path="./outputs/results_ndvi_dsm_h_1shot.json",
        results_rgb_path="./outputs/results_rgb_1shot.json",
        out_path=os.path.join(args.output_dir, "flair_composite_benchmark_comparison.png")
    )


if __name__ == "__main__":
    main()
