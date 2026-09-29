"""Few-Shot Evaluation Script for SAM 3 on the FLAIR Dataset.

Evaluates the training-free SAM 3 few-shot pipeline using the spatial-concatenation
canvas approach (Tsai et al., arXiv:2604.05433).

Sweeps across candidate 3-channel composites:
  - rgb         : Control (Red, Green, Blue)
  - cir         : Color Infrared (NIR, Red, Green)
  - ndvi_dsm_h  : Derived ecological priors (NDVI, nDSM height, Shannon diversity H)
  - rgb_ndvi    : Red, Green, NDVI
  - rgb_dsm     : Red, Green, nDSM height

Outputs class-wise IoU, mIoU, and FB-IoU to a structured JSON file.
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.flair import (
    FLAIRDataset, FLAIR_CLASSES, BASELINE_13, VEGETATION_CLASSES, FOLDS,
    split_by_domain,
)
from models.canvas import (
    build_canvas, box_to_canvas, box_to_canvas_xyxy, extract_query_mask,
    CANVAS_SIZE, SPLIT_RATIO,
)
from models.prompting import select_prompt_box, score_candidate_masks, STRATEGIES


def compute_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Tuple[float, float, float]:
    """Compute (iou, intersection, union) for binary masks."""
    pred_b = pred_bin.astype(bool)
    gt_b = gt_bin.astype(bool)
    intersection = float(np.logical_and(pred_b, gt_b).sum())
    union = float(np.logical_or(pred_b, gt_b).sum())
    if union == 0.0:
        return 1.0, 0.0, 0.0
    return intersection / union, intersection, union


class SAM3Wrapper:
    """Unified wrapper supporting native sam3, HuggingFace transformers, segment_anything (SAM 1), or mock inference."""

    def __init__(self, backend: str = "auto", device: str = "cuda", checkpoint: Optional[str] = None):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.backend = backend
        self.model = None
        self.processor = None
        self.predictor = None

        if self.backend in ("auto", "native"):
            try:
                from sam3.model.sam3_image_processor import Sam3Processor
                from sam3 import build_sam3_image_model
                print(f"Loading native SAM 3 model on {self.device}...")
                self.model = build_sam3_image_model(checkpoint=checkpoint)
                self.model.to(self.device).eval()
                self.processor = Sam3Processor(self.model)
                self.backend = "native"
                print("Successfully loaded native SAM 3 backend.")
                return
            except Exception as e:
                if self.backend == "native":
                    raise RuntimeError(f"Failed to load native sam3 backend: {e}")

        if self.backend in ("auto", "hf"):
            try:
                from transformers import Sam3Model, Sam3Processor
                print(f"Loading HuggingFace SAM 3 model on {self.device}...")
                model_id = checkpoint or "facebook/sam3"
                self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()
                self.processor = Sam3Processor.from_pretrained(model_id)
                self.backend = "hf"
                print("Successfully loaded HuggingFace SAM 3 backend.")
                return
            except Exception as e:
                if self.backend == "hf":
                    raise RuntimeError(f"Failed to load HuggingFace transformers sam3 backend: {e}")

        if self.backend in ("auto", "sam", "sam1", "segment_anything"):
            try:
                from segment_anything import sam_model_registry, SamPredictor
                candidate_paths = [checkpoint] if checkpoint else []
                candidate_paths.extend([
                    "../checkpoints/sam_vit_b_01ec64.pth",
                    "checkpoints/sam_vit_b_01ec64.pth",
                    "/media/homes/shrestha/SRP/checkpoints/sam_vit_b_01ec64.pth",
                    os.path.expanduser("~/SRP/checkpoints/sam_vit_b_01ec64.pth"),
                    "../models/sam_vit_b_01ec64.pth",
                    "models/sam_vit_b_01ec64.pth",
                ])
                resolved_ckpt = None
                for p in candidate_paths:
                    if p and os.path.exists(p):
                        resolved_ckpt = p
                        break

                if resolved_ckpt:
                    print(f"Loading SAM (ViT-B) from {resolved_ckpt} on {self.device}...")
                    sam = sam_model_registry["vit_b"](checkpoint=resolved_ckpt)
                    sam.to(self.device).eval()
                    self.predictor = SamPredictor(sam)
                    self.backend = "sam"
                    print("Successfully loaded SAM (segment-anything) foundation model.")
                    return
                elif self.backend in ("sam", "sam1", "segment_anything"):
                    raise FileNotFoundError(f"SAM checkpoint not found in candidate paths: {candidate_paths}")
            except Exception as e:
                if self.backend in ("sam", "sam1", "segment_anything"):
                    raise RuntimeError(f"Failed to load segment_anything backend: {e}")

        print("Notice: Neither native sam3 nor HuggingFace SAM 3 nor SAM checkpoint is available in this environment.")
        print("Falling back to mock inference mode (for pipeline verification without GPU weights).")
        self.backend = "mock"

    def predict(
        self,
        canvas: Optional[np.ndarray] = None,
        support_box_xywh: Optional[Tuple[int, int, int, int]] = None,
        layout=None,
        orig_size: Tuple[int, int] = (512, 512),
        support_image: Optional[np.ndarray] = None,
        query_image: Optional[np.ndarray] = None,
        support_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run few-shot prediction on query using exemplar guidance."""
        if self.backend == "mock":
            # Deterministic mock prediction: returns a synthetic region around target location
            x, y, w, h = support_box_xywh if support_box_xywh else (50, 50, 50, 50)
            mock_mask = np.zeros(orig_size, dtype=np.uint8)
            mock_mask[max(0, y):min(orig_size[0], y + h), max(0, x):min(orig_size[1], x + w)] = 1
            return mock_mask

        if self.backend == "native":
            norm_box = box_to_canvas(support_box_xywh, layout.support, orig_size, CANVAS_SIZE)
            with torch.no_grad():
                state = self.processor.set_image(canvas)
                output = self.processor.add_geometric_prompt(state, box=norm_box, label=True)
                canvas_mask = output.get("masks", output.get("pred_masks"))
                if isinstance(canvas_mask, torch.Tensor):
                    canvas_mask = canvas_mask.squeeze().cpu().numpy()
                if canvas_mask.ndim == 3:
                    canvas_mask = canvas_mask[0]
                pred_bin = canvas_mask > 0.5
            return extract_query_mask(pred_bin, layout, out_size=orig_size)

        if self.backend == "hf":
            xyxy_box = box_to_canvas_xyxy(support_box_xywh, layout.support, orig_size, CANVAS_SIZE)
            with torch.no_grad():
                inputs = self.processor(
                    images=canvas,
                    input_boxes=[[[*xyxy_box]]],
                    input_boxes_labels=[[1]],
                    return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs)
                pred_masks = outputs.pred_masks.squeeze().cpu().numpy()
                if pred_masks.ndim == 3:
                    pred_masks = pred_masks[0]
                pred_bin = pred_masks > 0.0
            return extract_query_mask(pred_bin, layout, out_size=orig_size)

        if self.backend == "sam":
            import torch.nn.functional as F

            def to_hwc_uint8(img):
                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu().numpy()
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = np.transpose(img, (1, 2, 0))
                if img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                if img.max() <= 1.0:
                    img = (img * 255.0).clip(0, 255).astype(np.uint8)
                return img.astype(np.uint8)

            assert support_image is not None and query_image is not None, \
                "SAM backend requires support_image and query_image for exemplar feature matching."

            s_hwc = to_hwc_uint8(support_image)
            q_hwc = to_hwc_uint8(query_image)
            H, W = q_hwc.shape[:2]

            with torch.no_grad():
                # 1. Encode support image with frozen SAM backbone
                self.predictor.set_image(s_hwc)
                feat_s = self.predictor.get_image_embedding()  # (1, 256, 64, 64)

                # 2. Extract exemplar prototype from target region
                if support_mask is not None:
                    mask_s = torch.from_numpy(support_mask).float().to(self.device)[None, None, ...]
                else:
                    x, y, w, h = support_box_xywh
                    mask_s = torch.zeros((1, 1, s_hwc.shape[0], s_hwc.shape[1]), device=self.device)
                    mask_s[0, 0, max(0, y):min(s_hwc.shape[0], y+h), max(0, x):min(s_hwc.shape[1], x+w)] = 1.0

                mask_s_down = F.interpolate(mask_s, size=feat_s.shape[-2:], mode="nearest")
                if mask_s_down.sum() == 0:
                    mask_s_down = torch.ones_like(feat_s[:, :1, :, :])

                target_feat = (feat_s * mask_s_down).sum(dim=(2, 3)) / (mask_s_down.sum() + 1e-6)
                target_embed = F.normalize(target_feat, p=2, dim=-1)  # (1, 256)

                # 3. Encode query image with frozen SAM backbone
                self.predictor.set_image(q_hwc)
                feat_q = self.predictor.get_image_embedding()  # (1, 256, 64, 64)
                feat_q_norm = F.normalize(feat_q, p=2, dim=1)  # (1, 256, 64, 64)

                # 4. Compute cosine similarity map
                sim = torch.einsum("bc,bchw->bhw", target_embed, feat_q_norm)
                sim_map = F.interpolate(sim.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze()

                # 5. Extract peak positive and negative prompt points
                max_val, max_idx = torch.max(sim_map.view(-1), dim=0)
                y_pos = int(max_idx.item() // W)
                x_pos = int(max_idx.item() % W)

                min_val, min_idx = torch.min(sim_map.view(-1), dim=0)
                y_neg = int(min_idx.item() // W)
                x_neg = int(min_idx.item() % W)

                point_coords = np.array([[x_pos, y_pos], [x_neg, y_neg]], dtype=np.float32)
                point_labels = np.array([1, 0], dtype=np.int32)

                # Box prompt around high-similarity region
                thresh = max(0.2, 0.7 * max_val.item())
                high_sim = (sim_map >= thresh).cpu().numpy()
                box_prompt = None
                if high_sim.sum() > 20:
                    ys, xs = np.where(high_sim)
                    box_prompt = np.array([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])

                # 6. Predict candidate masks with SAM decoder
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_prompt,
                    multimask_output=True,
                )

                # 7. Select best candidate mask by feature similarity against prototype
                best_sim = -float("inf")
                best_idx = int(np.argmax(scores))
                for idx, mask_cand in enumerate(masks):
                    cand_t = torch.from_numpy(mask_cand).float().to(self.device)[None, None, ...]
                    cand_down = F.interpolate(cand_t, size=feat_q.shape[-2:], mode="nearest")
                    if cand_down.sum() > 0:
                        cand_feat = (feat_q * cand_down).sum(dim=(2, 3)) / (cand_down.sum() + 1e-6)
                        cand_embed = F.normalize(cand_feat, p=2, dim=-1)
                        sim_val = torch.sum(cand_embed * target_embed).item()
                        if sim_val > best_sim:
                            best_sim = sim_val
                            best_idx = idx

                pred_bin = masks[best_idx].astype(np.uint8)
                return pred_bin

        raise ValueError(f"Unknown backend: {self.backend}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM / SAM 3 on FLAIR with ecological composites")
    parser.add_argument("--root", type=str, default="./data", help="FLAIR dataset root directory")
    parser.add_argument("--class_index", type=str, default=None, help="Path to precomputed class index JSON")
    parser.add_argument("--composite", type=str, default="cir", choices=["rgb", "cir", "ndvi_dsm_h", "rgb_ndvi", "rgb_dsm"], help="Input composite")
    parser.add_argument("--shannon_source", type=str, default="elevation", choices=["elevation", "ndvi"], help="Source for Shannon entropy")
    parser.add_argument("--k_shot", type=int, default=1, choices=[1, 5], help="Number of support shots")
    parser.add_argument("--strategy", type=str, default="ecological", choices=list(STRATEGIES), help="Prompt selection strategy")
    parser.add_argument("--holdout_domains", type=str, nargs="*", default=None, help="Optional domain IDs to hold out for evaluation")
    parser.add_argument("--fold", type=int, default=None, choices=[0, 1, 2, 3], help="Optional PASCAL-style fold index")
    parser.add_argument("--num_episodes", type=int, default=30, help="Episodes per class")
    parser.add_argument("--model_backend", type=str, default="auto", choices=["auto", "native", "hf", "sam", "sam1", "segment_anything", "mock"], help="Model backend")
    parser.add_argument("--sam_checkpoint", type=str, default=None, help="Path to SAM checkpoint (e.g. sam_vit_b_01ec64.pth)")
    parser.add_argument("--sam3_checkpoint", type=str, default=None, help="Path to local SAM 3 checkpoint or HF model id")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    print(f"\n========================================================")
    print(f"Few-Shot FLAIR Evaluation (Foundation Model Spatial Canvas)")
    print(f"  Composite     : {args.composite}")
    print(f"  Shannon source: {args.shannon_source}")
    print(f"  K-shot        : {args.k_shot}")
    print(f"  Strategy      : {args.strategy}")
    print(f"  Backend       : {args.model_backend}")
    print(f"========================================================\n")

    # Load dataset
    dataset = FLAIRDataset(
        root=args.root,
        composite=args.composite,
        shannon_source=args.shannon_source,
        class_index_file=args.class_index,
    )
    print(f"Loaded FLAIR dataset with {len(dataset)} total patches across {len(set(dataset.domains))} domains.")

    # Auto-load or build class index if not provided
    if not dataset.class_to_indices:
        default_index = os.path.join(args.output_dir, "flair_class_index.json")
        if os.path.exists(default_index):
            print(f"Loading cached class index from {default_index}...")
            dataset._load_class_index(default_index)
        else:
            print("Class index not found. Indexing patches by class (runs once and caches to disk)...")
            dataset.build_class_index(save_to=default_index)

    # Determine evaluation classes
    if args.fold is not None:
        eval_classes = FOLDS[args.fold]
        print(f"Evaluating Fold {args.fold}: classes {eval_classes}")
    else:
        # Default to the 13 baseline classes (or available classes)
        eval_classes = [c for c in BASELINE_13 if c in dataset.class_to_indices and len(dataset.class_to_indices[c]) >= 2]
        if not eval_classes:
            eval_classes = [c for c in dataset.class_to_indices if len(dataset.class_to_indices[c]) >= 2]

    # Split train/eval if holdout domains provided
    if args.holdout_domains:
        train_idxs, val_idxs = split_by_domain(dataset, holdout=args.holdout_domains)
        print(f"Domain hold-out: {len(train_idxs)} train patches, {len(val_idxs)} hold-out patches.")
        support_pool = set(train_idxs)
        query_pool = set(val_idxs)
    else:
        support_pool = set(range(len(dataset)))
        query_pool = set(range(len(dataset)))

    # Initialize SAM 3 wrapper
    model = SAM3Wrapper(backend=args.model_backend, checkpoint=args.sam3_checkpoint)

    class_ious = {c: [] for c in eval_classes}
    total_fg_inter, total_fg_union = 0.0, 0.0
    total_bg_inter, total_bg_union = 0.0, 0.0

    for c in eval_classes:
        c_name = FLAIR_CLASSES.get(c, str(c))
        c_patches = dataset.class_to_indices.get(c, [])
        valid_support = [idx for idx in c_patches if idx in support_pool]
        valid_query = [idx for idx in c_patches if idx in query_pool]

        if not valid_support or not valid_query:
            print(f"Skipping class {c:2d} ({c_name}): insufficient patches.")
            continue

        print(f"\nEvaluating Class {c:2d} ({c_name}): {len(valid_support)} support, {len(valid_query)} query candidates...")
        episodes_run = 0

        pbar = tqdm(total=args.num_episodes, desc=f"Class {c_name}")
        attempts = 0
        max_attempts = args.num_episodes * 4

        while episodes_run < args.num_episodes and attempts < max_attempts:
            attempts += 1
            q_idx = int(rng.choice(valid_query))
            s_candidates = [idx for idx in valid_support if idx != q_idx]
            if not s_candidates:
                continue

            s_idxs = rng.choice(s_candidates, size=min(args.k_shot, len(s_candidates)), replace=False)
            # Pick the primary support patch
            s_idx = int(s_idxs[0])

            s_sample = dataset[s_idx]
            q_sample = dataset[q_idx]

            s_mask_c = (s_sample["mask"].numpy() == c)
            q_mask_c = (q_sample["mask"].numpy() == c)

            if s_mask_c.sum() < 50 or q_mask_c.sum() < 50:
                continue

            # Select prompt box on support patch
            s_ndvi = s_sample["ndvi"].squeeze(0).numpy() if "ndvi" in s_sample else None
            s_shn = s_sample["shannon"].squeeze(0).numpy() if "shannon" in s_sample else None

            box = select_prompt_box(
                s_mask_c,
                strategy=args.strategy,
                ndvi=s_ndvi,
                shannon=s_shn,
                rng=rng,
            )
            if box is None:
                continue

            # Build spatial-concatenation canvas
            canvas, layout, _ = build_canvas(
                support_image=s_sample["image"].numpy(),
                query_image=q_sample["image"].numpy(),
                support_box_xywh=box.box_xywh,
            )

            # Predict query mask
            orig_size = (q_mask_c.shape[0], q_mask_c.shape[1])
            pred_query_bin = model.predict(
                canvas=canvas,
                support_box_xywh=box.box_xywh,
                layout=layout,
                orig_size=orig_size,
                support_image=s_sample["image"].numpy(),
                query_image=q_sample["image"].numpy(),
                support_mask=s_mask_c,
            )

            # Compute IoU
            iou, inter, union = compute_iou(pred_query_bin, q_mask_c)
            class_ious[c].append(iou)

            # Update foreground-background IoU tallies
            total_fg_inter += inter
            total_fg_union += union
            bg_pred = (pred_query_bin == 0)
            bg_gt = (q_mask_c == 0)
            total_bg_inter += float(np.logical_and(bg_pred, bg_gt).sum())
            total_bg_union += float(np.logical_or(bg_pred, bg_gt).sum())

            episodes_run += 1
            pbar.update(1)

        pbar.close()

    # Compute aggregate summary metrics
    valid_class_ious = {c: float(np.mean(ious)) for c, ious in class_ious.items() if ious}
    miou = float(np.mean(list(valid_class_ious.values()))) if valid_class_ious else 0.0
    iou_std = float(np.std(list(valid_class_ious.values()))) if valid_class_ious else 0.0

    fb_fg = (total_fg_inter / total_fg_union) if total_fg_union > 0 else 0.0
    fb_bg = (total_bg_inter / total_bg_union) if total_bg_union > 0 else 0.0
    fb_iou = (fb_fg + fb_bg) / 2.0

    print("\n================ Results Summary ================")
    print(f"Composite     : {args.composite}")
    print(f"K-shot        : {args.k_shot}")
    print(f"mIoU          : {miou:.4f} (std: {iou_std:.4f})")
    print(f"FB-IoU        : {fb_iou:.4f} (FG: {fb_fg:.4f}, BG: {fb_bg:.4f})")
    print("-------------------------------------------------")
    for c, score in valid_class_ious.items():
        c_name = FLAIR_CLASSES.get(c, str(c))
        print(f"  Class {c:2d} ({c_name:<22}): {score:.4f} (n={len(class_ious[c])})")
    print("=================================================\n")

    results_payload = {
        "composite": args.composite,
        "shannon_source": args.shannon_source,
        "k_shot": args.k_shot,
        "strategy": args.strategy,
        "miou": round(miou, 4),
        "std": round(iou_std, 4),
        "fb_iou": round(fb_iou, 4),
        "class_ious": {FLAIR_CLASSES.get(c, str(c)): round(s, 4) for c, s in valid_class_ious.items()},
    }

    out_file = os.path.join(args.output_dir, f"results_{args.composite}_{args.k_shot}shot.json")
    with open(out_file, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Saved evaluation results to {out_file}")


if __name__ == "__main__":
    main()
