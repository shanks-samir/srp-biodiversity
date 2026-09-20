from typing import List, Dict, Tuple, Optional
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class FewShotEpisodeSampler:
    """Samples K-shot episodes (Support set S with K examples, Query set Q) for target classes."""

    def __init__(
        self,
        dataset: Dataset,
        classes: List[int],
        k_shot: int = 5,
        q_queries: int = 1,
        seed: int = 42
    ):
        """
        Args:
            dataset: PyTorch Dataset yielding dict with 'image', 'mask', 'ndvi', 'shannon'
            classes: List of class IDs to sample episodes for (e.g. [2, 3] for novel vegetation)
            k_shot: Number of support examples per class (e.g. 1 or 5)
            q_queries: Number of query examples per episode
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.classes = classes
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.rng = random.Random(seed)

        # Index dataset to find which samples contain which classes
        self.class_to_indices: Dict[int, List[int]] = {c: [] for c in classes}
        self._build_class_index()

    def _build_class_index(self):
        """Scan dataset to locate samples containing each target class."""
        print(f"Building class index for {len(self.dataset)} samples across classes {self.classes}...")
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            mask = sample["mask"]
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)

            present_classes = np.unique(mask_np)
            for c in self.classes:
                # Require at least 50 pixels of the class to be considered a valid support/query
                if c in present_classes and (mask_np == c).sum() >= 50:
                    self.class_to_indices[c].append(idx)

        for c in self.classes:
            print(f"Class {c}: {len(self.class_to_indices[c])} samples available.")

    def sample_episode(self, target_class: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Sample one K-shot evaluation episode.
        
        Returns:
            Dictionary containing:
                - 'support_images': (K, 3, H, W)
                - 'support_masks': (K, 1, H, W) binary foreground mask for target class
                - 'support_ndvi': (K, 1, H, W)
                - 'support_shannon': (K, 1, H, W)
                - 'query_images': (Q, 3, H, W)
                - 'query_masks': (Q, 1, H, W) binary foreground mask for target class
                - 'query_ndvi': (Q, 1, H, W)
                - 'query_shannon': (Q, 1, H, W)
                - 'target_class': int
        """
        if target_class is None:
            # Pick a class that has enough samples
            valid_classes = [c for c in self.classes if len(self.class_to_indices[c]) >= (self.k_shot + self.q_queries)]
            if not valid_classes:
                valid_classes = self.classes
            target_class = self.rng.choice(valid_classes)

        pool = self.class_to_indices[target_class]
        total_needed = self.k_shot + self.q_queries

        if len(pool) >= total_needed:
            sampled_indices = self.rng.sample(pool, total_needed)
        else:
            # Sample with replacement if dataset is very small
            sampled_indices = self.rng.choices(pool, k=total_needed)

        support_idxs = sampled_indices[:self.k_shot]
        query_idxs = sampled_indices[self.k_shot:]

        # Fetch and format support set
        support_imgs, support_masks, support_ndvis, support_shannons = [], [], [], []
        for idx in support_idxs:
            item = self.dataset[idx]
            support_imgs.append(item["image"])
            bin_mask = (item["mask"] == target_class).float().unsqueeze(0)
            support_masks.append(bin_mask)
            support_ndvis.append(item["ndvi"])
            support_shannons.append(item["shannon"])

        # Fetch and format query set
        query_imgs, query_masks, query_ndvis, query_shannons = [], [], [], []
        for idx in query_idxs:
            item = self.dataset[idx]
            query_imgs.append(item["image"])
            bin_mask = (item["mask"] == target_class).float().unsqueeze(0)
            query_masks.append(bin_mask)
            query_ndvis.append(item["ndvi"])
            query_shannons.append(item["shannon"])

        return {
            "support_images": torch.stack(support_imgs, dim=0),      # (K, 3, H, W)
            "support_masks": torch.stack(support_masks, dim=0),        # (K, 1, H, W)
            "support_ndvi": torch.stack(support_ndvis, dim=0),        # (K, 1, H, W)
            "support_shannon": torch.stack(support_shannons, dim=0),  # (K, 1, H, W)
            "query_images": torch.stack(query_imgs, dim=0),          # (Q, 3, H, W)
            "query_masks": torch.stack(query_masks, dim=0),            # (Q, 1, H, W)
            "query_ndvi": torch.stack(query_ndvis, dim=0),            # (Q, 1, H, W)
            "query_shannon": torch.stack(query_shannons, dim=0),      # (Q, 1, H, W)
            "target_class": target_class
        }
