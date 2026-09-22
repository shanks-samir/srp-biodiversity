from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_encoder import VisualContextualPromptEncoder
from .ecological_adapter import EcologicalAdapter


class ViRefSAM(nn.Module):
    """Visual Reference-Guided Segment Anything Model with Ecological Channel Integration."""

    def __init__(
        self,
        sam_model,
        num_prompt_tokens: int = 4,
        adapter_bottleneck: int = 64,
        use_ndvi: bool = True,
        use_shannon: bool = True,
        freeze_sam: bool = True
    ):
        """
        Args:
            sam_model: Pre-trained SAM instance (e.g. from segment_anything.sam_model_registry)
            num_prompt_tokens: Number of summary tokens synthesized from support reference
            adapter_bottleneck: Hidden bottleneck channel dimension
            use_ndvi: Whether to fuse NDVI into feature space
            use_shannon: Whether to fuse Shannon Diversity
            freeze_sam: If True, freezes core SAM encoder and mask decoder weights
        """
        super().__init__()
        self.sam = sam_model
        embed_dim = self.sam.prompt_encoder.embed_dim

        # Freeze SAM parameters if requested
        if freeze_sam:
            for param in self.sam.parameters():
                param.requires_grad = False

        # Visual Contextual Prompt Encoder for synthesizing prompts from support set
        self.context_prompt_encoder = VisualContextualPromptEncoder(
            embed_dim=embed_dim,
            num_tokens=num_prompt_tokens
        )

        # Dynamic Target Alignment Adapter with Multimodal Fusion
        self.ecological_adapter = EcologicalAdapter(
            embed_dim=embed_dim,
            bottleneck_dim=adapter_bottleneck,
            use_ndvi=use_ndvi,
            use_shannon=use_shannon
        )

    def extract_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image embeddings using SAM's frozen vision encoder."""
        # Expected input shape: (B, 3, H, W) normalized to [0, 1] or preprocessed
        # SAM expects 1024x1024; preprocess if necessary
        if images.shape[-1] != 1024 or images.shape[-2] != 1024:
            images = F.interpolate(images, size=(1024, 1024), mode="bilinear", align_corners=False)
        
        # Scale to [0, 255] then apply SAM normalization
        input_images = images * 255.0
        # SAM internal normalization: mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375]
        pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=images.device).view(1, 3, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375], device=images.device).view(1, 3, 1, 1)
        normalized = (input_images - pixel_mean) / pixel_std

        with torch.set_grad_enabled(not self.training or any(p.requires_grad for p in self.sam.image_encoder.parameters())):
            embeddings = self.sam.image_encoder(normalized)
        return embeddings

    def forward_few_shot(
        self,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
        query_images: torch.Tensor,
        support_ndvi: Optional[torch.Tensor] = None,
        query_ndvi: Optional[torch.Tensor] = None,
        support_shannon: Optional[torch.Tensor] = None,
        query_shannon: Optional[torch.Tensor] = None,
        is_vegetation_class: bool = True
    ) -> torch.Tensor:
        """
        Execute full few-shot segmentation pipeline:
          1. Extract support and query embeddings via SAM's image encoder.
          2. Synthesize visual reference prompt tokens from (support_embeddings, support_masks, support_ndvi).
          3. Adapt query embeddings using the Ecological Adapter (query_embeddings + NDVI + Shannon).
          4. Feed adapted query embeddings + prompt tokens into SAM's mask decoder.
        
        Args:
            support_images: (K, 3, H, W)
            support_masks: (K, 1, H, W) binary mask for target class
            query_images: (Q, 3, H, W)
            support_ndvi: (K, 1, H, W)
            query_ndvi: (Q, 1, H, W)
            support_shannon: (K, 1, H, W)
            query_shannon: (Q, 1, H, W)
            is_vegetation_class: Whether target class is vegetation (controls NDVI weighting)
            
        Returns:
            predicted_masks: (Q, 1, H, W) logit map for query segmentation
        """
        Q, _, H_q, W_q = query_images.shape

        # 1. Feature extraction
        # Extract support embeddings in single-image chunks under no_grad to drop peak VRAM from ~8GB to ~1.2GB
        support_feats_list = []
        with torch.no_grad():
            for k in range(support_images.shape[0]):
                s_feat = self.extract_image_embeddings(support_images[k:k+1])
                support_feats_list.append(s_feat)
        support_feats = torch.cat(support_feats_list, dim=0)  # (K, 256, 64, 64)

        query_feats = self.extract_image_embeddings(query_images)      # (Q, 256, 64, 64)

        # 2. Synthesize reference prompt tokens from support set
        context_tokens = self.context_prompt_encoder(
            support_feats, support_masks, support_ndvi, is_vegetation_class=is_vegetation_class
        )  # (1, num_tokens, 256)
        # Expand tokens for all queries in batch
        prompt_tokens = context_tokens.expand(Q, -1, -1)  # (Q, num_tokens, 256)

        # 3. Dynamic Target Alignment / Ecological Adaptation on query features
        adapted_query_feats = self.ecological_adapter(
            query_feats, query_ndvi, query_shannon
        )  # (Q, 256, 64, 64)

        # 4. SAM Mask Decoder
        # Get SAM default empty sparse and dense prompt embeddings
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None
        )
        # Concatenate our visual contextual tokens with sparse embeddings
        combined_sparse_embeddings = torch.cat([sparse_embeddings.expand(Q, -1, -1), prompt_tokens], dim=1)

        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=adapted_query_feats,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=combined_sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings.expand(Q, -1, -1, -1),
            multimask_output=False
        )

        # Upscale low-resolution masks (256x256) to target query size (H_q, W_q)
        high_res_masks = F.interpolate(
            low_res_masks,
            size=(H_q, W_q),
            mode="bilinear",
            align_corners=False
        )
        return high_res_masks
