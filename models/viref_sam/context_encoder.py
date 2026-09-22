import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualContextualPromptEncoder(nn.Module):
    """Extracts contextual prompt tokens from K support image-mask pairs to guide SAM's mask decoder."""

    def __init__(self, embed_dim: int = 256, num_tokens: int = 4):
        """
        Args:
            embed_dim: Feature embedding dimension of SAM (256 for standard SAM mask decoder)
            num_tokens: Number of summary prompt tokens to synthesize from support examples
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # Project combined (foreground + background) prototype features into learned prompt tokens
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim * num_tokens)
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        support_embeddings: torch.Tensor,
        support_masks: torch.Tensor,
        support_ndvi: torch.Tensor = None,
        is_vegetation_class: bool = True
    ) -> torch.Tensor:
        """
        Args:
            support_embeddings: (K, C, H_feat, W_feat) from SAM image encoder (e.g. 256 x 64 x 64)
            support_masks: (K, 1, H_img, W_img) binary foreground mask
            support_ndvi: (K, 1, H_img, W_img) optional NDVI map
            is_vegetation_class: Whether target class is vegetation (apply NDVI weighting)
            
        Returns:
            prompt_tokens: (1, num_tokens, embed_dim) contextual tokens to pass to SAM prompt encoder
        """
        K, C, H_f, W_f = support_embeddings.shape

        # Downsample support mask to match feature map resolution
        mask_down = F.interpolate(support_masks.float(), size=(H_f, W_f), mode="bilinear", align_corners=False)
        mask_down = (mask_down > 0.5).float()
        bg_mask_down = 1.0 - mask_down

        # Weight foreground mask by NDVI ONLY for vegetation classes (prevents corrupting non-veg prototypes)
        fg_weights = mask_down
        if support_ndvi is not None and is_vegetation_class:
            ndvi_down = F.interpolate(support_ndvi.float(), size=(H_f, W_f), mode="bilinear", align_corners=False)
            # Re-scale NDVI from [-1, 1] to [0.5, 1.5] as importance multiplier
            weight = torch.clamp((ndvi_down + 1.0) / 2.0 + 0.5, 0.1, 2.0)
            fg_weights = fg_weights * weight

        # 1. Foreground prototype via Masked Average Pooling
        fg_sum = fg_weights.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        fg_feats = (support_embeddings * fg_weights).sum(dim=(2, 3), keepdim=True) / fg_sum  # (K, C, 1, 1)
        fg_proto = fg_feats.mean(dim=0).view(1, C)  # (1, C)

        # 2. Background prototype via Masked Average Pooling
        bg_sum = bg_mask_down.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        bg_feats = (support_embeddings * bg_mask_down).sum(dim=(2, 3), keepdim=True) / bg_sum  # (K, C, 1, 1)
        bg_proto = bg_feats.mean(dim=0).view(1, C)  # (1, C)

        # Concatenate foreground and background prototypes for discriminative context
        combined = torch.cat([fg_proto, bg_proto], dim=-1)  # (1, 2*C)

        # Project prototype into contextual prompt tokens
        tokens = self.proj(combined).view(1, self.num_tokens, self.embed_dim)  # (1, num_tokens, embed_dim)
        tokens = self.norm(tokens)
        return tokens
