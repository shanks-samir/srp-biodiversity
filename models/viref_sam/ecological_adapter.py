import torch
import torch.nn as nn
import torch.nn.functional as F


class EcologicalAdapter(nn.Module):
    """Dynamic Target Alignment Adapter with Multimodal Fusion for NDVI and Shannon Diversity."""

    def __init__(
        self,
        embed_dim: int = 256,
        bottleneck_dim: int = 64,
        use_ndvi: bool = True,
        use_shannon: bool = True
    ):
        """
        Args:
            embed_dim: Dimension of SAM feature embeddings (256)
            bottleneck_dim: Compression dimension for lightweight adapter
            use_ndvi: Whether to inject NDVI
            use_shannon: Whether to inject Shannon Diversity
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.use_ndvi = use_ndvi
        self.use_shannon = use_shannon

        extra_channels = int(use_ndvi) + int(use_shannon)
        self.extra_channels = extra_channels
        self.has_extra = extra_channels > 0

        if self.has_extra:
            # Convolutional stem to project ecological maps into bottleneck space
            # Using GroupNorm instead of BatchNorm2d for stability with batch size 1
            num_groups = min(8, bottleneck_dim)
            self.eco_stem = nn.Sequential(
                nn.Conv2d(extra_channels, bottleneck_dim, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=num_groups, num_channels=bottleneck_dim),
                nn.GELU(),
                nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=num_groups, num_channels=bottleneck_dim),
                nn.GELU()
            )
        else:
            self.eco_stem = None

        # Residual Adapter path
        self.down_proj = nn.Conv2d(embed_dim, bottleneck_dim, kernel_size=1)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1, groups=bottleneck_dim)
        self.up_proj = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1)

        # Scale factor initialized small to start as near-identity mapping
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        query_embeddings: torch.Tensor,
        query_ndvi: torch.Tensor = None,
        query_shannon: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            query_embeddings: (B, C, H, W) SAM encoder query features (e.g. B, 256, 64, 64)
            query_ndvi: (B, 1, H_img, W_img) optional NDVI map
            query_shannon: (B, 1, H_img, W_img) optional Shannon diversity map
            
        Returns:
            adapted_embeddings: (B, C, H, W) ecologically adapted embeddings
        """
        B, C, H, W = query_embeddings.shape

        # Residual down-projection
        res = self.down_proj(query_embeddings)  # (B, bottleneck_dim, H, W)
        res = self.conv(res)
        res = self.act(res)

        # Inject ecological priors if available
        if self.has_extra and (query_ndvi is not None or query_shannon is not None):
            eco_inputs = []
            if self.use_ndvi:
                if query_ndvi is not None:
                    ndvi_down = F.interpolate(query_ndvi, size=(H, W), mode="bilinear", align_corners=False)
                else:
                    ndvi_down = torch.zeros((B, 1, H, W), device=query_embeddings.device, dtype=query_embeddings.dtype)
                eco_inputs.append(ndvi_down)

            if self.use_shannon:
                if query_shannon is not None:
                    shannon_down = F.interpolate(query_shannon, size=(H, W), mode="bilinear", align_corners=False)
                else:
                    shannon_down = torch.zeros((B, 1, H, W), device=query_embeddings.device, dtype=query_embeddings.dtype)
                eco_inputs.append(shannon_down)

            if len(eco_inputs) > 0:
                eco_cat = torch.cat(eco_inputs, dim=1)  # (B, extra_channels, H, W)
                eco_feat = self.eco_stem(eco_cat)        # (B, bottleneck_dim, H, W)
                res = res + eco_feat

        # Up-projection to original feature dimension and residual addition
        adapted = query_embeddings + self.gamma * self.up_proj(res)
        return adapted
