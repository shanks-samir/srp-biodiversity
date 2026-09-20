import sys
import os
import torch
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ecological.indices import compute_ndvi_torch, compute_shannon_entropy_torch
from models.viref_sam.context_encoder import VisualContextualPromptEncoder
from models.viref_sam.ecological_adapter import EcologicalAdapter


def test_ecological_indices():
    print("[1/3] Testing Ecological Indices (NDVI & Shannon)...")
    B, H, W = 2, 64, 64
    nir = torch.rand(B, 1, H, W)
    red = torch.rand(B, 1, H, W)
    
    ndvi = compute_ndvi_torch(nir, red)
    assert ndvi.shape == (B, 1, H, W), f"Expected shape (2, 1, 64, 64), got {ndvi.shape}"
    assert ndvi.min() >= -1.0 and ndvi.max() <= 1.0, "NDVI outside [-1, 1]"
    
    ndvi_norm = (ndvi + 1.0) / 2.0
    shannon = compute_shannon_entropy_torch(ndvi_norm, kernel_size=7, num_bins=8)
    assert shannon.shape == (B, 1, H, W), f"Expected shape (2, 1, 64, 64), got {shannon.shape}"
    assert shannon.min() >= 0.0 and shannon.max() <= 1.0, "Shannon entropy outside [0, 1]"
    print("      --> Passed!")


def test_context_prompt_encoder():
    print("[2/3] Testing Visual Contextual Prompt Encoder...")
    K, C, H_feat, W_feat = 5, 256, 16, 16
    encoder = VisualContextualPromptEncoder(embed_dim=C, num_tokens=4)
    
    support_embeddings = torch.randn(K, C, H_feat, W_feat)
    support_masks = torch.randint(0, 2, (K, 1, H_feat, W_feat)).float()
    support_ndvi = torch.rand(K, 1, H_feat, W_feat) * 2 - 1
    
    tokens = encoder(support_embeddings, support_masks, support_ndvi)
    assert tokens.shape == (1, 4, C), f"Expected prompt tokens of shape (1, 4, 256), got {tokens.shape}"
    print("      --> Passed!")


def test_ecological_adapter():
    print("[3/3] Testing Dynamic Target Alignment Adapter...")
    B, C, H, W = 2, 256, 16, 16
    adapter = EcologicalAdapter(embed_dim=C, bottleneck_dim=64, use_ndvi=True, use_shannon=True)
    
    query_feats = torch.randn(B, C, H, W)
    query_ndvi = torch.randn(B, 1, 32, 32)
    query_shannon = torch.rand(B, 1, 32, 32)
    
    adapted = adapter(query_feats, query_ndvi, query_shannon)
    assert adapted.shape == (B, C, H, W), f"Expected adapted features of shape (2, 256, 16, 16), got {adapted.shape}"
    print("      --> Passed!")


if __name__ == "__main__":
    print("\n--- Running SRP Few-Shot Pipeline Verification Tests ---")
    test_ecological_indices()
    test_context_prompt_encoder()
    test_ecological_adapter()
    print("All architecture and index computation tests passed successfully!\n")
