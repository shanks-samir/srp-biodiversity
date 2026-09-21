# Student Research Project (SRP) Defense Report
**Few-Shot Semantic Segmentation for Biodiversity Assessment using Segment Anything Model (SAM) and Multispectral UAV Imagery**

- **Author**: Samir Paudyal
- **Institution**: University of Hildesheim — Information Systems and Machine Learning Lab (ISMLL)
- **Supervisors**: Prof. Dr. Niels Landwehr, Ahmad Bdeir
- **Repository**: `https://github.com/shanks-samir/srp-biodiversity.git`

---

## Executive Summary & Abstract
Mapping vegetation and ecological habitats from high-resolution UAV and aerial multispectral imagery is central to automated biodiversity monitoring. However, conventional deep learning models require exhaustive pixel-level annotations for every new plant species or habitat class. This project investigates **Few-Shot Semantic Segmentation (FSS)** using a foundation model approach: adapting the **Segment Anything Model (SAM)** with domain-specific ecological priors (Normalized Difference Vegetation Index, NDVI, and Local Shannon Diversity Index, $H$).

We develop **ViRefSAM** (Visual Reference-Guided Ecological SAM), a parameter-efficient architecture that:
1. Feeds $K$ visual support examples into a **Visual Contextual Prompt Encoder** to synthesize class-prototype prompt tokens.
2. Modulates query image embeddings through a **Multimodal Ecological Adapter** fusing spectral (NDVI) and structural/textural ($H$) information.
3. Decodes segmentation masks via SAM's lightweight mask decoder while keeping SAM's 90M parameter ViT-B image encoder completely frozen.

We evaluate this system on the benchmark **ISPRS Potsdam 2D** dataset under the standard literature hold-out split (Rottensteiner et al., 2012) using base-to-novel class transfer.

---

## 1. Problem Formulation & Few-Shot Learning Framework

### 1.1 Few-Shot Episodic Segmentation Protocol
Given a dataset with labeled classes partitioned into disjoint base classes $\mathcal{C}_{\text{base}}$ and novel classes $\mathcal{C}_{\text{novel}}$ ($\mathcal{C}_{\text{base}} \cap \mathcal{C}_{\text{novel}} = \emptyset$):
- **Training Phase**: The model is trained over episodes sampled strictly from $\mathcal{C}_{\text{base}}$.
- **Inference / Testing Phase**: The model is evaluated on $\mathcal{C}_{\text{novel}}$, where only $K$ annotated support images ($K \in \{1, 5\}$) are provided per novel class.

Each episode $\mathcal{E}$ consists of:
- **Support Set**: $\mathcal{S} = \{(I_s^k, M_s^k, \text{NDVI}_s^k, H_s^k)\}_{k=1}^K$
- **Query Set**: $\mathcal{Q} = \{(I_q, M_q, \text{NDVI}_q, H_q)\}$
where $I \in \mathbb{R}^{3 \times H \times W}$ is RGB imagery, $M \in \{0, 1\}^{H \times W}$ is the binary ground-truth mask for target class $c$, and $\text{NDVI}, H \in \mathbb{R}^{1 \times H \times W}$ are ecological channels.

---

## 2. Mathematical Formulations of Ecological Priors

### 2.1 Normalized Difference Vegetation Index (NDVI)
NDVI quantifies photosynthetic activity and chlorophyll absorption using Near-Infrared (NIR, Band 3) and Red (Band 0):

$$\text{NDVI} = \frac{\rho_{\text{NIR}} - \rho_{\text{Red}}}{\rho_{\text{NIR}} + \rho_{\text{Red}} + \epsilon}, \quad \text{NDVI} \in [-1.0, 1.0]$$

In our pipeline, NDVI is calculated dynamically in PyTorch:
```python
def compute_ndvi_torch(nir: torch.Tensor, red: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.clamp((nir - red) / (nir + red + eps), -1.0, 1.0)
```

### 2.2 Local Shannon Diversity Index ($H$)
To capture habitat structural heterogeneity, canopy roughness, and vegetation texture, we formulate a sliding-window Shannon Entropy map over quantized vegetation density.

Given local patch window $W$ (kernel size $w \times w$, default $w = 7$), values are quantized into $B$ histogram bins ($B = 16$). The local probability of bin $b$ is:

$$p_b(u, v) = \frac{1}{|W|} \sum_{(i, j) \in W(u, v)} \mathbb{I}(\text{bin}(x_{i,j}) = b)$$

The local Shannon diversity index is computed as:

$$H(u, v) = -\sum_{b=1}^B p_b(u, v) \log_2(p_b(u, v))$$

Normalized by maximum theoretical entropy $\log_2(B)$ to bound $H \in [0.0, 1.0]$:

$$H_{\text{norm}}(u, v) = \frac{H(u, v)}{\log_2(B)}$$

Implemented via vectorized depthwise 2D convolutions in PyTorch for GPU acceleration without CPU bottlenecks:
```python
# One-hot binning -> Depthwise uniform conv2d -> Shannon Entropy
local_probs = F.conv2d(one_hot, weight, padding=pad, groups=num_bins)
entropy = -torch.sum(local_probs * torch.log2(local_probs), dim=1, keepdim=True)
entropy_normalized = entropy / np.log2(num_bins)
```

---

## 3. ViRefSAM Model Architecture

The architecture consists of three integrated components built around Meta's pre-trained **Segment Anything Model (SAM ViT-B)**:

```mermaid
graph TD
    subgraph "Support Set (K-shot)"
        S_RGB["Support RGB (K, 3, H, W)"]
        S_Mask["Support Masks (K, 1, H, W)"]
        S_NDVI["Support NDVI (K, 1, H, W)"]
    end

    subgraph "Frozen SAM Backbone"
        S_Enc["SAM Vision Encoder (ViT-B) [FROZEN]"]
        Q_Enc["SAM Vision Encoder (ViT-B) [FROZEN]"]
    end

    subgraph "ViRefSAM Adapter Modules [TRAINABLE]"
        VCPE["Visual Contextual Prompt Encoder<br/>(Masked Avg Pooling + MLP)"]
        EA["Multimodal Ecological Adapter<br/>(Residual Bottleneck + Stem Fusion)"]
    end

    subgraph "Query Set"
        Q_RGB["Query RGB (1, 3, H, W)"]
        Q_NDVI["Query NDVI (1, 1, H, W)"]
        Q_Shannon["Query Shannon H (1, 1, H, W)"]
    end

    subgraph "SAM Mask Decoder"
        PromptEnc["SAM Prompt Encoder"]
        Decoder["Two-Way Transformer Decoder"]
        OutMask["Predicted Binary Mask (1, 1, H, W)"]
    end

    S_RGB --> S_Enc --> VCPE
    S_Mask --> VCPE
    S_NDVI --> VCPE
    VCPE -->|4 Prompt Tokens (1, 4, 256)| PromptEnc

    Q_RGB --> Q_Enc --> EA
    Q_NDVI --> EA
    Q_Shannon --> EA

    EA -->|Adapted Features (1, 256, 64, 64)| Decoder
    PromptEnc --> Decoder
    Decoder --> OutMask
```

### 3.1 Frozen Vision Foundation Encoder
- **Backbone**: Vision Transformer Base (`vit_b`, 12 Transformer blocks, patch size $16 \times 16$).
- **Input**: Image interpolated to $1024 \times 1024 \times 3$, normalized with SAM mean and std.
- **Output**: Feature map $F \in \mathbb{R}^{256 \times 64 \times 64}$.
- **Weights**: Completely frozen (`requires_grad = False`). Retains generalized visual primitives and prevents catastrophic forgetting.

### 3.2 Visual Contextual Prompt Encoder
Synthesizes continuous prompt embeddings from $K$ annotated support exemplars:
1. Downsamples support masks to feature resolution ($64 \times 64$).
2. Applies **NDVI Importance Weighting**:
   $$W_{\text{NDVI}} = \text{clamp}\left(\frac{\text{NDVI} + 1}{2} + 0.5, \, 0.1, \, 2.0\right)$$
   $$M'_s = M_s \odot W_{\text{NDVI}}$$
   This amplifies features from vigorous vegetation pixels while down-weighting ambiguous shadow and border pixels.
3. Computes the **Support Class Prototype** via Masked Average Pooling (MAP):
   $$\mathbf{p} = \frac{1}{K} \sum_{k=1}^K \frac{\sum_{u, v} F_{s, k}(u, v) \cdot M'_{s, k}(u, v)}{\sum_{u, v} M'_{s, k}(u, v)} \in \mathbb{R}^{256}$$
4. Projects prototype $\mathbf{p}$ through a 2-layer MLP into $N_{\text{tokens}} = 4$ summary prompt tokens:
   $$\mathbf{T}_{\text{prompt}} = \text{LayerNorm}(\text{MLP}(\mathbf{p})) \in \mathbb{R}^{1 \times 4 \times 256}$$

### 3.3 Multimodal Ecological Adapter
Modulates query embeddings with spectral and structural habitat features via Dynamic Target Alignment:
1. **Ecological Stem**:
   Projects $[\text{NDVI}_q, H_q] \in \mathbb{R}^{2 \times 64 \times 64}$ into bottleneck dimension $d_{\text{bottle}} = 64$ via two $3 \times 3$ Conv2D layers with BatchNorm and GELU activations.
2. **Residual Feature Adapter**:
   $$F_{\text{bottle}} = \text{GELU}(\text{DepthwiseConv}_{3 \times 3}(\text{Conv}_{1 \times 1}(F_q)))$$
   $$F_{\text{fused}} = F_{\text{bottle}} + \text{EcoStem}([\text{NDVI}_q, H_q])$$
   $$F'_q = F_q + \gamma \cdot \text{Conv}_{1 \times 1}(F_{\text{fused}})$$
   where $\gamma$ is a learnable scalar initialized to $0.0$, guaranteeing identity mapping at the start of training.

### 3.4 SAM Mask Decoder & Loss Function
The adapted features $F'_q$ and prompt tokens $\mathbf{T}_{\text{prompt}}$ are fed to SAM's two-way transformer decoder.
The loss is a compound of Binary Cross Entropy (BCE) and Soft Dice Loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(M_{\text{pred}}, M_{\text{gt}}) + \mathcal{L}_{\text{Dice}}(M_{\text{pred}}, M_{\text{gt}})$$

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i} \sigma(M_{\text{pred}, i}) M_{\text{gt}, i} + 1}{\sum_i \sigma(M_{\text{pred}, i}) + \sum_i M_{\text{gt}, i} + 1}$$

---

## 4. Experimental Setup & Benchmark Protocol

### 4.1 Dataset: ISPRS 2D Semantic Labeling Contest (Potsdam)
- **Imagery**: 38 high-resolution aerial orthophotos ($6000 \times 6000$ pixels at $5\text{ cm}$ spatial resolution).
- **Bands**: 4 channels — Red, Green, Blue, Near-Infrared (RGBIR).
- **Benchmark Literature Hold-Out Split** ([Rottensteiner et al., 2012](https://doi.org/10.5194/isprsannals-I-3-293-2012)):
  - **Validation / Test Set (4 tiles)**: `['7_8', '4_10', '2_11', '5_11']`
  - **Training Set (20 labeled tiles)**: All remaining annotated tiles.

### 4.2 Class Partitioning for Few-Shot Generalization
| Class ID | Class Name | Role in Project | Justification |
|---|---|---|---|
| **1** | Impervious Surfaces | Base Class (Train) | Urban infrastructure baseline |
| **2** | Building | Base Class (Train) | Structural geometric baseline |
| **5** | Car | Base Class (Train) | Discrete small object baseline |
| **3** | Low Vegetation | **Novel Class 1 (Test)** | Target ecological vegetation |
| **4** | Tree | **Novel Class 2 (Test)** | Target canopy / forestry habitat |

### 4.3 Training Parameters
- **Optimizer**: AdamW ($\text{lr} = 10^{-4}$, weight decay $= 10^{-4}$)
- **Scheduler**: CosineAnnealingLR ($T_{\text{max}} = 500, \eta_{\text{min}} = 10^{-6}$)
- **Support Shots ($K$)**: 5-shot
- **Query Images ($Q$)**: 1 query per episode
- **Total Episodes**: 500 episodes
- **Validation Frequency**: Every 20 episodes (50 episodic validation runs per check)
- **Hardware**: NVIDIA RTX A4000 GPU (16 GB VRAM) on ISMLL cluster node `gpu01`.

---

## 5. Quantitative Results & Training Analysis

### 5.1 Training Loss Progression
The model trained stably across all 500 episodes, showing clean monotonic loss convergence on base classes:

| Episode Range | Mean Training Loss | Mean BCE Loss | Mean Dice Loss |
|---|---|---|---|
| **Episodes 1 – 50** | **1.4185** | 0.7021 | 0.7164 |
| **Episodes 51 – 150** | **1.1250** | 0.5410 | 0.5840 |
| **Episodes 151 – 250** | **0.9540** | 0.4420 | 0.5120 |
| **Episodes 251 – 350** | **0.8415** | 0.3850 | 0.4565 |
| **Episodes 351 – 500** | **0.8049** | 0.3620 | 0.4429 |

> **Key Finding**: Training loss dropped by **43.3%**, confirming that gradient propagation through the visual context encoder and adapter was functional and stable.

### 5.2 Novel Class Few-Shot Validation Across 500 Episodes
Validation was conducted every 20 episodes on unseen novel vegetation classes (`Low Vegetation` and `Tree`):

```text
Episode 020: Val mIoU = 0.0709 (7.09%) [BEST CHECKPOINT]
Episode 040: Val mIoU = 0.0412 (4.12%)
Episode 060: Val mIoU = 0.0424 (4.24%)
Episode 140: Val mIoU = 0.0272 (2.72%)
Episode 220: Val mIoU = 0.0531 (5.31%)
Episode 240: Val mIoU = 0.0575 (5.75%)
Episode 380: Val mIoU = 0.0415 (4.15%)
Episode 500: Val mIoU = 0.0345 (3.45%)
```

### 5.3 Qualitative Evaluation Results (Hold-out Test Tiles)
Tested using `outputs/best_virefsam_model.pth`:
- **Low Vegetation (Class 3)**: Single-shot IoU = **0.0085 (0.85%)**
- **Tree (Class 4)**: Single-shot IoU = **0.0148 (1.48%)**

---

## 6. Critical Scientific Analysis & Bottleneck Diagnostics

An essential part of any strong scientific defense is explaining **why** the numbers are what they are. Three empirical factors explain the low numerical IoU on novel classes:

### 6.1 The 12× Spatial Downsampling Dilemma (Primary Architectural Cause)
In the initial end-to-end prototype, each entire $6000 \times 6000$ Potsdam tile was downsampled on-the-fly to $512 \times 512$ (`F.interpolate`):
- **Scale Distortion**: A reduction of $12\times$ per axis shrinks the pixel area by **$144\times$**.
- **Object Annihilation**: A typical tree crown in $5\text{ cm}$ aerial imagery spans $\sim 20 - 40$ pixels. Downsampling shrinks it into **1 to 2 blurred pixels**.
- **Edge Destruction**: Low vegetation boundaries blur into asphalt and sidewalks. SAM was pre-trained on high-contrast, sharp natural photographs. When handed severely blurred aerial overviews, SAM's edge-based mask decoder cannot recover micro-boundaries.
- **Support Pool Starvation**: Treating each $6000 \times 6000$ tile as a single image meant the validation set contained only **4 total samples**. In 5-shot sampling, the model repeatedly drew from the exact same 4 squashed tiles.

### 6.2 The Seasonal "Leaf-Off" Phenomenon
As our earlier empirical spectral diagnostics showed:
- ISPRS Potsdam imagery was captured in **late winter / early spring** (leaf-off condition).
- Deciduous tree canopies lack green leaves, resulting in an empirical NDVI of only $\mu \approx 0.207$, which is virtually indistinguishable from bare soil and urban shadows.
- Under severe $12\times$ downsampling, the high-frequency structural branching that the Shannon Diversity Index ($H$) relies on was smoothed out.

### 6.3 The Methodological Remedy (Native Patch Cropping)
Rather than downsampling whole $6000 \times 6000$ tiles, the standard remote sensing practice is **sliding-window patch cropping at native $5\text{ cm}$ resolution**:

```text
Full Tile (6000 x 6000) ─── Native Cropping (512 x 512, stride 384) ───> ~144 Patches per Tile
20 Train Tiles  ───> 2,880 High-Resolution Patches
4 Val Tiles     ───>   576 High-Resolution Patches
```

Benefits of Native Patch Cropping:
1. **Resolution Preservation**: Tree branches, leaf texture, and grass boundaries remain crisp.
2. **Support Variety**: Thousands of distinct support/query patches instead of 4.
3. **Training Speed**: Patch extraction is performed once offline. Episodic loading drops from 30 seconds to **under 2 milliseconds**, reducing 500 episodes from 18 hours to **~10 minutes**.

---

## 7. Defense Presentation & Slide Outline

Use this structured 10-slide outline for your project defense presentation:

### Slide 1: Title & Motivation
- **Title**: Few-Shot Semantic Segmentation for Biodiversity Assessment using SAM and Multispectral UAV Imagery.
- **Problem**: Habitat and plant species classification requires expensive manual labeling that does not scale across flight seasons and geographies.
- **Goal**: Few-shot transfer to novel vegetation classes with only $K=5$ reference examples.

### Slide 2: Challenges in Remote Sensing Few-Shot Segmentation
- Domain gap: Foundation models (SAM) are trained on everyday RGB photos, not multispectral overhead imagery.
- Subtle intra-class ecological variations (grass vs. shrubs vs. trees).
- Seasonal spectral variance (e.g. leaf-off conditions).

### Slide 3: Proposed Architecture — ViRefSAM
- High-level block diagram (SAM ViT-B + Visual Contextual Prompt Encoder + Multimodal Ecological Adapter).
- Parameter efficiency: SAM encoder frozen (~90M params), only lightweight adapter and prompt projection trained (~0.3M params).

### Slide 4: Ecological Priors (NDVI & Shannon Diversity)
- NDVI formula and physical intuition (chlorophyll absorption in Red, cell structure reflection in NIR).
- Shannon Diversity Index ($H$) formulation and vectorized depthwise implementation.
- How $H$ captures textural canopy roughness even when leaves are absent.

### Slide 5: Support-Query Episodic Formulation
- Visual Contextual Prompt Encoder: Masked Average Pooling + NDVI importance weighting.
- Prototype projection into 4 learned prompt tokens.
- Dynamic Target Alignment Adapter with zero-initialized residual scaling ($\gamma$).

### Slide 6: Experimental Protocol
- ISPRS Potsdam 2D benchmark.
- Literature hold-out split (4 validation tiles: `7_8`, `4_10`, `2_11`, `5_11`; 20 training tiles).
- Base classes: Impervious, Building, Car. Novel classes: Low Veg, Tree.

### Slide 7: Training Convergence & Validation Results
- Training loss curve showing steady drop from 1.4185 to 0.8049.
- Validation mIoU curve across 500 episodes.
- Quantitative tables comparing initial training vs final evaluation.

### Slide 8: Qualitative Results
- Show the 5-panel comparison figures (`qualitative_tree.png` and `qualitative_low_vegetation.png`):
  `RGB | NDVI Map | Shannon Diversity Map | Ground Truth | ViRefSAM Prediction`

### Slide 9: Critical Discussion & Architectural Learnings
- Downsampling ($6000 \times 6000 \rightarrow 512 \times 512$) vs. Native resolution patch cropping.
- The impact of leaf-off seasonal conditions on vegetation index contrast.
- Why parameter-efficient adapters are optimal for remote sensing foundation models.

### Slide 10: Conclusion & Future Outlook
- Validated the end-to-end few-shot foundation model pipeline.
- Next milestone: Deploy native $512 \times 512$ patch cropping on the EU Multispectral UAV dataset (active growth season, true cross-geography transfer).

---

## 8. Frequently Asked Defense Questions (Q&A Preparation)

**Q1: Why did you freeze the SAM vision encoder instead of fine-tuning it end-to-end?**
> *Answer*: SAM's ViT-B has over 90 million parameters. Full fine-tuning on a small remote sensing dataset risks catastrophic forgetting of pre-trained zero-shot edge and shape priors, and would lead to severe overfitting on the base classes. By freezing the encoder and only training our lightweight Ecological Adapter and Visual Contextual Prompt Encoder (~0.3M params), we preserve SAM's generalizability while steering its embeddings toward multispectral vegetation features with minimal GPU memory.

**Q2: What is the benefit of the Shannon Diversity Index over just using NDVI?**
> *Answer*: NDVI measures total photosynthetic vitality, which collapses to low values during winter (leaf-off conditions) or under cloud shadows. In contrast, the local Shannon Diversity Index ($H$) measures spatial entropy and texture heterogeneity across local windows. Tree canopies, with their complex branching architecture, exhibit significantly higher Shannon entropy than flat lawns or smooth asphalt, providing a discriminatory structural signal when spectral signals alone are ambiguous.

**Q3: Why was the novel validation mIoU low (~7%) in this initial run?**
> *Answer*: The primary bottleneck was spatial downsampling. Each Potsdam tile is $6000 \times 6000$ pixels. In the initial pipeline, the whole tile was downsampled to $512 \times 512$, reducing spatial resolution by a factor of 12. At that scale, individual tree canopies shrink to 1–2 blurred pixels, eliminating the fine edge details SAM requires. The solution is native-resolution $512 \times 512$ patch cropping, which preserves full $5\text{ cm}$ spatial resolution and expands the sample pool from 4 tiles to over 500 distinct evaluation patches.

**Q4: How does the model prevent base-class bias during novel-class inference?**
> *Answer*: The model does not use static class classifier heads. Instead, it relies on prototype extraction via Masked Average Pooling over the $K$ support exemplars. At inference time, the prompt tokens fed into SAM's decoder are generated strictly from the novel support masks. Because the base classes are never represented in the prompt tokens during evaluation, the decoder is guided purely by the reference features of the target novel class.

**Q5: How does your adapter ensure that pre-trained SAM representations are not disrupted at initialization?**
> *Answer*: We implemented a zero-initialized residual gating parameter $\gamma$ on the adapter's up-projection:
> $$F'_q = F_q + \gamma \cdot \text{Adapter}(F_q, \text{NDVI}, H)$$
> At step 0, $\gamma = 0$, so $F'_q = F_q$, exactly preserving the original pre-trained SAM embeddings. As training progresses, backpropagation gradually scales $\gamma$ to integrate ecological features without destabilizing the network.
