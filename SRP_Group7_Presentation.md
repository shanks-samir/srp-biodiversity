

Few-Shot Learning for Biodiversity
## Assessment
Semantic segmentation of habitat classes from UAV imagery with K ∈ {1, 5} labelled examples per
class
Group 7  SRP 2025/26

## Team Members
## Ali Eralp Sen (1751097)  Awais Ali (1751181)  Naman Sethi (1751127)
## Priyesh Shrestha (1751076)  Samir Paudyal (1751102)

## Supervisors:
## Prof. Dr. Niels Landwehr
## Ahmad Bdeir

Problem statement
Why this matters
•Biodiversity loss is accelerating due to climate change and human activity.
•Accurate, large-scale monitoring is critical for conservation and policy decisions.
•It is a global problem — different landscapes, climates, and vegetation across regions.
•Automated land-cover segmentation reduces the load on ecologists and policy makers.
Why UAV imagery
•High spatial resolution (0.1–0.3 m per pixel).
•Cost-effective and scalable to whole regions.
•Enables monitoring of vegetation types, species
distribution, and habitat structure.
The technical bottleneck
•Pixel-level annotation requires domain experts — slow
and expensive.
•Real biodiversity projects only have a handful of labelled
images per class.
•Few-shot segmentation: learn from K ∈ {1, 5} examples
instead of thousands.
Group 7 · Few-Shot Learning for Biodiversity Assessment2 / 20

Research question & formal setting
## Question
Can a model trained on base classes C_base learn to segment novel classes C_novel in unseen regions
from K labelled support examples per class?
Few-shot semantic segmentation — formal setting
Support set    S = {(x_i, y_i)}_{i=1..K}     K labelled (image, mask) pairs
Query image  x_q                                unlabelled, from same novel class
Predict        ŷ_q = f(x_q ; S)              segmentation mask for x_q
Objective    maximise mIoU over C_novel  where C_novel ∩ C_base = ∅
Group 7 · Few-Shot Learning for Biodiversity Assessment3 / 20

Related work — SAM & SAM2
SAM  (Kirillov et al., 2023)
•ViT image encoder (ViT-H, 636M parameters) trained on SA-1B: 11M images, 1.1B masks.
•Prompt encoder supports points, boxes, masks, and text.
•Lightweight mask decoder produces class-agnostic segmentation — segments regions but does not assign
semantic labels.
SAM2  (Ravi et al., 2024) — used in this work
•Hiera Vision Transformer — hierarchical, multi-scale features. We use Hiera-Small (~44M parameters).
•Trained on SA-V (video) + SA-1B, giving stronger spatial reasoning than the original SAM. ~6× faster on images.
•Empirical motivation: zero-shot oracle mIoU = 0.85 on LoveDA — the encoder already separates aerial structure
well.
Group 7 · Few-Shot Learning for Biodiversity Assessment
## 4 / 20

Related work — baselines we compare against
HRNet-W32  (Wang et al., 2020) — supervised baseline
•High-Resolution Network: maintains high-resolution features through every stage rather than downsampling then
upsampling.
•Parallel multi-resolution branches with repeated cross-resolution fusion. W32 = 32-channel base width.
•Strong on fine boundaries — but requires full pixel labels for every scene.
DeepLabv2 + CBST  (Zou et al., 2018) — Unsupervised Domain Adaptation
•DeepLabv2: ResNet-101 backbone with atrous (dilated) convolutions and ASPP for multi-scale context.
•CBST = Class-Balanced Self-Training: iteratively generates pseudo-labels on the unlabelled target with
class-balanced confidence thresholds.
•Setup: train on labelled Rural domain, adapt to unlabelled Urban domain via pseudo-labels.
Group 7 · Few-Shot Learning for Biodiversity Assessment5 / 20

Related work — few shot and ecological indices
Rouse et al. (1974) — Introduction of NDVI
•Original NDVI paper, developed for the NASA Earth Resources Technology Satellite (ERTS-1) Great Plains rangeland
project.
•Defines NDVI = (NIR − Red)/(NIR + Red); demonstrated strong correlation with green biomass and vegetation vigour.
•The most widely used vegetation index in remote sensing for over 50 years — established standard in ecology.
Wei et al. (2025) — Few-Shot UAV Vegetation Segmentation
•Most directly comparable prior work: combines UAV imagery, few-shot segmentation, and vegetation classes.
•Cross-matching + self-matching meta-learner over support–query features; SAM as a boundary refiner.
•Differs from our approach: meta-learning + matching rather than prototype matching; no ecological channels.
Group 7 · Few-Shot Learning for Biodiversity Assessment
## 6/20
Wang et al. (2019) — PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment
•Extends prototypical networks to semantic segmentation (the actual task we tackle).
•Uses masked average pooling to extract class-specific prototypes from support image–mask pairs.
•Achieves 48.1% / 55.7% mIoU on PASCAL-5ⁱ (1-shot / 5-shot) — sets the prototype-based baseline we build on.

Our approach — overview
Gap addressed
No prior work combines SAM2 + prototype-based few-shot + NDVI/Shannon ecological channels + multi-country UAV
evaluation in one pipeline.
Four components
SAM2 Hiera-S encoder — Frozen in Phase 1; last 4 transformer blocks unfrozen in Phase 2. Provides multi-scale
visual features.
FPN-style decoder — Trained head with skip connections that fuses encoder features at three resolutions back to
per-pixel predictions.
Ecological channels (NDVI + Shannon) — NDVI captures vegetation health, Shannon captures habitat heterogeneity.
Concatenated with encoder features at the decoder input. Encoder stays pure RGB.
Prototype-based few-shot — Average encoder features over K support examples → class prototype. Query pixels
assigned by cosine similarity.
Group 7 · Few-Shot Learning for Biodiversity Assessment7 / 20

Architecture — SAM2 encoder + FPN decoder + ecology fusion
RGB tile
## 1024×1024×3
SAM2 Hiera-S
encoder
## Stage 1
## Stage 2
## Stage 3
## Stage 4
Phase 1: frozen ❄
Phase 2: last 4 blocks trainable
f1
## 256×256×96
f2
## 128×128×192
f3
## 64×64×384
FPN decoder
## Stage 3
## Conv 384→128
## Stage 2
Concat f2  ·  Conv 320→128
## Stage 1
Concat f1  ·  Conv 224→64
Trained · 1.5M params
## Output
## 1024×1024×7
NDVI + Shannon  (2
channels)
Group 7 · Few-Shot Learning for Biodiversity Assessment
## 8 / 20

Ecological channels — NDVI & Shannon diversity
NDVI — Normalised Difference Vegetation Index
## Red
band
## NIR
band
## NDVI =
(NIR − Red)
/ (NIR + Red)
NDVI map
1 channel
Measures vegetation health. Healthy plants reflect NIR strongly → high
NDVI. A field that looks green in RGB but is stressed, drought-affected,
or degraded shows up clearly in NDVI. This is the signal ecologists use
to assess vegetation vigour.
Shannon diversity index — spatial heterogeneity
## NDVI
map
## Sliding
window
## Histogram
over k bins
## Shannon
map
H(p) = − Σᵢ pᵢ log pᵢ
over k-bin NDVI histogram in window w
Measures habitat heterogeneity in a spatial patch. High H = mixed
habitat (forest edges, mosaic landscapes — typically biodiversity-rich).
Low H = uniform region (monoculture). A quantity ecologists already
use in landscape ecology.
Where they enter the pipeline
RGB only → SAM2 encoder (frozen). NDVI and Shannon are computed separately, downsampled to match each encoder feature's
resolution, and concatenated with encoder features at the FPN decoder input. Encoder pretraining stays untouched.
Group 7 · Few-Shot Learning for Biodiversity Assessment9 /20

NDVI Example
## 10/20
Group 7 · Few-Shot Learning for Biodiversity Assessment

Few-shot mechanism — prototype matching
Step 1 — build a class prototype from K support examples
## Support 1
## (img +
mask)
## Support 2
## (img +
mask)
## ⋮
## Support K
## (img +
mask)
SAM2 encoder
## (frozen)
## Masked
average pool

extract features at
class-labelled pixels
Class prototype
p_c (256-d vector)
p_c = (1/K) Σᵢ MaskedAvgPool(F(xᵢ), yᵢ)
Step 2 — classify each query pixel by similarity to prototype
## Query
image
SAM2 encoder
Pixel features
## F(x_q)
cosine_sim(F(x_q), p_c)
Predicted mask
ŷ_q
Group 7 · Few-Shot Learning for Biodiversity Assessment11 /20

Overall pipeline — five staged experiments
## 1
Supervised baseline
HRNet-W32, full labels
LoveDA Urban+Rural
## 2
UDA baseline
DeepLabv2 + CBST, no target labels
LoveDA Rural→Urban
## 3
SAM2 + FPN fine-tune
Phase 1 frozen / Phase 2 partial unfreeze
LoveDA Urban+Rural
## 4a
Few-shot novel classes
Prototype matching from Phase 2 encoder
LoveDA base→novel, K∈{1,5}
## 4b
+ Ecology channels
NDVI + Shannon at decoder fusion
## Vaihingen
## 5
Cross-country transfer
Same pipeline, target country labels K=5
LoveDA → EU UAV (DE first)
Stage / approachData setup
Group 7 · Few-Shot Learning for Biodiversity Assessment12 / 20

Datasets — LoveDA (public benchmark, used for validation)
Size:      5,987 tiles · 1024×1024 px · 0.3 m GSD
## Classes:  7 — Background, Building, Road, Water, Barren, Forest, Agriculture
Domains: Urban (2,713) · Rural (3,274) — strong distributional shift
Sensor:   RGB only · Google Earth · cities: Nanjing, Changzhou, Wuhan (CN)
Pixel distribution
Background 43%  ·  Agriculture 24%  ·  Forest 12%  ·  Building 10%  ·  Road 6%  ·  Water 3%  ·  Barren 1.7%
Heavy imbalance — Barren at 1.7% is the hardest class. Motivates inverse-frequency loss weighting.
Group 7 · Few-Shot Learning for Biodiversity Assessment13 / 20

Datasets — EU UAV  &  ISPRS Vaihingen
EU Multispectral UAV  [Primary target]
•Countries: Portugal, Germany, UK, Austria.
•Sensors: RGB on all sites; NIR available on a subset.
•Habitats: grassland, forest edges, shrubland, agricultural fields, riparian zones.
•Labels: pixel-level on a subset — sparse, motivating the few-shot setting.
ISPRS Vaihingen  [Supplementary]
•High-resolution aerial imagery from Vaihingen, Germany. Channels: RGB + Infrared + Digital Surface Model
## (DSM).
•Full pixel-level annotations across all tiles.
•Role: validate the NDVI/Shannon fusion under fully-supervised conditions before applying it to the sparse EU
UAV labels.
Group 7 · Few-Shot Learning for Biodiversity Assessment14 / 20

Experiment plan
StageMethodSetupQuestion answered
-  SupervisedHRNet-W32LoveDA Urban+Rural, full labelsBest score with full supervision?
-  UDADeepLabv2 + CBSTTrain Rural, adapt Urban (no labels)Cost of removing target labels?
-  SAM2 fine-tuneFrozen ViT + FPN (Ph 1, 2)LoveDA Urban+RuralDoes the foundation model help?
4a. Few-shot novelPrototype matchingBase→novel split; K ∈ {1, 5}Can prototypes segment unseen classes?
4b. + EcologyPrototype + NDVI + ShannonSame as 4a, decoder-side fusionDo ecological channels help?
## 5.  Cross-country
Same pipeline, K target
labels
LoveDA → EU UAV (Germany first)Does it transfer to new countries?
Stages 4a → 4b → 5 form the research contribution. Stages 1–3 are baselines and ablations.
Group 7 · Few-Shot Learning for Biodiversity Assessment
## 15 / 20

Training details
HRNet-W32DeepLabv2 + CBSTSAM2 + FPN
BackboneHRNetV2-W32ResNet-50, OS=16Hiera-S ViT (44M)
Input resolution512×512 crop512×5121024×1024 full
Batch size1682
Iterations / epochs15,000 iters15,000 iters20 epochs
OptimizerSGD, mom=0.9SGD, mom=0.9Adam
Weight decay1e-45e-4default
Learning rate0.01 (poly, p=0.9)0.011e-4 (cosine)
AugmentationFlip + Rot90 (p=0.75)——
Training setUrban + Rural
## Rural (lbl), Urban (unlbl)
## Urban + Rural
Trainable params29M (all)44M (all)1.5M (decoder only)
Result (mIoU val)0.580.3510.4746 (Ph 1)
Group 7 · Few-Shot Learning for Biodiversity Assessment16/20

Stage 3 Phase 1 results — per-class mIoU on LoveDA
## Observations
Largest gains over UDA on
minority classes:
## Barren  0.07 → 0.23  (+0.16)
## Agriculture  0.25 → 0.49  (+0.24)
## Forest  0.23 → 0.37  (+0.14)

Gap to oracle (0.85) sits on fine
spatial classes — motivates
Phase 2 encoder unfreeze.
- SAM2 Zero-Shot uses ground-truth majority vote for class assignment — upper bound, not a usable model.
Group 7 · Few-Shot Learning for Biodiversity Assessment
## 17/20
ClassHRNetUDASAM2 ZS *SAM2 FT
## Background0.530.320.700.5173
## Building0.560.550.780.5995
## Road0.550.440.880.4999
## Water0.680.590.970.6167
## Barren0.240.070.870.2285
## Forest0.420.230.780.3704
## Agriculture0.480.250.950.4901
mIoU0.5800.3510.8460.4746

## Supervised Result Visualized
Group 7 · Few-Shot Learning for Biodiversity Assessment19/20

Aggregate comparison
## Interpretation
HRNet >> UDA:
HRNet (0.58) leads UDA (0.351) by 0.23
— CBST pseudo-labels fail to recover
supervised performance under domain
shift.

SAM2 zero-shot = upper bound:
0.85 with oracle labelling — encoder
separates classes well.

SAM2 fine-tune:
SAM2 FT (0.47) trails HRNet (0.58) with
frozen encoder — Phase 2 unfreeze
targets 0.60–0.72 to close the gap.
- oracle: ground-truth used to assign classes to SAM2's class-agnostic masks
Group 7 · Few-Shot Learning for Biodiversity Assessment18/20

Next steps & open questions
Immediate — Stage 3 Phase 2 encoder fine-tune
Unfreeze the last 4 Hiera blocks. LR 2e-5 for encoder, 2e-4 for decoder. 20 more epochs on the same cosine schedule. Open
question: how much of the 0.85 oracle is reachable without overfitting?
Stages 4a & 4b — prototype-based few-shot on LoveDA, then with ecology
Split LoveDA classes into base / novel. Build class prototypes from K∈{1,5} support examples. 4a: prototypes from RGB features
only. 4b: prototypes from RGB features + NDVI + Shannon (decoder-side fusion). Open question: do ecological channels improve
over RGB alone, or is SAM2 sufficient?
Stage 5 — cross-country transfer to EU UAV
Validate the NDVI/Shannon fusion on ISPRS Vaihingen (full labels) before transferring to the EU UAV dataset — starting with
Germany. Open question: does the prototype-based pipeline generalise across countries with only K labelled examples per class?
Group 7 · Few-Shot Learning for Biodiversity Assessment
## 20/20