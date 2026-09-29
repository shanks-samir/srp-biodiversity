"""Deriving a geometric prompt from a land-cover mask.

FSS-SAM3 prompts SAM 3 with a tight bounding box around one representative object INSTANCE,
taken from instance-level annotations. Their motivation: a box around a whole semantic mask
"may contain multiple object instances or large background regions" and is ambiguous.

That reasoning applies with far more force to land cover, and the method gives no recipe for
it. FLAIR has no instance annotations, and its classes are "stuff": herbaceous vegetation,
deciduous, brushwood. A box around the full semantic mask of "herbaceous vegetation" is
frequently the entire tile -- a prompt that says "the target is somewhere in this image",
which carries no information. The paper's own warning is that SAM 3 is "highly sensitive to
the spatial precision of geometric prompts".

So the question becomes: given a stuff mask, WHICH region do you box? This module answers it,
and the answer is where the ecological priors earn their place. Rather than fusing NDVI into
a feature map and hoping a learned gate uses it, NDVI and Shannon entropy *select the
exemplar* -- a decision that is interpretable, cheap, and directly ablatable against
non-ecological baselines.

Strategies, ordered as an ablation ladder:
    random      pick any qualifying component            (floor)
    largest     biggest component by area                (the obvious baseline)
    compact     best fill ratio, i.e. most box-like      (geometry only, no ecology)
    ecological  most spectrally typical of its class     (the contribution)

`ecological` is deliberately built on top of `compact`: a spectrally perfect but sprawling
component still makes a vacuous box, so compactness gates and ecology ranks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

STRATEGIES = ("random", "largest", "compact", "ecological")


@dataclass
class PromptBox:
    """A derived prompt box and the diagnostics needed to ablate the choice that produced it."""

    box_xywh: Tuple[int, int, int, int]
    component_id: int
    area_px: int
    fill_ratio: float          # component area / bbox area -- how box-like the region is
    strategy: str
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def is_vacuous(self) -> bool:
        """True when the box is so unlike its contents that it conveys little.

        A fill ratio below 0.35 means most of what SAM 3 sees inside the prompt is NOT the
        target class, which is precisely the ambiguity FSS-SAM3's instance-level prompting
        was introduced to avoid.
        """
        return self.fill_ratio < 0.35


def _bbox_of(component: np.ndarray) -> Tuple[int, int, int, int]:
    """Tight (x, y, w, h) around a boolean component."""
    ys, xs = np.nonzero(component)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def find_components(
    class_mask: np.ndarray,
    min_area_px: int = 500,
    max_area_frac: float = 0.60,
    connectivity: int = 2,
) -> List[Tuple[int, np.ndarray, Tuple[int, int, int, int], float]]:
    """Label the class mask and keep components that could make a meaningful prompt.

    Args:
        class_mask: (H, W) boolean or 0/1 for ONE target class
        min_area_px: drop specks. 500 px at 0.2 m GSD is 20 m^2.
        max_area_frac: drop components covering more than this fraction of the patch. A
            component spanning most of the tile yields a box spanning most of the tile, which
            tells SAM 3 nothing. This is the stuff-class guard and it has no analogue in the
            original method.
        connectivity: 2 = 8-connected (diagonals count), appropriate for organic boundaries

    Returns:
        list of (label_id, component_mask, bbox_xywh, fill_ratio), unordered
    """
    mask = np.asarray(class_mask).astype(bool)
    if not mask.any():
        return []

    structure = ndimage.generate_binary_structure(2, connectivity)
    labelled, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return []

    total_px = mask.size
    out = []
    for label_id in range(1, n + 1):
        component = labelled == label_id
        area = int(component.sum())
        if area < min_area_px or area > max_area_frac * total_px:
            continue
        x, y, w, h = _bbox_of(component)
        out.append((label_id, component, (x, y, w, h), area / float(w * h)))
    return out


def select_prompt_box(
    class_mask: np.ndarray,
    strategy: str = "ecological",
    ndvi: Optional[np.ndarray] = None,
    shannon: Optional[np.ndarray] = None,
    min_area_px: int = 500,
    max_area_frac: float = 0.60,
    min_fill_ratio: float = 0.35,
    rng: Optional[np.random.Generator] = None,
) -> Optional[PromptBox]:
    """Choose one component of `class_mask` and return a prompt box around it.

    The `ecological` strategy scores each candidate by how SPECTRALLY TYPICAL it is of its
    own class within this patch -- the component whose mean NDVI sits closest to the class
    median, penalised for being un-box-like. The intent is to avoid prompting SAM 3 with a
    shaded, senescent or otherwise atypical fragment, which would define the concept badly
    for the whole episode. Shannon entropy enters the same way when supplied, so structurally
    atypical fragments are also avoided.

    Returns None when nothing qualifies -- the caller should skip the episode rather than
    fall back to a whole-mask box, which would silently reintroduce the vacuous prompt.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    rng = rng or np.random.default_rng()

    candidates = find_components(class_mask, min_area_px, max_area_frac)
    if not candidates:
        return None

    # Compactness gates every strategy except the deliberately naive baselines, so the
    # ablation compares like with like on everything except the ranking signal.
    if strategy in ("compact", "ecological"):
        gated = [c for c in candidates if c[3] >= min_fill_ratio]
        if gated:
            candidates = gated

    if strategy == "random":
        label_id, component, bbox, fill = candidates[rng.integers(len(candidates))]
        stats = {}

    elif strategy == "largest":
        label_id, component, bbox, fill = max(candidates, key=lambda c: int(c[1].sum()))
        stats = {}

    elif strategy == "compact":
        label_id, component, bbox, fill = max(candidates, key=lambda c: c[3])
        stats = {}

    else:  # ecological
        mask = np.asarray(class_mask).astype(bool)
        signals = {}
        if ndvi is not None:
            signals["ndvi"] = np.asarray(ndvi, dtype=np.float64)
        if shannon is not None:
            signals["shannon"] = np.asarray(shannon, dtype=np.float64)

        if not signals:
            raise ValueError(
                "strategy='ecological' needs at least one of ndvi= or shannon=. "
                "Use strategy='compact' for the geometry-only control."
            )

        # Class-level reference: what this class looks like across the whole patch.
        reference = {k: float(np.median(v[mask])) for k, v in signals.items()}
        # Spread normalises the deviation so NDVI and Shannon contribute comparably.
        spread = {
            k: float(np.std(v[mask])) or 1.0
            for k, v in signals.items()
        }

        best, best_score = None, -np.inf
        for label_id, component, bbox, fill in candidates:
            component_stats = {k: float(v[component].mean()) for k, v in signals.items()}
            deviation = np.mean([
                abs(component_stats[k] - reference[k]) / spread[k] for k in signals
            ])
            # Typicality dominates; compactness breaks ties and penalises sprawl.
            score = -deviation + 0.5 * fill
            if score > best_score:
                best_score = score
                best = (label_id, component, bbox, fill, component_stats, deviation)

        label_id, component, bbox, fill, component_stats, deviation = best
        stats = {**component_stats,
                 **{f"class_median_{k}": v for k, v in reference.items()},
                 "deviation": float(deviation),
                 "score": float(best_score)}

    return PromptBox(
        box_xywh=bbox,
        component_id=int(label_id),
        area_px=int(component.sum()),
        fill_ratio=float(fill),
        strategy=strategy,
        stats=stats,
    )


def score_candidate_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    ndvi: Optional[np.ndarray] = None,
    shannon: Optional[np.ndarray] = None,
    reference: Optional[Dict[str, float]] = None,
    weight: float = 1.0,
) -> np.ndarray:
    """Re-score SAM 3's candidate masks using ecological evidence.

    The second insertion point for the priors, and the cheaper of the two to ablate. FSS-SAM3
    recovers its final mask as a pixelwise UNION over every candidate clearing a 0.5 gate,
    with no per-instance selection -- `raw_map = torch.max(pred_masks[:, 0], dim=0)[0]`. That
    is a crude rule and it leaves room: a candidate whose interior is spectrally wrong for the
    target class can be down-weighted before the union is taken.

    Because this operates on cached candidate sets, dozens of rules can be swept offline in
    seconds for the compute cost of a single inference pass.

    Args:
        masks: (N, H, W) boolean candidate masks
        scores: (N,) the model's own confidence per candidate
        ndvi, shannon: (H, W) ecological rasters for the QUERY image at its native resolution
        reference: expected {'ndvi': v, 'shannon': v} for the target class. Derived from the
            support set in a real episode -- never from the query's ground truth.
        weight: 0 reproduces the model's own ranking exactly; higher trusts ecology more.

    Returns:
        (N,) adjusted scores.
    """
    masks = np.asarray(masks).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    if reference is None or weight == 0.0:
        return scores

    signals = {}
    if ndvi is not None and "ndvi" in reference:
        signals["ndvi"] = np.asarray(ndvi, dtype=np.float64)
    if shannon is not None and "shannon" in reference:
        signals["shannon"] = np.asarray(shannon, dtype=np.float64)
    if not signals:
        return scores

    adjusted = scores.copy()
    for i, mask in enumerate(masks):
        if not mask.any():
            adjusted[i] = -np.inf
            continue
        deviation = np.mean([
            abs(float(v[mask].mean()) - reference[k]) for k, v in signals.items()
        ])
        adjusted[i] = scores[i] - weight * deviation
    return adjusted
