"""Spatial-concatenation canvas for training-free few-shot segmentation with SAM 3.

SAM 3 does not support cross-image exemplar prompting. Its exemplar tokens cross-attend to
the feature map of the frame they were drawn on, so a box placed on a support image cannot
address a separate query image. The canvas trick exists to work around exactly that: paste
support and query onto ONE image, and the encoder's self-attention carries the correspondence
between the two halves. It is a necessity, not a shortcut.

Geometry follows Tsai, Lin & Wang, "Few-Shot Semantic Segmentation Meets SAM3", and their
released implementation. The numbers below are theirs, not tuned by us:

  canvas       1008 x 1008 RGB
  orientation  vertical (support and query stacked) -- their ablation finds it beats horizontal
  split_ratio  0.6 of the canvas to the SUPPORT. Counter-intuitive -- the support gets MORE
               area than the query -- but it is what reproduces their headline numbers.
  resize       forced, aspect ratio deliberately destroyed. Their ablation: "FR consistently
               outperforms ARP, indicating that maximizing token density within the attention
               space is more beneficial than preserving geometric fidelity."

The two tiles partition the canvas exactly: no padding, no separator, no gutter.

IMPORTANT: the support mask never reaches the model. Only a box, as four normalised numbers
through the geometry encoder. Nothing is drawn into the pixels -- the green rectangles in the
paper's figures are matplotlib overlays on saved JPEGs, not model input.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

CANVAS_SIZE = 1008
SPLIT_RATIO = 0.6
Rect = Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass(frozen=True)
class CanvasLayout:
    """Where the support and query tiles sit on the canvas."""

    support: Rect
    query: Rect
    canvas_size: int = CANVAS_SIZE

    @property
    def query_box_xyxy(self) -> Tuple[int, int, int, int]:
        """Query rectangle as (x0, y0, x1, y1), for cropping the prediction back out."""
        x, y, w, h = self.query
        return x, y, x + w, y + h


def compute_layout(
    canvas_size: int = CANVAS_SIZE,
    split_ratio: float = SPLIT_RATIO,
    orientation: str = "vertical",
    swap_order: bool = False,
) -> CanvasLayout:
    """Partition the canvas into a support tile and a query tile.

    Args:
        canvas_size: edge length; SAM 3's processor resizes to 1008 so anything else is
            resized again downstream and wastes resolution
        split_ratio: fraction of the canvas given to the SUPPORT (0.6 reproduces the paper)
        orientation: 'vertical' (stacked) or 'horizontal' (side by side)
        swap_order: False puts the support first (top / left), True puts it second

    The support always receives thickness `int(canvas_size * split_ratio)`; swap_order only
    decides which side it occupies.
    """
    if not 0.0 < split_ratio < 1.0:
        raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio}")
    if orientation not in ("vertical", "horizontal"):
        raise ValueError(f"orientation must be 'vertical' or 'horizontal', got {orientation!r}")

    split = int(canvas_size * split_ratio)
    rem = canvas_size - split

    if orientation == "vertical":
        if swap_order:
            query = (0, 0, canvas_size, rem)
            support = (0, rem, canvas_size, split)
        else:
            support = (0, 0, canvas_size, split)
            query = (0, split, canvas_size, rem)
    else:
        if swap_order:
            query = (0, 0, rem, canvas_size)
            support = (rem, 0, split, canvas_size)
        else:
            support = (0, 0, split, canvas_size)
            query = (split, 0, rem, canvas_size)

    return CanvasLayout(support=support, query=query, canvas_size=canvas_size)


def _to_pil(image: np.ndarray) -> Image.Image:
    """Accept (3, H, W) or (H, W, 3), float in [0, 1] or uint8, return an RGB PIL image."""
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-dimensional image, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"expected 3 channels, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _paste(canvas: Image.Image, image: np.ndarray, rect: Rect) -> Tuple[float, float]:
    """Force-resize `image` into `rect` and paste. Returns the (sx, sy) scale factors."""
    x, y, w, h = rect
    pil = _to_pil(image)
    sx, sy = w / pil.width, h / pil.height
    canvas.paste(pil.resize((w, h), Image.BILINEAR), (x, y))
    return sx, sy


def box_to_canvas(
    box_xywh: Tuple[float, float, float, float],
    rect: Rect,
    orig_size: Tuple[int, int],
    canvas_size: int = CANVAS_SIZE,
) -> Tuple[float, float, float, float]:
    """Map a box from original-image pixels to normalised centre-format canvas coordinates.

    Args:
        box_xywh: (x, y, w, h) in the source image's own pixel coordinates
        rect: the canvas rectangle that image was pasted into
        orig_size: (width, height) of the source image before the forced resize
        canvas_size: canvas edge length

    Returns:
        (cx, cy, w, h), **centre format, normalised to [0, 1]** -- what the NATIVE `sam3`
        package expects via `processor.add_geometric_prompt(state, box=..., label=True)`.

    WARNING: the two SAM 3 APIs take different box formats, and a box in the wrong format is
    silently valid but points somewhere else entirely -- far worse than a crash. Use
    `box_to_canvas_xyxy` for the HuggingFace `transformers` API (`Sam3Processor`), which wants
    absolute xyxy PIXELS. See `models/canvas.py` docstring and the README for which path you
    are on.
    """
    bx, by, bw, bh = box_xywh
    ox, oy, ow, oh = rect
    orig_w, orig_h = orig_size
    sx, sy = ow / orig_w, oh / orig_h

    px, py = bx * sx + ox, by * sy + oy
    return (
        (px + bw * sx / 2) / canvas_size,
        (py + bh * sy / 2) / canvas_size,
        (bw * sx) / canvas_size,
        (bh * sy) / canvas_size,
    )


def box_to_canvas_xyxy(
    box_xywh: Tuple[float, float, float, float],
    rect: Rect,
    orig_size: Tuple[int, int],
    canvas_size: int = CANVAS_SIZE,
) -> Tuple[int, int, int, int]:
    """Same mapping, but returning absolute xyxy PIXELS on the canvas.

    This is the format the HuggingFace `transformers` API wants:

        processor(images=canvas, input_boxes=[[[x1, y1, x2, y2]]],
                  input_boxes_labels=[[1]], return_tensors="pt")

    where the label is 1 for a positive exemplar and 0 for a negative one.
    """
    cx, cy, w, h = box_to_canvas(box_xywh, rect, orig_size, canvas_size)
    x1 = int(round((cx - w / 2) * canvas_size))
    y1 = int(round((cy - h / 2) * canvas_size))
    x2 = int(round((cx + w / 2) * canvas_size))
    y2 = int(round((cy + h / 2) * canvas_size))
    # Clamp: a box straddling the tile edge must not address the other tile.
    return (
        max(0, min(x1, canvas_size - 1)),
        max(0, min(y1, canvas_size - 1)),
        max(1, min(x2, canvas_size)),
        max(1, min(y2, canvas_size)),
    )


def build_canvas(
    support_image: np.ndarray,
    query_image: np.ndarray,
    support_box_xywh: Optional[Tuple[float, float, float, float]] = None,
    canvas_size: int = CANVAS_SIZE,
    split_ratio: float = SPLIT_RATIO,
    orientation: str = "vertical",
    swap_order: bool = False,
) -> Tuple[np.ndarray, CanvasLayout, Optional[Tuple[float, float, float, float]]]:
    """Compose the support/query canvas and map the support box into canvas coordinates.

    Args:
        support_image: (3, H, W) or (H, W, 3); float in [0, 1] or uint8
        query_image: same
        support_box_xywh: prompt box in the SUPPORT image's own pixel coordinates

    Returns:
        canvas: (canvas_size, canvas_size, 3) uint8
        layout: where each tile landed, needed to crop the prediction back out
        norm_box: (cx, cy, w, h) normalised, or None if no box was supplied
    """
    layout = compute_layout(canvas_size, split_ratio, orientation, swap_order)
    canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))

    support_pil = _to_pil(support_image)
    orig_size = (support_pil.width, support_pil.height)

    _paste(canvas, support_image, layout.support)
    _paste(canvas, query_image, layout.query)

    norm_box = None
    if support_box_xywh is not None:
        norm_box = box_to_canvas(support_box_xywh, layout.support, orig_size, canvas_size)

    return np.array(canvas), layout, norm_box


def extract_query_mask(
    canvas_mask: np.ndarray,
    layout: CanvasLayout,
    out_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Recover the query-image mask from a whole-canvas prediction.

    Crops the query rectangle and undoes the forced resize. Note this discards anything the
    model predicted on the support half, which is correct -- the support tile contains the
    very object the model was prompted with, so it is guaranteed to fire there and those
    pixels are not a prediction about the query.

    Args:
        canvas_mask: (canvas_size, canvas_size) binary or float
        layout: the layout used to build the canvas
        out_size: (height, width) of the original query image; None keeps tile resolution
    """
    x0, y0, x1, y1 = layout.query_box_xyxy
    crop = canvas_mask[y0:y1, x0:x1]
    if out_size is None:
        return crop

    out_h, out_w = out_size
    binary = crop.dtype == bool or set(np.unique(crop)).issubset({0, 1})
    pil = Image.fromarray((crop > 0.5).astype(np.uint8) * 255 if binary
                          else (np.clip(crop, 0, 1) * 255).astype(np.uint8))
    # NEAREST for masks: bilinear would invent intermediate values along every boundary.
    resized = np.array(pil.resize((out_w, out_h), Image.NEAREST if binary else Image.BILINEAR))
    return (resized > 127).astype(np.uint8) if binary else resized.astype(np.float32) / 255.0
