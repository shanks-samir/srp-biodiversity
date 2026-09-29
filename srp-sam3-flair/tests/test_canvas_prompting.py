"""Tests for canvas geometry and ecological prompt selection.

No SAM 3, no data, no GPU. What is under test is the arithmetic that silently produces wrong
science: whether the canvas rectangles match the published geometry, whether a box survives
the round trip from source pixels to normalised canvas coordinates, and whether the
ecological selector actually prefers what it claims to prefer.

    python tests/test_canvas_prompting.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.canvas import (  # noqa: E402
    CANVAS_SIZE, SPLIT_RATIO, compute_layout, build_canvas, box_to_canvas,
    extract_query_mask,
)
from models.prompting import (  # noqa: E402
    find_components, select_prompt_box, score_candidate_masks, STRATEGIES,
)


def test_layout_matches_published_geometry():
    print("[1/7] Canvas layout reproduces the published geometry...")
    lay = compute_layout()
    # 1008 * 0.6 = 604 to the support, 404 to the query, stacked vertically, support on top.
    assert lay.support == (0, 0, 1008, 604), f"support rect {lay.support}"
    assert lay.query == (0, 604, 1008, 404), f"query rect {lay.query}"

    # The tiles must partition the canvas exactly: no gap, no overlap, no separator.
    assert lay.support[3] + lay.query[3] == CANVAS_SIZE
    assert lay.support[1] == 0 and lay.query[1] == lay.support[3]

    # The support deliberately gets MORE area than the query.
    assert lay.support[3] > lay.query[3], "support should get the larger share at ratio 0.6"

    swapped = compute_layout(swap_order=True)
    assert swapped.query == (0, 0, 1008, 404) and swapped.support == (0, 404, 1008, 604)
    # swap_order moves the support, it must not resize it.
    assert swapped.support[3] == lay.support[3]

    horiz = compute_layout(orientation="horizontal")
    assert horiz.support == (0, 0, 604, 1008) and horiz.query == (604, 0, 404, 1008)

    for bad in (0.0, 1.0, -0.2, 1.5):
        try:
            compute_layout(split_ratio=bad)
            raise AssertionError(f"split_ratio={bad} should have raised")
        except ValueError:
            pass
    print("      --> Passed!")


def test_canvas_composition():
    print("[2/7] Canvas is fully covered and tiles land in the right halves...")
    support = np.zeros((3, 512, 512), dtype=np.float32); support[0] = 1.0   # pure red
    query = np.zeros((3, 512, 512), dtype=np.float32);  query[2] = 1.0      # pure blue

    canvas, lay, _ = build_canvas(support, query)
    assert canvas.shape == (1008, 1008, 3) and canvas.dtype == np.uint8

    # No black fill should survive -- the two tiles cover the canvas exactly.
    assert not (canvas.sum(axis=2) == 0).any(), "canvas has uncovered (black) pixels"

    top, bottom = canvas[:604], canvas[604:]
    assert top[..., 0].mean() > 200 and top[..., 2].mean() < 55, "support half is not red"
    assert bottom[..., 2].mean() > 200 and bottom[..., 0].mean() < 55, "query half is not blue"

    # Non-square input must be force-resized to fill, not letterboxed.
    wide = np.zeros((3, 100, 900), dtype=np.float32); wide[1] = 1.0
    canvas2, _, _ = build_canvas(wide, query)
    assert not (canvas2[:604].sum(axis=2) == 0).any(), "aspect-preserving padding detected"
    print("      --> Passed!")


def test_box_round_trip():
    print("[3/7] Box maps into normalised centre-format canvas coordinates...")
    lay = compute_layout()
    # A box covering the whole 512x512 support must fill the support rect exactly.
    norm = box_to_canvas((0, 0, 512, 512), lay.support, (512, 512))
    cx, cy, w, h = norm
    assert np.isclose(w, 1.0), f"width {w} should span the canvas"
    assert np.isclose(h, 604 / 1008, atol=1e-6), f"height {h} should span the support tile"
    assert np.isclose(cx, 0.5) and np.isclose(cy, (604 / 2) / 1008)

    # A quarter-sized box in the source's top-left must stay in the support's top-left.
    cx, cy, w, h = box_to_canvas((0, 0, 256, 256), lay.support, (512, 512))
    assert np.isclose(w, 0.5) and np.isclose(h, (604 / 2) / 1008)
    assert cx < 0.5 and cy < (604 / 1008) / 2 + 1e-9

    # Everything must stay inside the unit square, and centre format must not be confused
    # with corner format -- cx is a CENTRE, so it cannot be 0 for a box at the origin.
    for v in box_to_canvas((10, 20, 100, 80), lay.support, (512, 512)):
        assert 0.0 <= v <= 1.0
    assert box_to_canvas((0, 0, 100, 100), lay.support, (512, 512))[0] > 0.0

    # And the box must follow the support when swap_order moves it.
    swapped = compute_layout(swap_order=True)
    assert box_to_canvas((0, 0, 512, 512), swapped.support, (512, 512))[1] > 0.5
    print("      --> Passed!")


def test_query_mask_extraction():
    print("[4/7] Query mask is recovered from a whole-canvas prediction...")
    lay = compute_layout()
    canvas_mask = np.zeros((1008, 1008), dtype=np.uint8)
    canvas_mask[:604] = 1          # model fired on the support half (it always will)
    canvas_mask[700:800, 100:200] = 1   # and on a region of the query half

    crop = extract_query_mask(canvas_mask, lay)
    assert crop.shape == (404, 1008), f"crop shape {crop.shape}"
    # The support half must be discarded entirely.
    assert crop[:90].sum() == 0, "support-half prediction leaked into the query mask"
    assert crop[96:196, 100:200].sum() > 0, "query-half prediction was lost"

    resized = extract_query_mask(canvas_mask, lay, out_size=(512, 512))
    assert resized.shape == (512, 512)
    assert set(np.unique(resized)).issubset({0, 1}), "mask stopped being binary"
    print("      --> Passed!")


def test_component_filtering():
    print("[5/7] Stuff-class guards: specks dropped, sprawl dropped...")
    mask = np.zeros((512, 512), dtype=bool)
    mask[10:14, 10:14] = True         # 16 px speck -> below min_area
    mask[100:160, 100:160] = True     # 3600 px block -> keep
    comps = find_components(mask, min_area_px=500, max_area_frac=0.60)
    assert len(comps) == 1, f"expected 1 qualifying component, got {len(comps)}"
    assert comps[0][2] == (100, 100, 60, 60)
    assert np.isclose(comps[0][3], 1.0), "a solid square should have fill ratio 1"

    # A component covering most of the tile yields a vacuous box and must be rejected.
    sprawl = np.zeros((512, 512), dtype=bool); sprawl[:400] = True   # 78% of the patch
    assert find_components(sprawl, min_area_px=500, max_area_frac=0.60) == []

    assert find_components(np.zeros((64, 64), dtype=bool)) == []
    print("      --> Passed!")


def test_ecological_selection():
    print("[6/7] Ecological strategy prefers the spectrally typical component...")
    mask = np.zeros((512, 512), dtype=bool)
    mask[50:110, 50:110] = True      # A: typical
    mask[200:260, 200:260] = True    # B: atypical (shaded -- much lower NDVI)
    mask[400:460, 400:460] = True    # C: typical

    ndvi = np.zeros((512, 512), dtype=np.float32)
    ndvi[50:110, 50:110] = 0.70
    ndvi[200:260, 200:260] = 0.15    # the outlier
    ndvi[400:460, 400:460] = 0.72

    picked = select_prompt_box(mask, strategy="ecological", ndvi=ndvi)
    assert picked is not None
    x, y, w, h = picked.box_xywh
    assert (x, y) != (200, 200), "ecological strategy picked the shaded outlier"
    assert picked.stats["ndvi"] > 0.5, f"selected component NDVI {picked.stats['ndvi']}"
    assert not picked.is_vacuous

    # All strategies must return a usable, correctly-shaped box.
    for strat in STRATEGIES:
        kw = {"ndvi": ndvi} if strat == "ecological" else {}
        box = select_prompt_box(mask, strategy=strat, rng=np.random.default_rng(0), **kw)
        assert box is not None and box.strategy == strat
        assert box.box_xywh[2] > 0 and box.box_xywh[3] > 0

    # 'ecological' without any signal is a silent no-op waiting to happen -- it must raise.
    try:
        select_prompt_box(mask, strategy="ecological")
        raise AssertionError("ecological with no ndvi/shannon should have raised")
    except ValueError:
        pass

    # An unqualifiable mask must return None, never a whole-mask fallback box.
    assert select_prompt_box(np.zeros((512, 512), dtype=bool), strategy="largest") is None

    # A sprawling diagonal component is un-box-like and should be flagged vacuous.
    diag = np.zeros((512, 512), dtype=bool)
    for i in range(0, 300):
        diag[i:i + 8, i:i + 8] = True
    got = select_prompt_box(diag, strategy="largest")
    if got is not None:
        assert got.fill_ratio < 0.35 and got.is_vacuous
    print("      --> Passed!")


def test_candidate_rescoring():
    print("[7/7] Ecological re-scoring reorders SAM 3's candidate set...")
    h = w = 64
    masks = np.zeros((3, h, w), dtype=bool)
    masks[0, 0:20, 0:20] = True      # spectrally right, but the model ranks it lowest
    masks[1, 20:40, 20:40] = True    # spectrally wrong, model ranks it highest
    masks[2, 40:60, 40:60] = True

    ndvi = np.zeros((h, w), dtype=np.float32)
    ndvi[0:20, 0:20] = 0.70
    ndvi[20:40, 20:40] = 0.05
    ndvi[40:60, 40:60] = 0.40

    scores = np.array([0.55, 0.90, 0.60])
    ref = {"ndvi": 0.70}

    unchanged = score_candidate_masks(masks, scores, ndvi=ndvi, reference=ref, weight=0.0)
    assert np.allclose(unchanged, scores), "weight=0 must reproduce the model's own ranking"
    assert int(np.argmax(unchanged)) == 1

    adjusted = score_candidate_masks(masks, scores, ndvi=ndvi, reference=ref, weight=1.0)
    assert int(np.argmax(adjusted)) == 0, "re-scoring did not promote the spectrally correct mask"

    # No reference means no opinion.
    assert np.allclose(score_candidate_masks(masks, scores, ndvi=ndvi), scores)

    # An empty candidate must be driven out of contention, not silently scored 0.
    empties = np.zeros((1, h, w), dtype=bool)
    assert score_candidate_masks(empties, np.array([0.9]), ndvi=ndvi, reference=ref)[0] == -np.inf
    print("      --> Passed!")


if __name__ == "__main__":
    print("\n--- Canvas geometry and ecological prompting tests ---")
    test_layout_matches_published_geometry()
    test_canvas_composition()
    test_box_round_trip()
    test_query_mask_extraction()
    test_component_filtering()
    test_ecological_selection()
    test_candidate_rescoring()
    print("\nAll canvas and prompting tests passed.\n")
