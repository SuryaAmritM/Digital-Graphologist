from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from skimage.morphology import skeletonize


def _find_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, val in enumerate(values):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _ink_mask_without_vertical_rulers(ink_mask: np.ndarray) -> np.ndarray:
    """Remove tall thin strokes (notebook margin lines) from bounds logic.

    A vertical margin line makes global y_min the top of the page; the headline
    band then misses the real Devanagari line and every glyph is flagged floating.
    """
    h, w = ink_mask.shape
    if not ink_mask.any():
        return ink_mask

    ink_u8 = (ink_mask.astype(np.uint8) * 255)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink_u8, connectivity=8)
    out = ink_mask.copy()
    max_thin_w = max(14, w // 45)
    for i in range(1, n):
        _x, _y, bw, bh, _area = stats[i]
        if bw <= max_thin_w and bh > 0.42 * h and bh > 5 * max(1, bw):
            out[labels == i] = False
    return out


def _text_vertical_bounds(ink_mask: np.ndarray) -> tuple[int, int, int, int]:
    """Min/max x/y of ink after dropping vertical rulers (falls back to full ink)."""
    filtered = _ink_mask_without_vertical_rulers(ink_mask)
    ys, xs = np.where(filtered)
    if len(xs) == 0:
        ys, xs = np.where(ink_mask)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _top_band_line_instability(
    binary_image: np.ndarray, band_y0: int, band_y1: int
) -> float:
    """How much the headline wanders vertically in the top band (0 = stable, 1 = unstable).

    Binarized ink can stay connected for shaky handwriting; skeleton columns capture
    micro-tremor and waviness better than continuity alone.
    """
    ink_mask = binary_image == 0
    band_h = max(1, band_y1 - band_y0 + 1)
    slice_ink = ink_mask[band_y0 : band_y1 + 1, :]
    if not slice_ink.any():
        return 0.0

    sk = skeletonize(slice_ink)
    if not sk.any():
        return 0.0

    cols = np.where(sk.any(axis=0))[0]
    if len(cols) < 4:
        return 0.0

    y_means: list[float] = []
    for c in cols:
        rows = np.where(sk[:, c])[0]
        if rows.size:
            y_means.append(float(np.mean(rows)))

    if len(y_means) < 4:
        return 0.0

    y_arr = np.array(y_means, dtype=np.float64)
    # Light smoothing reduces single-column skeleton noise on neat writing.
    if y_arr.size >= 5:
        k = np.array([0.25, 0.5, 0.25], dtype=np.float64)
        y_s = np.convolve(y_arr, k, mode="same")
    else:
        y_s = y_arr
    vertical_spread = float(np.std(y_s))
    rough = float(np.mean(np.abs(np.diff(y_s))))

    norm = max(1.0, band_h * 0.45)
    instability = (vertical_spread * 0.55 + rough * 0.45) / norm
    return float(min(1.0, max(0.0, instability)))


def analyze_shirorekha(binary_image: np.ndarray) -> dict[str, Any]:
    """Analyze top horizontal continuity for Devanagari-style anchor line.

    Word clusters are separated by horizontal gaps. Measuring continuity across
    the full text span treats word gaps as Shirorekha breaks. We close small
    gaps to segment words, then score continuity and gaps *within* each segment.
    """
    ink_mask = binary_image == 0
    if not ink_mask.any():
        return {
            "continuity_ratio": 0.0,
            "break_count": 0,
            "break_boxes": [],
            "top_band": (0, 0),
            "anchor_y1": 0,
            "shiro_span": (0, 0),
        }

    y_min, y_max, x_min, x_max = _text_vertical_bounds(ink_mask)
    h_img = binary_image.shape[0]
    text_height = max(1, y_max - y_min + 1)
    text_width = max(1, x_max - x_min + 1)
    # Thin strip: actual headline metrics (continuity + instability).
    headline_h = max(10, min(32, int(text_height * 0.078)))
    # Wider anchor: Shirorekha + upper connectors for attachment / dilation (not for jitter).
    anchor_h = max(18, min(52, int(text_height * 0.13)))
    band_y0 = y_min
    band_y1 = min(h_img - 1, y_min + headline_h - 1)
    anchor_y1 = min(h_img - 1, y_min + anchor_h - 1)
    if anchor_y1 < band_y1:
        anchor_y1 = band_y1

    top_band_2d = ink_mask[band_y0 : band_y1 + 1, :].astype(np.uint8) * 255
    # Bridge only intra-word gaps; too large a kernel merges a whole line and
    # treats word spacing as many Shirorekha breaks.
    close_w = max(21, min(40, text_width // 45))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, 1))
    closed_band = cv2.morphologyEx(top_band_2d, cv2.MORPH_CLOSE, h_kernel)
    closed_bin = closed_band > 0

    n_seg, seg_labels, stats, _ = cv2.connectedComponentsWithStats(
        closed_bin.astype(np.uint8), connectivity=8
    )

    segment_continuities: list[float] = []
    break_boxes: list[tuple[int, int, int, int]] = []
    break_count = 0

    min_segment_area = max(40, text_width // 80)
    min_segment_width = max(24, text_width // 120)
    stroke_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1))

    for seg_id in range(1, n_seg):
        x, y, w, h, area = stats[seg_id]
        if area < min_segment_area or w < min_segment_width:
            continue
        x0, x1 = x, x + w - 1
        slice_orig = ink_mask[band_y0 : band_y1 + 1, x0 : x1 + 1].astype(np.uint8) * 255
        slice_bridged = cv2.morphologyEx(slice_orig, cv2.MORPH_CLOSE, stroke_bridge)
        col_has = slice_bridged.any(axis=0)
        if col_has.size == 0:
            continue
        seg_cont = float(col_has.mean())
        segment_continuities.append(seg_cont)

        gap_runs = _find_runs(~col_has)
        # Wider threshold: thin stroke gaps are bridged above; remaining gaps are structural.
        significant_gaps = [(s, e) for s, e in gap_runs if (e - s + 1) >= 10]
        break_count += len(significant_gaps)
        for s, e in significant_gaps:
            gap_x0 = x0 + s
            gap_x1 = x0 + e
            break_boxes.append(
                (gap_x0, band_y0, gap_x1 - gap_x0 + 1, band_y1 - band_y0 + 1)
            )

    if segment_continuities:
        mean_cont = float(np.mean(segment_continuities))
        min_cont = float(np.min(segment_continuities))
        # Blend: one weak word cluster should pull the score down (dysgraphia signal).
        continuity_ratio = 0.65 * mean_cont + 0.35 * min_cont
    else:
        col_has_global = ink_mask[band_y0 : band_y1 + 1, x_min : x_max + 1].any(axis=0)
        continuity_ratio = float(col_has_global.mean()) if col_has_global.size else 0.0

    if ink_mask[band_y0 : band_y1 + 1, :].any(axis=0).any():
        cols = np.where(ink_mask[band_y0 : band_y1 + 1, :].any(axis=0))[0]
        shiro_x0, shiro_x1 = int(cols.min()), int(cols.max())
    else:
        shiro_x0, shiro_x1 = x_min, x_max

    top_band_instability = _top_band_line_instability(binary_image, band_y0, band_y1)

    return {
        "continuity_ratio": continuity_ratio,
        "segment_mean_continuity": float(np.mean(segment_continuities))
        if segment_continuities
        else continuity_ratio,
        "segment_min_continuity": float(np.min(segment_continuities))
        if segment_continuities
        else continuity_ratio,
        "top_band_instability": top_band_instability,
        "break_count": break_count,
        "break_boxes": break_boxes,
        "top_band": (band_y0, band_y1),
        "anchor_y1": anchor_y1,
        "shiro_span": (shiro_x0, shiro_x1),
    }


def _column_vertical_attachment(
    headline_dil: np.ndarray,
    shiro_x0: int,
    shiro_x1: int,
    x: int,
    y: int,
    w: int,
    h: int,
    comp_mask: np.ndarray,
) -> bool:
    """True if some column has dilated headline ink and this component in that column."""
    xa = max(shiro_x0, x)
    xb = min(shiro_x1, x + w - 1)
    if xa > xb:
        return False
    _, w_img = headline_dil.shape
    for col in range(xa, xb + 1):
        if col < 0 or col >= w_img:
            continue
        if not headline_dil[:, col].any():
            continue
        local_col = col - x
        if 0 <= local_col < w and comp_mask[:, local_col].any():
            return True
    return False


def analyze_hanging_attachment(binary_image: np.ndarray, rule_a: dict[str, Any]) -> dict[str, Any]:
    """Detect lower components that do not connect to the Shirorekha region.

    Uses a vertically dilated Shirorekha band so stem connections count as
    attachment (avoids false floats from strict row bounds).
    """
    ink_mask = (binary_image == 0).astype(np.uint8)
    ink_bool = binary_image == 0
    band_y0, band_y1 = rule_a["top_band"]
    headline_y1 = band_y1
    anchor_y1 = int(rule_a.get("anchor_y1", band_y1))
    shiro_x0, shiro_x1 = rule_a["shiro_span"]

    h_img, w_img = binary_image.shape[:2]
    band_h = max(1, anchor_y1 - band_y0 + 1)
    # Wider anchor ink + small extension so dilated reach meets lower bodies.
    y1_attach = min(h_img - 1, anchor_y1 + max(3, band_h // 5))
    shiro_strip = np.zeros_like(ink_mask)
    shiro_strip[band_y0 : y1_attach + 1, :] = ink_mask[band_y0 : y1_attach + 1, :]
    v_reach = max(14, min(40, (y1_attach - band_y0 + 1) * 3))
    # Wider footprint so lateral matras and joint strokes still count as attached.
    join_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, v_reach))
    shiro_reach = cv2.dilate(shiro_strip, join_kernel)

    headline_slice = ink_bool[band_y0 : headline_y1 + 1, :].astype(np.uint8) * 255
    hl_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5))
    headline_dil = cv2.dilate(headline_slice, hl_kernel, iterations=1) > 0

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)

    floating_boxes: list[tuple[int, int, int, int]] = []
    attached_count = 0
    checked_count = 0

    # Ignore dots, punctuation, and noise blobs; focus on letter-sized bodies.
    min_body_area = max(120, (w_img * h_img) // 40000)

    for label in range(1, n_labels):
        x, y, w, h, area = stats[label]
        if area < min_body_area:
            continue

        comp_x0, comp_x1 = x, x + w - 1
        overlap_x = not (comp_x1 < shiro_x0 or comp_x0 > shiro_x1)
        if not overlap_x:
            continue

        comp_mask = labels[y : y + h, x : x + w] == label
        row_indices = np.where(comp_mask.any(axis=1))[0] + y
        if len(row_indices) == 0:
            continue
        has_body_below_band = bool((row_indices > anchor_y1).any())
        if not has_body_below_band:
            continue
        if h < 10:
            continue

        checked_count += 1
        roi_labels = labels[y : y + h, x : x + w]
        roi_reach = shiro_reach[y : y + h, x : x + w]
        touches_shiro = bool(np.any((roi_labels == label) & (roi_reach > 0)))
        column_linked = _column_vertical_attachment(
            headline_dil,
            shiro_x0,
            shiro_x1,
            x,
            y,
            w,
            h,
            comp_mask,
        )

        if touches_shiro or column_linked:
            attached_count += 1
        else:
            floating_boxes.append((x, y, w, h))

    floating_count = len(floating_boxes)
    attachment_ratio = attached_count / checked_count if checked_count > 0 else 1.0

    return {
        "checked_count": checked_count,
        "attached_count": attached_count,
        "floating_count": floating_count,
        "attachment_ratio": float(attachment_ratio),
        "floating_boxes": floating_boxes,
    }
