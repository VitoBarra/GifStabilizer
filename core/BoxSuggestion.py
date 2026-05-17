#!/usr/bin/env python3
"""
BoxSuggestion.py — Suggest stable, high-information anchor boxes for image alignment.

Goal:
  Given a sequence of frames, suggest candidate ROI boxes that are:
    - rich in structure (edges / detail)  -> good for correlation
    - stable across frames (low temporal variance) -> robust anchor

Typical usage:
  from PIL import Image
  from BoxSuggestion import SuggestConfig, suggest_anchor_boxes

  frames = [Image.open(p).convert("RGBA") for p in paths]
  cfg = SuggestConfig(box_size=(520, 170), topk=8, stride=24, lambda_stability=2.0, region="top-right")
  boxes = suggest_anchor_boxes(frames, cfg)

Requires:
  pip install pillow numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import  List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

from core.DataModel import Box



def _phase_corr_shift(a: np.ndarray, b: np.ndarray) -> Tuple[int, int, float]:
    """
    Phase correlation shift between a (ref) and b (cur).
    Returns (dx, dy, peak).
    dx,dy are integer pixel shifts such that shifting b by (dx,dy) aligns it to a.
    """
    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)
    R = Fa * np.conj(Fb)
    denom = np.abs(R)
    denom[denom < 1e-12] = 1e-12
    R /= denom
    r = np.fft.ifft2(R)
    r = np.abs(r)

    py, px = np.unravel_index(np.argmax(r), r.shape)
    peak = float(r[py, px])

    h, w = r.shape
    dy = int(py) if py <= h // 2 else int(py) - h
    dx = int(px) if px <= w // 2 else int(px) - w
    return dx, dy, peak


def _shift2d(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Circular shift (roll). Good enough for stability estimation."""
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)


def _to_float_norm(a: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-norm (robust for phase correlation)."""
    a = a.astype(np.float32)
    a = a - float(a.mean())
    n = float(np.linalg.norm(a))
    if n > 1e-8:
        a = a / n
    return a



# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class SuggestConfig:
    # Target ROI size in pixels (in reference frame coords)
    box_size: Tuple[int, int] = (520, 170)  # (W,H)

    # Slide step (pixels)
    stride: int = 24

    # Number of suggestions to return (after NMS)
    topk: int = 8

    # Stability weight λ (penalize temporal variance)
    # score = detail - λ * variance_norm
    lambda_stability: float = 2.0

    # Use edges as detail map (recommended)
    use_edges: bool = True

    # Optional region constraint:
    # None, or one of:
    # "full", "top", "bottom", "left", "right",
    # "top-left", "top-right", "bottom-left", "bottom-right", "center"
    region: Optional[str] = None

    # NMS overlap threshold (IoU); lower = more diverse
    iou_thresh: float = 0.35

    # Speed: downscale factor auto; set to 1 to disable
    auto_downscale: bool = True

    # When frames differ in size, pad to reference size with this RGBA
    pad_rgba: Tuple[int, int, int, int] = (255, 255, 255, 255)


# -----------------------------
# Public API
# -----------------------------

def suggest_anchor_boxes(frames: Sequence[Image.Image], cfg: SuggestConfig) -> List[Box]:
    """
    Return up to cfg.topk suggested boxes (x1,y1,x2,y2) in reference frame coords.

    Reference frame is frames[0].

    Notes:
      - If frames are different sizes, they are padded to reference size (top-left aligned).
      - Suggestions are deterministic.
    """
    if len(frames) == 0:
        return []

    ref = frames[0].convert("RGBA")
    iw, ih = ref.size
    W, H = cfg.box_size

    if W <= 0 or H <= 0 or W > iw or H > ih:
        raise ValueError(f"box_size {cfg.box_size} must fit inside reference frame {ref.size}")

    stride = max(1, int(cfg.stride))
    topk = max(1, int(cfg.topk))
    lam = float(cfg.lambda_stability)

    down = _choose_downscale(ref.size, cfg)  # 1..N
    if down > 1:
        ref_small_size = (max(1, iw // down), max(1, ih // down))
        W2, H2 = max(8, W // down), max(8, H // down)
        stride2 = max(1, stride // down)
    else:
        ref_small_size = (iw, ih)
        W2, H2 = W, H
        stride2 = stride

    # Build stable score map at small size
    score = _score_map_stable(frames, ref_size=ref_small_size, cfg=cfg)  # float32 [H,W]
    # Apply region constraint at small size
    region_box_small = _region_box(ref_small_size, cfg.region)
    # Enumerate candidates & pick topk with NMS
    boxes_small = _topk_boxes_from_score(
        score=score,
        W=W2,
        H=H2,
        stride=stride2,
        topk=topk,
        iou_thresh=cfg.iou_thresh,
        region=region_box_small,
    )

    # Scale back to full-res coords
    boxes: List[Box] = []
    for x1, y1, x2, y2 in boxes_small:
        boxes.append((int(x1 * down), int(y1 * down), int(x2 * down), int(y2 * down)))

    # Clamp to ref
    out: List[Box] = []
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, min(iw - 1, x1))
        y1 = max(0, min(ih - 1, y1))
        x2 = max(x1 + 1, min(iw, x2))
        y2 = max(y1 + 1, min(ih, y2))
        out.append((x1, y1, x2, y2))

    return out


# -----------------------------
# Core logic
# -----------------------------

def _choose_downscale(size: Tuple[int, int], cfg: SuggestConfig) -> int:
    if not cfg.auto_downscale:
        return 1
    w, h = size
    m = max(w, h)
    if m > 2600:
        return 3
    if m > 1600:
        return 2
    return 1


def _pad_to_ref(im: Image.Image, ref_size: Tuple[int, int], pad_rgba: Tuple[int, int, int, int]) -> Image.Image:
    if im.size == ref_size:
        return im.convert("RGBA")
    canvas = Image.new("RGBA", ref_size, pad_rgba)
    canvas.paste(im.convert("RGBA"), (0, 0))
    return canvas


def _to_gray_np(im: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    g = im.convert("L")
    if g.size != size:
        g = g.resize(size, Image.BILINEAR)
    return (np.asarray(g, dtype=np.float32) / 255.0)


def _edge_np(im: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    g = im.convert("L")
    if g.size != size:
        g = g.resize(size, Image.BILINEAR)
    e = g.filter(ImageFilter.FIND_EDGES)
    return (np.asarray(e, dtype=np.float32) / 255.0)


def _score_map_stable(frames: Sequence[Image.Image], ref_size: Tuple[int, int], cfg: SuggestConfig) -> np.ndarray:
    """
    Motion-compensated stability:
      1) estimate global integer shift for each frame vs reference (FFT phase correlation)
      2) shift frames to align to reference at ref_size
      3) compute variance over time on aligned stack
      4) score = detail(ref) - λ * normalized_variance
    """
    if len(frames) == 0:
        return np.zeros((ref_size[1], ref_size[0]), dtype=np.float32)

    ref_full = frames[0].convert("RGBA")
    ref_big_size = ref_full.size

    # reference grayscale at ref_size
    ref_refsize = _to_gray_np(_pad_to_ref(ref_full, ref_big_size, cfg.pad_rgba), ref_size)
    ref_refsize_n = _to_float_norm(ref_refsize)

    # compute detail map from reference at ref_size
    if cfg.use_edges:
        detail = _edge_np(_pad_to_ref(ref_full, ref_big_size, cfg.pad_rgba), ref_size)
    else:
        a = ref_refsize
        gx = np.abs(a[:, 1:] - a[:, :-1])
        gy = np.abs(a[1:, :] - a[:-1, :])
        detail = np.zeros_like(a)
        detail[:, 1:] += gx
        detail[1:, :] += gy
        detail = np.clip(detail, 0.0, 1.0)

    a = ref_refsize

    gx = np.abs(a[:, 1:] - a[:, :-1])
    gy = np.abs(a[1:, :] - a[:-1, :])

    contrast = np.zeros_like(a)
    contrast[:, 1:] += gx
    contrast[1:, :] += gy

    contrast = np.clip(contrast, 0.0, 1.0)

    detail = np.clip(detail + 0.35 * contrast, 0.0, 1.0)

    # build aligned grayscale stack
    grays_aligned = [ref_refsize]
    for im in frames[1:]:
        im2 = _pad_to_ref(im, ref_big_size, cfg.pad_rgba)
        cur = _to_gray_np(im2, ref_size)
        cur_n = _to_float_norm(cur)

        dx, dy, _peak = _phase_corr_shift(ref_refsize_n, cur_n)
        cur_aligned = _shift2d(cur, dx, dy)
        grays_aligned.append(cur_aligned)

    stack = np.stack(grays_aligned, axis=0)  # [T,H,W]
    var_map = stack.var(axis=0).astype(np.float32)

    # robust normalize variance -> [0..1]
    v95 = float(np.percentile(var_map, 95))
    if v95 > 1e-8:
        v = np.clip(var_map / v95, 0.0, 1.0)
    else:
        v = np.zeros_like(var_map)

    score = detail - float(cfg.lambda_stability) * v
    return score.astype(np.float32)



def _integral(a: np.ndarray) -> np.ndarray:
    """Summed area table with 1px pad."""
    return np.pad(a, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)


def _window_sum(integ: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    return float(integ[y2, x2] - integ[y1, x2] - integ[y2, x1] + integ[y1, x1])


def _region_box(size: Tuple[int, int], region: Optional[str]) -> Optional[Box]:
    """
    Return (x1,y1,x2,y2) constraint in the given size space.
    """
    if region is None or region == "full":
        return None

    w, h = size
    # Split into thirds-ish: center is middle, corners are quarter-ish
    x_mid1 = w // 3
    x_mid2 = 2 * w // 3
    y_mid1 = h // 3
    y_mid2 = 2 * h // 3

    region = region.lower().strip()

    if region == "top":
        return (0, 0, w, y_mid2)
    if region == "bottom":
        return (0, y_mid1, w, h)
    if region == "left":
        return (0, 0, x_mid2, h)
    if region == "right":
        return (x_mid1, 0, w, h)
    if region == "center":
        return (x_mid1, y_mid1, x_mid2, y_mid2)

    if region == "top-left":
        return (0, 0, x_mid2, y_mid2)
    if region == "top-right":
        return (x_mid1, 0, w, y_mid2)
    if region == "bottom-left":
        return (0, y_mid1, x_mid2, h)
    if region == "bottom-right":
        return (x_mid1, y_mid1, w, h)

    raise ValueError(f"Unknown region: {region}")


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _topk_boxes_from_score(
    score: np.ndarray,
    W: int,
    H: int,
    stride: int,
    topk: int,
    iou_thresh: float,
    region: Optional[Box],
) -> List[Box]:
    """
    Pick top-k windows by average score using integral images + NMS.
    """
    Himg, Wimg = score.shape
    if W > Wimg or H > Himg:
        return []

    integ = _integral(score)
    area = float(W * H)

    if region is None:
        rx1, ry1, rx2, ry2 = 0, 0, Wimg, Himg
    else:
        rx1, ry1, rx2, ry2 = region
        rx1 = max(0, min(Wimg - 1, rx1))
        ry1 = max(0, min(Himg - 1, ry1))
        rx2 = max(rx1 + 1, min(Wimg, rx2))
        ry2 = max(ry1 + 1, min(Himg, ry2))

    # Candidate enumeration (coarse stride)
    candidates: List[Tuple[float, Box]] = []
    y_start = ry1
    y_end = max(ry1, ry2 - H)
    x_start = rx1
    x_end = max(rx1, rx2 - W)

    for y1 in range(y_start, y_end + 1, stride):
        y2 = y1 + H
        for x1 in range(x_start, x_end + 1, stride):
            x2 = x1 + W
            s = _window_sum(integ, x1, y1, x2, y2) / area
            candidates.append((s, (x1, y1, x2, y2)))

    candidates.sort(key=lambda t: t[0], reverse=True)

    chosen: List[Box] = []
    for s, box in candidates:
        if len(chosen) >= topk:
            break
        if all(_iou(box, c) < iou_thresh for c in chosen):
            chosen.append(box)

    return chosen
