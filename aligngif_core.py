from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageFilter

Box = Tuple[int, int, int, int]
RGBA = Tuple[int, int, int, int]


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class Anchor:
    """An anchor region used to estimate alignment shift."""
    box: Box
    weight: float = 1.0
    name: str = ""  # optional label for debugging/UI


@dataclass(frozen=True)
class AlignConfig:
    method: str = "fft"          # "fft" or "fft-refine"
    use_edges: bool = True
    bg: RGBA = (255, 255, 255, 255)

    # FFT-refine only:
    refine_radius: int = 6       # pixels
    refine_weighted_score: bool = True  # apply weights in refinement too


@dataclass
class BestMatch:
    dx: int
    dy: int
    # For FFT: score = -peak (higher peak => better; negative so "lower is better")
    # For refinement: score = weighted SAD on anchors (lower is better)
    score: float


ProgressFn = Optional[Callable[[int, int, BestMatch], None]]  # (tested, total, best)


# -----------------------------
# Image helpers
# -----------------------------

def to_canvas(im: Image.Image, size: Tuple[int, int], bg: RGBA) -> Image.Image:
    canvas = Image.new("RGBA", size, bg)
    canvas.paste(im.convert("RGBA"), (0, 0))
    return canvas


def edge_l(im: Image.Image) -> Image.Image:
    return im.convert("L").filter(ImageFilter.FIND_EDGES)


def _prep_patch(im: Image.Image, box: Box, use_edges: bool) -> Image.Image:
    patch = im.crop(box)
    return edge_l(patch) if use_edges else patch.convert("L")


def _to_float_array(img_l: Image.Image) -> np.ndarray:
    a = np.asarray(img_l, dtype=np.float32)
    a = a - a.mean()
    denom = np.linalg.norm(a)
    if denom > 1e-8:
        a = a / denom
    return a


# -----------------------------
# FFT phase correlation
# -----------------------------

def phase_correlation_shift(a: np.ndarray, b: np.ndarray) -> Tuple[int, int, float]:
    """
    Returns (dx, dy, peak) such that shifting b by (dx,dy) aligns to a (integer shift).
    """
    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)

    R = Fa * np.conj(Fb)
    denom = np.abs(R)
    denom[denom < 1e-12] = 1e-12
    R /= denom

    r = np.fft.ifft2(R)
    r = np.abs(r)

    peak_y, peak_x = np.unravel_index(np.argmax(r), r.shape)
    peak_val = float(r[peak_y, peak_x])

    h, w = r.shape
    dy = int(peak_y) if peak_y <= h // 2 else int(peak_y) - h
    dx = int(peak_x) if peak_x <= w // 2 else int(peak_x) - w
    return dx, dy, peak_val


def estimate_shift_fft(
    ref_img: Image.Image,
    cur_img: Image.Image,
    anchors: Sequence[Anchor],
    use_edges: bool,
) -> BestMatch:
    """
    Estimate a global shift by combining per-anchor FFT shifts via weighted average.
    """
    if not anchors:
        raise ValueError("At least one anchor is required.")

    dx_sum = 0.0
    dy_sum = 0.0
    w_sum = 0.0
    peak_sum = 0.0

    for a in anchors:
        if a.weight <= 0:
            continue

        ref_patch = _prep_patch(ref_img, a.box, use_edges)
        cur_patch = _prep_patch(cur_img, a.box, use_edges)

        dx, dy, peak = phase_correlation_shift(_to_float_array(ref_patch), _to_float_array(cur_patch))

        dx_sum += a.weight * dx
        dy_sum += a.weight * dy
        w_sum += a.weight
        peak_sum += a.weight * peak

    if w_sum <= 0:
        raise ValueError("All anchors have non-positive weights; nothing to use.")

    dx_final = int(round(dx_sum / w_sum))
    dy_final = int(round(dy_sum / w_sum))

    # score: negative peak so "lower is better"
    peak_avg = peak_sum / w_sum
    return BestMatch(dx=dx_final, dy=dy_final, score=-float(peak_avg))


# -----------------------------
# Refinement (local search only)
# -----------------------------

def _weighted_sad_score(
    ref_img: Image.Image,
    cur_img_shifted: Image.Image,
    anchors: Sequence[Anchor],
    use_edges: bool,
    weighted: bool,
) -> float:
    """
    Weighted sum of absolute differences on anchor patches.
    Lower is better.
    """
    total = 0.0
    for a in anchors:
        if a.weight <= 0:
            continue
        w = a.weight if weighted else 1.0
        ref_p = _prep_patch(ref_img, a.box, use_edges)
        cur_p = _prep_patch(cur_img_shifted, a.box, use_edges)
        diff = ImageChops.difference(ref_p, cur_p)
        total += w * float(sum(diff.getdata()))
    return total


def refine_shift_local(
    ref_img: Image.Image,
    cur_img: Image.Image,
    anchors: Sequence[Anchor],
    use_edges: bool,
    init_dx: int,
    init_dy: int,
    refine_radius: int,
    weighted_score: bool,
    progress: ProgressFn = None,
) -> BestMatch:
    """
    Local search around (init_dx, init_dy) within +/- refine_radius using SAD on anchors.
    """
    if refine_radius < 0:
        raise ValueError("refine_radius must be >= 0")

    best = BestMatch(dx=init_dx, dy=init_dy, score=float("inf"))
    total_tests = (2 * refine_radius + 1) ** 2
    tested = 0

    for dx in range(init_dx - refine_radius, init_dx + refine_radius + 1):
        for dy in range(init_dy - refine_radius, init_dy + refine_radius + 1):
            tested += 1
            shifted = ImageChops.offset(cur_img, dx, dy)
            s = _weighted_sad_score(ref_img, shifted, anchors, use_edges, weighted_score)
            if s < best.score:
                best = BestMatch(dx=dx, dy=dy, score=s)
            if progress:
                progress(tested, total_tests, best)

    return best


# -----------------------------
# Public API
# -----------------------------

def align_frames(
    images: Sequence[Image.Image],
    anchors: Sequence[Anchor],
    cfg: AlignConfig = AlignConfig(),
    progress: Optional[Callable[[int, int, int, BestMatch], None]] = None,
) -> List[Image.Image]:
    """
    Align frames to the first frame using FFT (optionally refined).
    progress callback signature:
        progress(frame_idx, total_frames, stage, best)
    where stage is 0=FFT estimate, 1=refine (if enabled).
    """
    if len(images) < 2:
        return list(images)
    if cfg.method not in ("fft", "fft-refine"):
        raise ValueError("cfg.method must be 'fft' or 'fft-refine'")
    if not anchors:
        raise ValueError("At least one anchor is required.")

    ref = images[0].convert("RGBA")
    aligned: List[Image.Image] = [ref]
    size = ref.size

    for i, im in enumerate(images[1:], start=2):
        cur = to_canvas(im, size, cfg.bg)

        best_fft = estimate_shift_fft(ref, cur, anchors, cfg.use_edges)
        if progress:
            progress(i, len(images), 0, best_fft)

        if cfg.method == "fft-refine" and cfg.refine_radius > 0:
            best = refine_shift_local(
                ref_img=ref,
                cur_img=cur,
                anchors=anchors,
                use_edges=cfg.use_edges,
                init_dx=best_fft.dx,
                init_dy=best_fft.dy,
                refine_radius=cfg.refine_radius,
                weighted_score=cfg.refine_weighted_score,
                progress=(lambda t, T, b: progress(i, len(images), 1, b)) if progress else None,
            )
        else:
            best = best_fft

        shifted = ImageChops.offset(cur, best.dx, best.dy)
        aligned.append(shifted)

    return aligned
