"""
example_main.py — run alignment directly from code (no CLI args)

Produces in ./export/ :
  anchors_debug.png
  aligned_fft.gif
  aligned_fft_refine.gif
"""

import time
from pathlib import Path

from PIL import Image

from aligngif_core import Anchor, AlignConfig, align_frames
from debug import save_anchor_debug_image


# ----------------------------
# Paths
# ----------------------------
EXPORT_PATH = Path("export")
DATA_PATH = Path("data")

EXPORT_PATH.mkdir(exist_ok=True)


# ----------------------------
# Input frames
# ----------------------------
paths = [
    DATA_PATH / "frame1.png",
    DATA_PATH / "frame2.png",
    DATA_PATH / "frame3.png",
    DATA_PATH / "frame4.png",
]

images = [Image.open(p).convert("RGBA") for p in paths]


# ----------------------------
# Anchors (title + tree)
# ----------------------------
anchors = [
    Anchor(box=(650, 0, 1180, 170), weight=1.4, name="title"),
    Anchor(box=(380, 410, 800, 680), weight=1.0, name="tree"),
]


# ----------------------------
# Save anchor debug overlay
# ----------------------------
debug_img_path = EXPORT_PATH / "anchors_debug.png"
save_anchor_debug_image(images[0], anchors, debug_img_path)


# ----------------------------
# Output GIF settings
# ----------------------------
duration_ms = 800
loop = 0


def save_gif(frames, filename: str):
    """Save GIF into export folder."""
    out_path = EXPORT_PATH / filename

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        disposal=2,
    )
    print(f"✅ Saved: {out_path}")


def progress(frame_idx: int, total_frames: int, stage: int, best):
    """Progress callback."""
    if stage == 0:
        print(
            f"[frame {frame_idx}/{total_frames}] "
            f"FFT dx={best.dx:+d} dy={best.dy:+d} score={best.score:.4f}"
        )


# ----------------------------
# Run FFT
# ----------------------------
print("\n--- Running FFT ---")
cfg_fft = AlignConfig(method="fft", use_edges=True)

t0 = time.time()
aligned_fft = align_frames(images, anchors, cfg_fft, progress=progress)
print(f"FFT done in {time.time() - t0:.2f}s")

save_gif(aligned_fft, "aligned_fft.gif")


# ----------------------------
# Run FFT-refine
# ----------------------------
print("\n--- Running FFT-refine ---")
cfg_ref = AlignConfig(
    method="fft-refine",
    use_edges=True,
    refine_radius=6,
    refine_weighted_score=True,
)

t0 = time.time()
aligned_ref = align_frames(images, anchors, cfg_ref, progress=progress)
print(f"FFT-refine done in {time.time() - t0:.2f}s")

save_gif(aligned_ref, "aligned_fft_refine.gif")


print("\n✅ All outputs saved inside ./export/")
