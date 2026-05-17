"""
example_main.py — run alignment directly from code (no CLI args)

Runs two variants:
  1) Non-padded frames from DATA_PATH
  2) Padded frames from PADDED_PATH

Produces in ./export/ :
  anchors_debug_noPad.png
  anchors_debug_pad.png
  aligned_fft_refine_NoPadded.gif
  aligned_fft_refine_padded.gif
"""

import time
from typing import List, Tuple

from PIL import Image

from core.Alignment import align_frames
from core.DataModel import AlignConfig, Anchor
from core.PhotoExpander import pad_folder_to_max_size
from example.HWT import ALIG_EXPORT_PATH, DATA_PATH, PADDED_PATH
from utility.debug import save_debug_image




# ----------------------------
# Shared settings
# ----------------------------
duration_ms = 800
loop = 0

anchors = [
    Anchor(box=(40, 30, 400, 640), weight=1.4, name="box"),
]

cfg_ref = AlignConfig(
    method="fft-refine",
    use_edges=True,
    refine_radius=6,
    refine_weighted_score=True,
)


def save_gif(frames: List[Image.Image], filename: str) -> None:
    """Save GIF into export folder."""
    out_path = ALIG_EXPORT_PATH / filename
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        disposal=2,
    )
    print(f"✅ Saved: {out_path}")


def progress(frame_idx: int, total_frames: int, stage: int, best) -> None:
    """Progress callback."""
    if stage == 0:
        print(
            f"[frame {frame_idx}/{total_frames}] "
            f"FFT dx={best.dx:+d} dy={best.dy:+d} score={best.score:.4f}"
        )


def load_frames(base_path, n: int = 8) -> List[Image.Image]:
    paths = [base_path / f"frame{i}.png" for i in range(1, n + 1)]
    return [Image.open(p).convert("RGBA") for p in paths]


def run_variant(
    *,
    label: str,
    base_path,
    debug_filename: str,
    gif_filename: str,
) -> None:
    print(f"\n--- Running FFT-refine ({label}) ---")

    frames = load_frames(base_path, n=8)

    # Anchor debug overlay
    debug_img_path = ALIG_EXPORT_PATH / debug_filename
    save_debug_image(frames[0], anchors, debug_img_path)
    print(f"✅ Saved: {debug_img_path}")

    # Alignment
    t0 = time.time()
    aligned_ref = align_frames(frames, anchors, cfg_ref, progress=progress)
    print(f"FFT-refine ({label}) done in {time.time() - t0:.2f}s")

    # Output
    save_gif(aligned_ref, gif_filename)


def main() -> None:
    run_variant(
        label="no padding",
        base_path=DATA_PATH,
        debug_filename="anchors_debug_noPad.png",
        gif_filename="aligned_fft_refine_NoPadded.gif",
    )

    # create padded data
    tw, th = pad_folder_to_max_size(DATA_PATH, PADDED_PATH, mode="replicate")
    print(f"\nall data padded to:({tw}, {th})")

    run_variant(
        label="padded",
        base_path=PADDED_PATH,
        debug_filename="anchors_debug_pad.png",
        gif_filename="aligned_fft_refine_padded.gif",
    )

    print("done")


if __name__ == "__main__":
    main()
