#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image

from aligngif_core import Anchor, AlignConfig, align_frames


def parse_box(vals):
    x1, y1, x2, y2 = map(int, vals)
    return (x1, y1, x2, y2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--duration", type=int, default=800)
    p.add_argument("--loop", type=int, default=0)

    p.add_argument("--method", choices=["fft", "fft-refine"], default="fft-refine")
    p.add_argument("--refine-radius", type=int, default=6)
    p.add_argument("--no-edges", action="store_true")

    # repeatable anchors:
    # --anchor "x1 y1 x2 y2 weight name"
    p.add_argument(
        "--anchor",
        action="append",
        default=[],
        help='Repeatable. Format: "x1 y1 x2 y2 weight name". name optional.',
    )

    args = p.parse_args()

    images = [Image.open(f).convert("RGBA") for f in args.frames]

    anchors = []
    for a in args.anchor:
        parts = a.split()
        if len(parts) not in (5, 6):
            raise SystemExit('Bad --anchor. Use: "x1 y1 x2 y2 weight [name]"')
        box = parse_box(parts[:4])
        weight = float(parts[4])
        name = parts[5] if len(parts) == 6 else ""
        anchors.append(Anchor(box=box, weight=weight, name=name))

    cfg = AlignConfig(
        method=args.method,
        use_edges=not args.no_edges,
        refine_radius=args.refine_radius,
    )

    aligned = align_frames(images, anchors, cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aligned[0].save(
        out_path,
        save_all=True,
        append_images=aligned[1:],
        duration=args.duration,
        loop=args.loop,
        disposal=2,
    )
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
