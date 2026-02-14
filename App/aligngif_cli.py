#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image

from core.Alignment import align_frames
from core.DataModel import AlignConfig, Anchor


def parse_box(vals):
    x1, y1, x2, y2 = map(int, vals)
    return (x1, y1, x2, y2)


def parse_size(s: str):
    # "WIDTHxHEIGHT" (also accepts "WIDTH,HEIGHT")
    s = s.lower().replace(",", "x").strip()
    if "x" not in s:
        raise argparse.ArgumentTypeError('Size must be like "800x600"')
    w, h = s.split("x", 1)
    return (int(w), int(h))


def pad_to_size_edge_replicate(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Pad by extending edge pixels to reach target size (centered)."""
    w, h = img.size
    if target_w < w or target_h < h:
        raise ValueError(f"target {target_w}x{target_h} is smaller than image {w}x{h}")

    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top

    out = Image.new(img.mode, (target_w, target_h))
    out.paste(img, (pad_left, pad_top))

    # top/bottom
    if pad_top > 0:
        top = img.crop((0, 0, w, 1)).resize((w, pad_top), Image.NEAREST)
        out.paste(top, (pad_left, 0))
    if pad_bottom > 0:
        bottom = img.crop((0, h - 1, w, h)).resize((w, pad_bottom), Image.NEAREST)
        out.paste(bottom, (pad_left, pad_top + h))

    # left/right (include corners by sampling from already-padded canvas)
    if pad_left > 0:
        left = out.crop((pad_left, 0, pad_left + 1, target_h)).resize((pad_left, target_h), Image.NEAREST)
        out.paste(left, (0, 0))
    if pad_right > 0:
        right = out.crop((pad_left + w - 1, 0, pad_left + w, target_h)).resize((pad_right, target_h), Image.NEAREST)
        out.paste(right, (pad_left + w, 0))

    return out


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

    # padding (optional)
    p.add_argument(
        "--pad",
        choices=["none", "max"],
        default="none",
        help='Padding after alignment. "max" pads all frames to the max width/height found.',
    )
    p.add_argument(
        "--pad-size",
        type=parse_size,
        default=None,
        help='Optional explicit target size like "800x600". Overrides --pad=max.',
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

    # --- optional padding step ---
    if args.pad_size is not None:
        tw, th = args.pad_size
        aligned = [pad_to_size_edge_replicate(im, tw, th) for im in aligned]
    elif args.pad == "max":
        tw = max(im.size[0] for im in aligned)
        th = max(im.size[1] for im in aligned)
        aligned = [pad_to_size_edge_replicate(im, tw, th) for im in aligned]
    # -----------------------------

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
