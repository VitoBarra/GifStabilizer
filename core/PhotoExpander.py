# borderpad/pad.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

from PIL import Image

ImageLikePath = Union[str, Path]


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Padding:
    left: int
    top: int
    right: int
    bottom: int


def _is_image_path(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_EXTS


def _ensure_mode(img: Image.Image) -> Image.Image:
    # Keep alpha if present; otherwise go RGB.
    if img.mode in ("RGB", "RGBA"):
        return img
    if "A" in img.getbands():
        return img.convert("RGBA")
    return img.convert("RGB")


def compute_target_size_from_paths(paths: Sequence[Path]) -> Tuple[int, int]:
    max_w, max_h = 0, 0
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
        max_w = max(max_w, w)
        max_h = max(max_h, h)
    return max_w, max_h


def compute_center_padding(src_size: Tuple[int, int], target_size: Tuple[int, int]) -> Padding:
    w, h = src_size
    tw, th = target_size
    if tw < w or th < h:
        raise ValueError(f"target_size {target_size} must be >= src_size {src_size}")

    left = (tw - w) // 2
    right = tw - w - left
    top = (th - h) // 2
    bottom = th - h - top
    return Padding(left, top, right, bottom)


def pad_to_size(
    img: Image.Image,
    target_size: Tuple[int, int],
    *,
    mode: str = "replicate",
    anchor: str = "center",
) -> Image.Image:
    """
    Pad an image to target_size by extending edges.

    Parameters
    - img: PIL Image
    - target_size: (width, height)
    - mode: "replicate" or "mirror"
    - anchor: currently only "center" (kept for API evolution)

    Returns
    - New PIL Image with exact target_size.
    """
    if anchor != "center":
        raise ValueError("only anchor='center' is supported for now")

    img = _ensure_mode(img)
    w, h = img.size
    tw, th = target_size

    pad = compute_center_padding((w, h), (tw, th))
    out = Image.new(img.mode, (tw, th))
    out.paste(img, (pad.left, pad.top))

    # Fill top/bottom strips (over the pasted image)
    if pad.top > 0:
        strip = img.crop((0, 0, w, 1))
        if mode == "mirror":
            strip = strip.transpose(Image.FLIP_TOP_BOTTOM)
        strip = strip.resize((w, pad.top), resample=Image.NEAREST)
        out.paste(strip, (pad.left, 0))

    if pad.bottom > 0:
        strip = img.crop((0, h - 1, w, h))
        if mode == "mirror":
            strip = strip.transpose(Image.FLIP_TOP_BOTTOM)
        strip = strip.resize((w, pad.bottom), resample=Image.NEAREST)
        out.paste(strip, (pad.left, pad.top + h))

    # Fill left/right (use already padded canvas so corners are handled)
    if pad.left > 0:
        col = out.crop((pad.left, 0, pad.left + 1, th))
        if mode == "mirror":
            col = col.transpose(Image.FLIP_LEFT_RIGHT)
        col = col.resize((pad.left, th), resample=Image.NEAREST)
        out.paste(col, (0, 0))

    if pad.right > 0:
        col = out.crop((pad.left + w - 1, 0, pad.left + w, th))
        if mode == "mirror":
            col = col.transpose(Image.FLIP_LEFT_RIGHT)
        col = col.resize((pad.right, th), resample=Image.NEAREST)
        out.paste(col, (pad.left + w, 0))

    return out


def pad_folder_to_max_size(
    input_dir: ImageLikePath,
    output_dir: ImageLikePath,
    *,
    mode: str = "replicate",
    overwrite: bool = True,
    exts: Optional[Iterable[str]] = None,
    quality: int = 95,
) -> Tuple[int, int]:
    """
    Read all images in input_dir, compute (max_w, max_h), pad each to that size, save to output_dir.

    Returns (target_w, target_h).
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed = {e.lower() if e.startswith(".") else "." + e.lower() for e in (exts or SUPPORTED_EXTS)}
    paths = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed]
    if not paths:
        raise ValueError(f"no images found in {in_dir}")

    target = compute_target_size_from_paths(paths)

    for p in paths:
        out_path = out_dir / p.name
        if out_path.exists() and not overwrite:
            continue

        with Image.open(p) as im:
            im = _ensure_mode(im)
            padded = pad_to_size(im, target, mode=mode)

        save_kwargs = {}
        if out_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs.update(dict(quality=quality, subsampling=0, optimize=True))
            if padded.mode == "RGBA":
                bg = Image.new("RGB", padded.size, (255, 255, 255))
                bg.paste(padded, mask=padded.split()[-1])
                padded = bg

        padded.save(out_path, **save_kwargs)

    return target
