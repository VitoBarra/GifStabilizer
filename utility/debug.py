from typing import List

from PIL import ImageDraw, ImageFont, Image

from core.DataModel import Anchor, Box


from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from core.DataModel import Anchor


def save_debug_image(
    image: Image.Image,
    anchors: List[Anchor],
    out_path: str = "debug.png",
    max_draw: Optional[int] = None,
    title: str = "",
):
    """
    Save an image with Anchor boxes drawn on top.

    This works for:
      - manually defined anchors
      - automatically suggested anchors
      - debugging alignment regions

    Parameters
    ----------
    image:
        Reference frame image.
    anchors:
        List of Anchor objects.
    out_path:
        Output PNG filename.
    max_draw:
        Draw only the first N anchors (default: all).
    title:
        Optional prefix shown in labels.
    """

    img = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)

    # Default font (portable)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    colors = ["yellow", "lime", "cyan", "magenta", "orange", "red", "white"]

    if max_draw is None:
        max_draw = len(anchors)

    for i, a in enumerate(anchors[:max_draw]):
        x1, y1, x2, y2 = a.box
        color = colors[i % len(colors)]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label text
        name = a.name or f"anchor{i+1}"
        prefix = f"{title} " if title else ""
        label = f"{prefix}{name}  w={a.weight:g}"

        tx, ty = x1 + 4, y1 + 4

        # Background behind label
        draw.rectangle(
            [tx - 2, ty - 2, tx + 220, ty + 16],
            fill=(0, 0, 0, 180),
        )

        # Label
        draw.text((tx, ty), label, fill=color, font=font)

    img.save(out_path)
    print(f"✅ Debug image saved: {out_path}")
