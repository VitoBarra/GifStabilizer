from PIL import ImageDraw, ImageFont, Image

from aligngif_core import Anchor


def save_anchor_debug_image(
    image: Image.Image,
    anchors: list[Anchor],
    out_path: str = "anchors_debug.png",
):
    """
    Saves a copy of the image with anchor boxes drawn on top.

    Each anchor will show:
      - rectangle outline
      - name
      - weight
    """

    img = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)

    # Try default font (works everywhere)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    for i, a in enumerate(anchors):
        x1, y1, x2, y2 = a.box

        # Choose color (cycled)
        colors = ["yellow", "lime", "cyan", "red", "magenta", "orange"]
        color = colors[i % len(colors)]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label text
        label = f"{a.name or f'anchor{i+1}'}  w={a.weight}"

        # Background box behind text
        text_x, text_y = x1 + 4, y1 + 4
        draw.rectangle(
            [text_x - 2, text_y - 2, text_x + 160, text_y + 16],
            fill=(0, 0, 0, 180),
        )

        # Text itself
        draw.text((text_x, text_y), label, fill=color, font=font)

    img.save(out_path)
    print(f"✅ Anchor debug image saved: {out_path}")
