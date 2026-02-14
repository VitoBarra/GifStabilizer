from pathlib import Path

from PIL import Image

DATA_PATH = Path("raw_data")

# ----------------------------
# Input frames
# ----------------------------
paths = [
    DATA_PATH / "frame1.png",
    DATA_PATH / "frame2.png",
    DATA_PATH / "frame3.png",
    DATA_PATH / "frame4.png",
]
# ----------------------------
# Load frames
# ----------------------------
frames = [Image.open(p).convert("RGBA")for p in paths]

