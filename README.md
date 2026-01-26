# GifAligner (AlignGIF)

A comprehensive Python toolkit to **align a sequence of image frames** (and export a GIF) using **anchor regions** (ROIs).
It includes:

- a **core alignment library** (FFT / FFT-refine)
- a **Tkinter GUI** to load frames, reorder them, draw anchors, preview before/after, and export
- a **box suggestion module** that proposes stable/high-detail anchor boxes across frames
- a **photo expander module** for intelligent image padding and resizing
- a **command-line interface (CLI)** for batch processing with flexible anchor configuration
- **example projects** demonstrating alignment workflows on real data

---

## Features

### Alignment (core)
- Align frames relative to a **reference frame** (frame #1)
- Multiple **anchors**:
  - each anchor = `box (x1,y1,x2,y2) + weight + name`
  - alignment score is computed over the anchor regions
- Methods:
  - `fft` (fast translation estimate)
  - `fft-refine` (FFT + local refinement)
- Optional edge-based preprocessing (`use_edges=True`) for more robust correlation

### GUI (`App/aligngif_gui.py`)
- Load multiple frames (png/jpg/webp/…)
- **Drag & drop reorder** frames (top = reference)
- Add/remove anchors, edit name/weight
- Draw anchor boxes on the reference frame (click-drag)
- Suggest anchor boxes for the selected anchor (configurable)
- Preview Before/After (stacked) with stable panel layout (no UI jumping)
- Export aligned GIF

### Box Suggestion (`core/BoxSuggestion.py`)
Suggests candidate ROI boxes that are:
- high structure/detail (edges)
- stable over time (low variance across frames)

### Photo Expander (`core/PhotoExpander.py`)
Provides intelligent padding utilities:
- **Center-padding** — Smart alignment of smaller images within larger canvas
- **Background modes** — solid color, edge-replication, or transparency
- **Format preservation** — Maintains RGB/RGBA modes appropriately

---

## Requirements

- Python 3.10+ recommended (works on newer versions too)
- Dependencies:
  - `pillow`
  - `numpy`

Install:

```bash
pip install -r requirements.txt
```

---

## Run the GUI

Start the graphical interface with:

```bash
python App/aligngif_gui.py
```

### Quick Workflow

1. Click **Load images…**  
   Select all frames you want to align (PNG/JPG/WEBP/…).

2. (Optional) Reorder frames  
   Drag frames inside the **Frames order** list.  
   - The **top frame is always the reference frame**.

3. Create anchors  
   In the **Anchors** section:
   - Click **Add** to create a new anchor
   - Select it in the list
   - Set its **Name** and **Weight**
   - Click **Apply**

4. Draw anchor boxes  
   With an anchor selected, draw its ROI box directly on the reference image:
   - click + drag on the canvas
   - release to confirm the box

5. (Optional) Use box suggestion  
   Under **Suggest box (for selected anchor)**:
   - Tune region/size/stride/stability settings
   - Click **Suggest** to apply the best proposal
   - Click **Next** to cycle through alternative suggestions

6. Preview alignment  
   Click **Preview (Before/After)**  
   - The GUI will show stacked animation:
     - **Before** (original)
     - **After** (aligned)

7. Export GIF  
   Click **Export aligned GIF…**  
   Choose an output filename and the aligned animation will be saved.

---

## Run the CLI

For **batch processing** without the GUI, use the command-line interface:

```bash
python App/aligngif_cli.py <frame1> <frame2> ... <frameN> --out <output.gif> [options]
```

### CLI Options

**Core alignment:**
- `--method {fft, fft-refine}` — Alignment method (default: `fft-refine`)
- `--refine-radius N` — Refinement radius in pixels (default: `6`)
- `--no-edges` — Disable edge-based preprocessing (default: edges are used)

**Anchors:**
- `--anchor "x1 y1 x2 y2 weight [name]"` — Define an anchor (repeatable)
  - `x1, y1, x2, y2` — Bounding box coordinates
  - `weight` — Importance weight for this anchor
  - `name` — Optional anchor name for debugging

**Padding** (applied after alignment):
- `--pad {none, max}` — Padding strategy (default: `none`)
  - `none` — No padding
  - `max` — Pad all frames to max width/height found
- `--pad-size WIDTHxHEIGHT` — Explicit target size (e.g., `800x600`), overrides `--pad`

**GIF export:**
- `--duration MS` — Frame duration in milliseconds (default: `800`)
- `--loop N` — Loop count (default: `0` = infinite)

### CLI Example

```bash
python App/aligngif_cli.py frame1.png frame2.png frame3.png \
  --out aligned.gif \
  --method fft-refine \
  --anchor "40 30 400 640 1.4 main_box" \
  --pad max \
  --duration 500
```

---

## Photo Expander Module

The **PhotoExpander** module provides intelligent padding utilities for images programmatically:

```python
from core.PhotoExpander import pad_folder_to_max_size, compute_center_padding

# Pad all images in a folder to max size found, centering them
pad_folder_to_max_size(
    input_dir="raw_frames/",
    output_dir="padded_frames/",
    bg_color=(0, 0, 0)
)

# Compute padding needed to fit source to target size
padding = compute_center_padding(
    src_size=(640, 480),
    target_size=(800, 600)
)
```

Features:
- **Center-padding** — Smart alignment of smaller images within larger canvas
- **Background modes** — solid color, edge-replication, or transparency
- **Format preservation** — Maintains RGB/RGBA modes appropriately

---

## Example Projects

The `example/` directory contains two working projects demonstrating the toolkit in action:

### Tree Example

Located in `example/Tree/`:
- **alignment_TreeExample.py** — Basic alignment workflow
- **boxSuggestion_TreeExample.py** — Using the box suggestion feature to propose anchor regions
- **raw_data/** — Frame images for testing
- **ReferenceUnaligned.gif** — Reference unaligned animation

Run the Tree example:
```bash
python -m example.Tree.alignment_TreeExample
python -m example.Tree.boxSuggestion_TreeExample
```

### HWT Example

Located in `example/HWT/`:
- **alignment_HWT.py** — Demonstrates padding and alignment on multiple frame formats
- **raw_data/** — 8 image frames for testing (various sizes)
- **ReferenceUnaligned.gif** — Reference unaligned animation

Run the HWT example:
```bash
python -m example.HWT.alignment_HWT
```

This example shows:
- Alignment with custom anchors
- Padding frames after alignment
- Exporting results to GIF

---

## Demo: Before vs After Alignment

Here is a simple demonstration of what **GifAligner** can achieve.

### Before vs After Alignment

**Unaligned frames** (original input):

![unaligned GIF](Docs/Tree_unaligned.gif)

**Aligned frames** (after running the tool):

![aligned GIF](Docs/Tree_aligned.gif)

As you can see, the content becomes visually stable once the anchor-based alignment is applied.

