#!/usr/bin/env python3
"""
aligngif_gui.py — Generalized GUI for Alignment.py
- Unlimited anchors (boxes) with name + weight
- Methods: FFT / FFT-refine
- Drag & drop reordering of frames (top = reference frame)
- Reference frame is ALWAYS the first frame in the list (after reorder)
- Selection canvas ALWAYS shows the reference frame (frame 1)
- Preview Before/After stacked vertically
- Preview speed uses GIF duration
- Preview images are padded to FIXED panel size (NO UI jumping, NO growing on repeated Preview)
- Layout uses a horizontal PanedWindow: reference column ~2x preview column

Requires:
  pip install pillow numpy
"""

import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Tuple

from PIL import Image, ImageTk

from core.Alignment import align_frames
from core.DataModel import Box, Anchor, AlignConfig


@dataclass
class AnchorUI:
    name: str
    weight: float
    box: Optional[Box] = None
    color: str = "yellow"


def _cycle_color(i: int) -> str:
    colors = ["yellow", "lime", "cyan", "magenta", "orange", "red", "white"]
    return colors[i % len(colors)]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AlignGIF GUI")
        self.geometry("1600x900")

        # State
        self.paths: List[str] = []
        self.images: List[Image.Image] = []

        # Reference frame is always images[0]
        self.preview_img: Optional[Image.Image] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.scale = 1.0

        # Anchors
        self.anchors: List[AnchorUI] = []
        self.selected_anchor_idx: Optional[int] = None

        # Drag-selection of boxes on canvas
        self._drag_start = None
        self._current_rect_id = None

        # Frames list (drag & drop reorder)
        self.frames_listbox: Optional[tk.Listbox] = None
        self._frames_drag_from: Optional[int] = None

        # Preview animation state
        self.before_frames_tk: List[ImageTk.PhotoImage] = []
        self.after_frames_tk: List[ImageTk.PhotoImage] = []
        self._anim_job = None
        self._anim_idx = 0
        self._anim_delay_ms = 200

        # Fixed preview panel containers (to prevent UI jumping)
        self.before_frame: Optional[ttk.Frame] = None
        self.after_frame: Optional[ttk.Frame] = None

        # Freeze preview sizes ONCE (prevents "growing" when clicking Preview multiple times)
        self._preview_sizes_frozen = False
        self._before_box_wh: Optional[Tuple[int, int]] = None
        self._after_box_wh: Optional[Tuple[int, int]] = None

        # Paned layout
        self._paned: Optional[ttk.PanedWindow] = None

        self._build_ui()

    # ---------------- UI ----------------

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        # LEFT PANEL
        left = ttk.Frame(root, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(left, text="Load images…", command=self.load_images).pack(fill=tk.X)

        ttk.Label(left, text="Frames order (drag to reorder) — top = reference").pack(anchor="w", pady=(10, 0))

        self.frames_listbox = tk.Listbox(left, height=10)
        self.frames_listbox.pack(fill=tk.X, pady=(4, 10))

        self.frames_listbox.bind("<ButtonPress-1>", self.on_frames_drag_start)
        self.frames_listbox.bind("<B1-Motion>", self.on_frames_drag_motion)
        self.frames_listbox.bind("<ButtonRelease-1>", self.on_frames_drag_end)

        ttk.Separator(left).pack(fill=tk.X, pady=8)

        # Anchors section
        ttk.Label(left, text="Anchors").pack(anchor="w")

        anchors_bar = ttk.Frame(left)
        anchors_bar.pack(fill=tk.X, pady=(4, 4))
        ttk.Button(anchors_bar, text="Add", command=self.add_anchor).pack(side=tk.LEFT)
        ttk.Button(anchors_bar, text="Remove", command=self.remove_anchor).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(anchors_bar, text="Clear all", command=self.clear_anchors).pack(side=tk.LEFT, padx=(6, 0))

        self.anchor_list = tk.Listbox(left, height=7)
        self.anchor_list.pack(fill=tk.X, pady=(0, 6))
        self.anchor_list.bind("<<ListboxSelect>>", self.on_anchor_select)

        edit = ttk.Frame(left)
        edit.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(edit, text="Name").grid(row=0, column=0, sticky="w")
        self.anchor_name = tk.StringVar(value="")
        ttk.Entry(edit, textvariable=self.anchor_name, width=18).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        ttk.Label(edit, text="Weight").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.anchor_weight = tk.DoubleVar(value=1.0)
        ttk.Entry(edit, textvariable=self.anchor_weight, width=10).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(4, 0))

        ttk.Button(edit, text="Apply", command=self.apply_anchor_edits).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )
        edit.columnconfigure(1, weight=1)

        ttk.Label(left, text="Tip: select an anchor, then click-drag on the reference image to set its box.").pack(
            anchor="w", pady=(0, 12)
        )

        # Algorithm section
        ttk.Label(left, text="Algorithm").pack(anchor="w")
        self.method = tk.StringVar(value="fft-refine")
        ttk.Combobox(
            left,
            textvariable=self.method,
            values=["fft", "fft-refine"],
            state="readonly",
        ).pack(fill=tk.X, pady=(0, 10))

        self.use_edges = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Use edge detection", variable=self.use_edges).pack(anchor="w", pady=(0, 10))

        self.refine_radius = tk.IntVar(value=6)
        self._spin(left, "FFT refine radius", self.refine_radius, 0, 100)

        # Output & preview settings
        self.duration = tk.IntVar(value=800)
        self.loop = tk.IntVar(value=0)
        self.preview_max_side = tk.IntVar(value=520)  # compute-only downscale

        self._spin(left, "GIF duration (ms)", self.duration, 10, 10000)
        self._spin(left, "GIF loop (0=∞)", self.loop, 0, 1000)

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        ttk.Label(left, text="Preview settings").pack(anchor="w")
        self._spin(left, "Preview compute max side (px)", self.preview_max_side, 200, 2000)

        ttk.Button(left, text="Preview (Before/After)", command=self.build_preview).pack(fill=tk.X, pady=(10, 4))
        ttk.Button(left, text="Export aligned GIF…", command=self.export_gif).pack(fill=tk.X)

        self.progress = ttk.Progressbar(left, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=(10, 0))

        self.status = ttk.Label(left, text="", wraplength=360, justify="left")
        self.status.pack(fill=tk.X, pady=(6, 0))

        # RIGHT PANEL: reference + preview (PanedWindow)
        right = ttk.Frame(root, padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._paned = ttk.PanedWindow(right, orient=tk.HORIZONTAL)
        self._paned.pack(fill=tk.BOTH, expand=True)

        # Reference pane
        sel = ttk.Frame(self._paned, padding=0)
        self._paned.add(sel, weight=2)  # ~2x space

        ttk.Label(sel, text="Reference frame (frame 1). Select anchor boxes here.").pack(anchor="w")

        self.canvas = tk.Canvas(sel, background="#222")
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Preview pane
        prev = ttk.Frame(self._paned, padding=0)
        self._paned.add(prev, weight=1)

        ttk.Label(prev, text="Preview (Before / After)").pack(anchor="w")

        panels = ttk.Frame(prev)
        panels.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        ttk.Label(panels, text="Before", anchor="w").pack(fill=tk.X)

        self.before_frame = ttk.Frame(panels)
        self.before_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 10))
        self.before_frame.pack_propagate(False)

        self.before_label = ttk.Label(self.before_frame, text="(load images then preview)", anchor="center")
        self.before_label.pack(fill=tk.BOTH, expand=True)

        ttk.Label(panels, text="After", anchor="w").pack(fill=tk.X)

        self.after_frame = ttk.Frame(panels)
        self.after_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.after_frame.pack_propagate(False)

        self.after_label = ttk.Label(self.after_frame, text="(set anchors then preview)", anchor="center")
        self.after_label.pack(fill=tk.BOTH, expand=True)

        # Redraw selection on resize
        self.bind("<Configure>", lambda e: self.redraw_selection_preview())

        # Set initial sash so reference pane is ~double preview pane
        self.after(120, self._set_initial_sash)

        # Freeze preview panel sizes ONCE after layout settles
        self.after(200, self._freeze_preview_panel_sizes_once)

        # Defaults
        self.add_anchor(default_name="title", default_weight=1.4)
        self.add_anchor(default_name="tree", default_weight=1.0)
        self.anchor_list.selection_set(0)
        self.on_anchor_select()

    def _spin(self, parent, label, var, lo, hi):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label).pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=lo, to=hi, textvariable=var, width=8).pack(side=tk.RIGHT)

    def _set_initial_sash(self):
        if not self._paned:
            return
        try:
            w = max(600, self._paned.winfo_width())
            # left ~66%, right ~34%
            self._paned.sashpos(0, int(w * 0.66))
        except Exception:
            pass

    # ---------------- Preview panel freezing (ONE TIME) ----------------

    def _freeze_preview_panel_sizes_once(self):
        if self._preview_sizes_frozen:
            return
        if not self.before_frame or not self.after_frame:
            return

        self.update_idletasks()

        bw = max(280, self.before_frame.winfo_width())
        bh = max(220, self.before_frame.winfo_height())
        aw = max(280, self.after_frame.winfo_width())
        ah = max(220, self.after_frame.winfo_height())

        self.before_frame.config(width=bw, height=bh)
        self.after_frame.config(width=aw, height=ah)
        self.before_frame.pack_propagate(False)
        self.after_frame.pack_propagate(False)

        self._before_box_wh = (bw, bh)
        self._after_box_wh = (aw, ah)
        self._preview_sizes_frozen = True

    # ---------------- Frames (drag & drop reorder) ----------------

    def refresh_frames_list(self):
        if not self.frames_listbox:
            return
        self.frames_listbox.delete(0, tk.END)
        for i, p in enumerate(self.paths):
            base = p.split("/")[-1].split("\\")[-1]
            tag = "  [ref]" if i == 0 else ""
            self.frames_listbox.insert(tk.END, f"{i+1}. {base}{tag}")

        self.preview_img = self.images[0] if self.images else None
        self.redraw_selection_preview()

    def on_frames_drag_start(self, event):
        if not self.frames_listbox:
            return
        idx = self.frames_listbox.nearest(event.y)
        if idx < 0 or idx >= len(self.paths):
            self._frames_drag_from = None
            return
        self._frames_drag_from = idx
        self.frames_listbox.selection_clear(0, tk.END)
        self.frames_listbox.selection_set(idx)

    def on_frames_drag_motion(self, event):
        if not self.frames_listbox or self._frames_drag_from is None:
            return

        to_idx = self.frames_listbox.nearest(event.y)
        from_idx = self._frames_drag_from

        if to_idx == from_idx:
            return
        if to_idx < 0 or to_idx >= len(self.paths):
            return

        path = self.paths.pop(from_idx)
        img = self.images.pop(from_idx)
        self.paths.insert(to_idx, path)
        self.images.insert(to_idx, img)

        self._frames_drag_from = to_idx

        self.refresh_frames_list()
        self.frames_listbox.selection_clear(0, tk.END)
        self.frames_listbox.selection_set(to_idx)
        self.frames_listbox.activate(to_idx)

        if to_idx == 0 or from_idx == 0:
            self.status.config(text="Reference frame changed. Anchors are kept; verify they still match the new reference.")

    def on_frames_drag_end(self, event):
        self._frames_drag_from = None

    # ---------------- Anchors ----------------

    def add_anchor(self, default_name: Optional[str] = None, default_weight: float = 1.0):
        idx = len(self.anchors)
        name = default_name if default_name is not None else f"anchor{idx+1}"
        a = AnchorUI(name=name, weight=float(default_weight), box=None, color=_cycle_color(idx))
        self.anchors.append(a)
        self.refresh_anchor_list(select_idx=idx)

    def remove_anchor(self):
        idx = self.selected_anchor_idx
        if idx is None or idx < 0 or idx >= len(self.anchors):
            return
        del self.anchors[idx]
        for i, a in enumerate(self.anchors):
            a.color = _cycle_color(i)
        self.refresh_anchor_list(select_idx=min(idx, len(self.anchors) - 1))
        self.redraw_selection_preview()

    def clear_anchors(self):
        self.anchors.clear()
        self.selected_anchor_idx = None
        self.refresh_anchor_list(select_idx=None)
        self.redraw_selection_preview()

    def refresh_anchor_list(self, select_idx: Optional[int]):
        self.anchor_list.delete(0, tk.END)
        for i, a in enumerate(self.anchors):
            if a.box is None:
                coord_txt = "unset"
            else:
                x1, y1, x2, y2 = a.box
                coord_txt = f"({x1},{y1})-({x2},{y2})  {x2-x1}x{y2-y1}"
            self.anchor_list.insert(tk.END, f"{i+1}. {a.name}  w={a.weight:g}  {coord_txt}")

        if select_idx is not None and self.anchors:
            select_idx = max(0, min(select_idx, len(self.anchors) - 1))
            self.anchor_list.selection_clear(0, tk.END)
            self.anchor_list.selection_set(select_idx)
            self.anchor_list.activate(select_idx)
            self.selected_anchor_idx = select_idx
            self.anchor_name.set(self.anchors[select_idx].name)
            self.anchor_weight.set(self.anchors[select_idx].weight)
        else:
            self.selected_anchor_idx = None
            self.anchor_name.set("")
            self.anchor_weight.set(1.0)

    def on_anchor_select(self, event=None):
        sel = self.anchor_list.curselection()
        if not sel:
            self.selected_anchor_idx = None
            return
        idx = int(sel[0])
        self.selected_anchor_idx = idx
        a = self.anchors[idx]
        self.anchor_name.set(a.name)
        self.anchor_weight.set(a.weight)
        self.redraw_selection_preview()

    def apply_anchor_edits(self):
        idx = self.selected_anchor_idx
        if idx is None or idx < 0 or idx >= len(self.anchors):
            return
        self.anchors[idx].name = self.anchor_name.get().strip() or self.anchors[idx].name
        try:
            self.anchors[idx].weight = float(self.anchor_weight.get())
        except Exception:
            messagebox.showerror("Invalid weight", "Weight must be a number.")
            return
        self.refresh_anchor_list(select_idx=idx)
        self.redraw_selection_preview()

    # ---------------- Images / selection canvas ----------------

    def load_images(self):
        paths = filedialog.askopenfilenames(
            title="Select input frames",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return

        self.stop_animation()

        self.paths = list(paths)
        try:
            self.images = [Image.open(p).convert("RGBA") for p in self.paths]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load images:\n{e}")
            return

        self.preview_img = self.images[0]
        self.before_label.config(text="(click Preview)", image="")
        self.after_label.config(text="(click Preview)", image="")
        self.status.config(text="Images loaded. Drag frames to reorder (top is reference).")
        self.refresh_frames_list()

        # If user loads images after initial freeze, freeze again once (but still only once total)
        self.after(120, self._freeze_preview_panel_sizes_once)

    def redraw_selection_preview(self):
        if self.preview_img is None:
            return

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        iw, ih = self.preview_img.size
        s = min(cw / iw, ch / ih)
        s = max(0.05, min(1.5, s))
        self.scale = s

        disp = self.preview_img.resize((int(iw * s), int(ih * s)), Image.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(disp)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.preview_photo, anchor="nw")

        for i, a in enumerate(self.anchors):
            self._draw_annotated_box(a, i)

    def _draw_annotated_box(self, a: AnchorUI, idx: int):
        if not a.box:
            return
        x1, y1, x2, y2 = a.box
        s = self.scale
        sx1, sy1, sx2, sy2 = x1 * s, y1 * s, x2 * s, y2 * s

        outline = a.color
        width = 3 if idx == self.selected_anchor_idx else 2

        self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=outline, width=width)
        text = f"{idx+1}:{a.name}  w={a.weight:g}"
        self.canvas.create_rectangle(sx1, sy1, sx1 + 260, sy1 + 18, fill="#000000", outline="")
        self.canvas.create_text(
            sx1 + 4, sy1 + 2, anchor="nw", text=text, fill=outline, font=("TkDefaultFont", 9, "bold")
        )

    # ---------------- Mouse events ----------------

    def on_mouse_down(self, event):
        if self.preview_img is None:
            return
        if self.selected_anchor_idx is None:
            messagebox.showwarning("No anchor selected", "Select an anchor in the list first.")
            return
        self._drag_start = (event.x, event.y)
        if self._current_rect_id:
            self.canvas.delete(self._current_rect_id)
            self._current_rect_id = None

    def on_mouse_drag(self, event):
        if not self._drag_start or self.preview_img is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y

        if self._current_rect_id:
            self.canvas.delete(self._current_rect_id)

        idx = self.selected_anchor_idx
        color = self.anchors[idx].color if idx is not None else "yellow"
        self._current_rect_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2)

    def on_mouse_up(self, event):
        if not self._drag_start or self.preview_img is None:
            return
        if self.selected_anchor_idx is None:
            return

        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        self._drag_start = None

        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))

        s = self.scale
        if s <= 0:
            return

        ox0 = int(round(x0 / s))
        oy0 = int(round(y0 / s))
        ox1 = int(round(x1 / s))
        oy1 = int(round(y1 / s))

        iw, ih = self.preview_img.size
        ox0 = max(0, min(iw - 1, ox0))
        oy0 = max(0, min(ih - 1, oy0))
        ox1 = max(0, min(iw, ox1))
        oy1 = max(0, min(ih, oy1))

        if abs(ox1 - ox0) < 5 or abs(oy1 - oy0) < 5:
            return

        self.anchors[self.selected_anchor_idx].box = (ox0, oy0, ox1, oy1)
        self.refresh_anchor_list(select_idx=self.selected_anchor_idx)
        self.redraw_selection_preview()

    # ---------------- Preview / Export ----------------

    def _compute_downscale_factor(self, im: Image.Image, max_side: int) -> float:
        w, h = im.size
        return min(1.0, max_side / max(w, h))

    def _scale_box(self, box: Box, s: float) -> Box:
        x1, y1, x2, y2 = box
        return (int(round(x1 * s)), int(round(y1 * s)), int(round(x2 * s)), int(round(y2 * s)))

    def _build_core_anchors(self) -> List[Anchor]:
        core = []
        for a in self.anchors:
            if a.box is None or a.weight <= 0:
                continue
            core.append(Anchor(box=a.box, weight=float(a.weight), name=a.name))
        return core

    def _validate_ready(self) -> bool:
        if not self.images:
            messagebox.showwarning("Missing input", "Load images first.")
            return False
        if not self._build_core_anchors():
            messagebox.showwarning("Missing anchors", "Create at least one anchor box with positive weight.")
            return False
        return True

    def _fit_to_box_fixed(self, im: Image.Image, W: int, H: int) -> Image.Image:
        """Always returns an image of EXACT size (W,H), letterboxed, with white background."""
        if W <= 0 or H <= 0:
            return im
        im = im.convert("RGBA")
        w, h = im.size

        s = min(W / w, H / h)
        s = min(1.0, s)  # never upscale
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))

        resized = im.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        canvas.paste(resized, ((W - nw) // 2, (H - nh) // 2))
        return canvas

    def build_preview(self):
        if not self._validate_ready():
            return

        # Freeze preview sizes ONCE (guards the "growing" issue)
        self._freeze_preview_panel_sizes_once()

        self.stop_animation()
        self.progress.start(10)
        self.status.config(text="Building preview (before/after)...")
        self._set_controls_enabled(False)

        self._anim_delay_ms = max(10, int(self.duration.get()))
        max_side = int(self.preview_max_side.get())

        cfg = AlignConfig(
            method=self.method.get(),
            use_edges=bool(self.use_edges.get()),
            bg=(255, 255, 255, 255),
            refine_radius=int(self.refine_radius.get()),
            refine_weighted_score=True,
        )

        def worker():
            try:
                s_comp = self._compute_downscale_factor(self.images[0], max_side)

                def downscale_for_compute(im: Image.Image) -> Image.Image:
                    if s_comp >= 1.0:
                        return im
                    w, h = im.size
                    return im.resize((int(round(w * s_comp)), int(round(h * s_comp))), Image.LANCZOS)

                small_imgs = [downscale_for_compute(im) for im in self.images]
                before = small_imgs

                preview_anchors = [
                    Anchor(box=self._scale_box(a.box, s_comp), weight=a.weight, name=a.name)
                    for a in self._build_core_anchors()
                ]

                after = align_frames(small_imgs, preview_anchors, cfg)
                self._ui_preview_ready(before, after)
            except Exception as e:
                self._ui_done_err(str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _ui_preview_ready(self, before_pil, after_pil):
        def done():
            self.progress.stop()
            self.status.config(text=f"Preview ready. (delay={self._anim_delay_ms}ms)")
            self._set_controls_enabled(True)

            # Use frozen sizes (so repeated Preview never changes anything)
            self._freeze_preview_panel_sizes_once()
            bw, bh = self._before_box_wh or (420, 300)
            aw, ah = self._after_box_wh or (420, 300)

            before_fit = [self._fit_to_box_fixed(im, bw, bh) for im in before_pil]
            after_fit = [self._fit_to_box_fixed(im, aw, ah) for im in after_pil]

            # Replace frames (no accumulation)
            self.before_frames_tk = [ImageTk.PhotoImage(im) for im in before_fit]
            self.after_frames_tk = [ImageTk.PhotoImage(im) for im in after_fit]

            self._anim_idx = 0
            self._tick_animation()

        self.after(0, done)

    def _tick_animation(self):
        if not self.before_frames_tk or not self.after_frames_tk:
            return
        i = self._anim_idx % len(self.before_frames_tk)
        self.before_label.config(image=self.before_frames_tk[i], text="")
        self.after_label.config(image=self.after_frames_tk[i], text="")
        self._anim_idx += 1
        self._anim_job = self.after(self._anim_delay_ms, self._tick_animation)

    def stop_animation(self):
        if self._anim_job is not None:
            try:
                self.after_cancel(self._anim_job)
            except Exception:
                pass
        self._anim_job = None
        self.before_frames_tk = []
        self.after_frames_tk = []
        self._anim_idx = 0

    def export_gif(self):
        if not self._validate_ready():
            return

        out_path = filedialog.asksaveasfilename(
            title="Save GIF",
            defaultextension=".gif",
            filetypes=[("GIF", "*.gif")],
        )
        if not out_path:
            return

        self.progress.start(10)
        self.status.config(text="Running alignment + exporting GIF...")
        self._set_controls_enabled(False)

        cfg = AlignConfig(
            method=self.method.get(),
            use_edges=bool(self.use_edges.get()),
            bg=(255, 255, 255, 255),
            refine_radius=int(self.refine_radius.get()),
            refine_weighted_score=True,
        )
        anchors = self._build_core_anchors()

        def worker():
            try:
                aligned = align_frames(self.images, anchors, cfg)
                aligned[0].save(
                    out_path,
                    save_all=True,
                    append_images=aligned[1:],
                    duration=int(self.duration.get()),
                    loop=int(self.loop.get()),
                    disposal=2,
                )
                self._ui_done_ok(f"Saved: {out_path}")
            except Exception as e:
                self._ui_done_err(str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _ui_done_ok(self, msg: str):
        def done():
            self.progress.stop()
            self.status.config(text=msg)
            self._set_controls_enabled(True)
            messagebox.showinfo("Done", msg)
        self.after(0, done)

    def _ui_done_err(self, err: str):
        def done():
            self.progress.stop()
            self.status.config(text="Error.")
            self._set_controls_enabled(True)
            messagebox.showerror("Error", err)
        self.after(0, done)

    # ---------------- Enable/disable controls ----------------

    def _set_controls_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for child in self.winfo_children():
            self._set_state_recursive(child, state)
        self.canvas.configure(state="normal")
        if self.frames_listbox:
            self.frames_listbox.configure(state="normal")

    def _set_state_recursive(self, widget, state: str):
        try:
            widget.configure(state=state)
        except Exception:
            pass
        for c in widget.winfo_children():
            self._set_state_recursive(c, state)


if __name__ == "__main__":
    App().mainloop()
