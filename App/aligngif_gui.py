#!/usr/bin/env python3
"""
aligngif_gui.py — Generalized GUI for Alignment.py (+ Anchor suggestion)
LEFT PANEL IS SCROLLABLE (so Export button is always reachable)

Features
- Drag & drop reorder of frames (top = reference frame)
- Unlimited anchors (name + weight + box)
- Suggestion tool per anchor (config is SAVED per-anchor and restored on reselect)
- Methods: FFT / FFT-refine
- Before/After preview stacked vertically
- Preview speed uses GIF duration
- Preview panels are fixed-size, letterboxed (no UI jumping, no growing on repeated Preview)

Requires:
  pip install pillow numpy
"""

import threading
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Tuple

from PIL import Image, ImageTk

from core.Alignment import align_frames
from core.BoxSuggestion import SuggestConfig, suggest_anchor_boxes
from core.DataModel import Box, Anchor, AlignConfig


# -----------------------------
# Data models (GUI side)
# -----------------------------

@dataclass
class SuggestUIState:
    region: str = "full"   # "full" means no constraint
    box_w: int = 520
    box_h: int = 170
    stride: int = 24
    topk: int = 8
    lam: float = 2.5
    use_edges: bool = True


@dataclass
class AnchorUI:
    name: str
    weight: float
    box: Optional[Box] = None
    color: str = "yellow"
    suggest: SuggestUIState = field(default_factory=SuggestUIState)


def _cycle_color(i: int) -> str:
    colors = ["yellow", "lime", "cyan", "magenta", "orange", "red", "white"]
    return colors[i % len(colors)]


# -----------------------------
# App
# -----------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AlignGIF GUI")
        self.geometry("1600x900")

        # Frames
        self.paths: List[str] = []
        self.images: List[Image.Image] = []

        # Reference frame (always images[0])
        self.preview_img: Optional[Image.Image] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.scale = 1.0

        # Anchors
        self.anchors: List[AnchorUI] = []
        self.selected_anchor_idx: Optional[int] = None

        # Canvas drag box
        self._drag_start = None
        self._current_rect_id = None

        # Frames list reorder (drag & drop)
        self.frames_listbox: Optional[tk.Listbox] = None
        self._frames_drag_from: Optional[int] = None

        # Preview animation state
        self.before_frames_tk: List[ImageTk.PhotoImage] = []
        self.after_frames_tk: List[ImageTk.PhotoImage] = []
        self._anim_job = None
        self._anim_idx = 0
        self._anim_delay_ms = 200

        # Fixed preview containers (prevent UI jumping)
        self.before_frame: Optional[ttk.Frame] = None
        self.after_frame: Optional[ttk.Frame] = None
        self._preview_sizes_frozen = False
        self._before_box_wh: Optional[Tuple[int, int]] = None
        self._after_box_wh: Optional[Tuple[int, int]] = None

        # Paned layout
        self._paned: Optional[ttk.PanedWindow] = None

        # Suggestion UI variables (persisted per-anchor)
        self.suggest_region = tk.StringVar(value="full")
        self.suggest_box_w = tk.IntVar(value=520)
        self.suggest_box_h = tk.IntVar(value=170)
        self.suggest_stride = tk.IntVar(value=24)
        self.suggest_topk = tk.IntVar(value=8)
        self.suggest_lambda = tk.DoubleVar(value=2.5)
        self.suggest_use_edges = tk.BooleanVar(value=True)
        self._suggest_index = 0

        # Algorithm + export vars
        self.method = tk.StringVar(value="fft-refine")
        self.use_edges = tk.BooleanVar(value=True)
        self.refine_radius = tk.IntVar(value=6)
        self.duration = tk.IntVar(value=800)
        self.loop = tk.IntVar(value=0)
        self.preview_max_side = tk.IntVar(value=520)

        # Scroll refs
        self._left_canvas: Optional[tk.Canvas] = None
        self._left_inner: Optional[ttk.Frame] = None
        self._left_scrollbar: Optional[ttk.Scrollbar] = None

        # Suggest persistence guard
        self._loading_suggest_to_ui = False

        # UI widgets we reference later
        self.anchor_list: Optional[tk.Listbox] = None
        self.anchor_name: Optional[tk.StringVar] = None
        self.anchor_weight: Optional[tk.DoubleVar] = None
        self.canvas: Optional[tk.Canvas] = None
        self.progress: Optional[ttk.Progressbar] = None
        self.status: Optional[ttk.Label] = None
        self.before_label: Optional[ttk.Label] = None
        self.after_label: Optional[ttk.Label] = None

        self._build_ui()
        self._bind_suggest_var_traces()

    # ---------------- UI ----------------

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        # ============================
        # LEFT PANEL (SCROLLABLE)
        # ============================
        left_container = ttk.Frame(root, padding=0)
        left_container.pack(side=tk.LEFT, fill=tk.Y)

        self._left_canvas = tk.Canvas(left_container, highlightthickness=0)
        self._left_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)

        self._left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=self._left_canvas.yview)
        self._left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._left_canvas.configure(yscrollcommand=self._left_scrollbar.set)

        self._left_inner = ttk.Frame(self._left_canvas, padding=10)
        left_window_id = self._left_canvas.create_window((0, 0), window=self._left_inner, anchor="nw")

        def _sync_inner_width(event):
            if self._left_canvas:
                self._left_canvas.itemconfigure(left_window_id, width=event.width)

        self._left_canvas.bind("<Configure>", _sync_inner_width)

        def _update_scrollregion(_event=None):
            if self._left_canvas:
                self._left_canvas.configure(scrollregion=self._left_canvas.bbox("all"))

        self._left_inner.bind("<Configure>", _update_scrollregion)
        self._install_mousewheel_scrolling(self._left_canvas)

        left = self._left_inner

        ttk.Button(left, text="Load images…", command=self.load_images).pack(fill=tk.X)

        ttk.Label(left, text="Frames order (drag to reorder) — top = reference").pack(anchor="w", pady=(10, 0))
        self.frames_listbox = tk.Listbox(left, height=10, exportselection=False)
        self.frames_listbox.pack(fill=tk.X, pady=(4, 10))
        self.frames_listbox.bind("<ButtonPress-1>", self.on_frames_drag_start)
        self.frames_listbox.bind("<B1-Motion>", self.on_frames_drag_motion)
        self.frames_listbox.bind("<ButtonRelease-1>", self.on_frames_drag_end)

        ttk.Separator(left).pack(fill=tk.X, pady=8)

        # Anchors
        ttk.Label(left, text="Anchors").pack(anchor="w")

        anchors_bar = ttk.Frame(left)
        anchors_bar.pack(fill=tk.X, pady=(4, 4))
        ttk.Button(anchors_bar, text="Add", command=self.add_anchor).pack(side=tk.LEFT)
        ttk.Button(anchors_bar, text="Remove", command=self.remove_anchor).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(anchors_bar, text="Clear all", command=self.clear_anchors).pack(side=tk.LEFT, padx=(6, 0))

        self.anchor_list = tk.Listbox(left, height=7, exportselection=False)
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

        ttk.Button(edit, text="Apply", command=self.apply_anchor_edits).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        edit.columnconfigure(1, weight=1)

        ttk.Label(left, text="Tip: select an anchor, then click-drag on the reference image to set its box.").pack(anchor="w", pady=(0, 8))

        # Suggestion
        ttk.Label(left, text="Suggest box (for selected anchor)").pack(anchor="w", pady=(6, 0))

        sugg = ttk.LabelFrame(left, text="Suggestion settings", padding=8)
        sugg.pack(fill=tk.X, pady=(6, 10))

        row = ttk.Frame(sugg); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Region").pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=self.suggest_region,
            values=["full", "top", "bottom", "left", "right", "top-left", "top-right", "bottom-left", "bottom-right", "center"],
            state="readonly",
            width=14,
        ).pack(side=tk.RIGHT)

        row = ttk.Frame(sugg); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Box W").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=40, to=4000, textvariable=self.suggest_box_w, width=8).pack(side=tk.RIGHT)

        row = ttk.Frame(sugg); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Box H").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=40, to=4000, textvariable=self.suggest_box_h, width=8).pack(side=tk.RIGHT)

        row = ttk.Frame(sugg); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Stride").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=4, to=512, textvariable=self.suggest_stride, width=8).pack(side=tk.RIGHT)

        row = ttk.Frame(sugg); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="TopK").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=1, to=50, textvariable=self.suggest_topk, width=8).pack(side=tk.RIGHT)

        row = ttk.Frame(sugg); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="λ stability").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.suggest_lambda, width=10).pack(side=tk.RIGHT)

        ttk.Checkbutton(sugg, text="Use edges", variable=self.suggest_use_edges).pack(anchor="w", pady=(4, 2))

        btns = ttk.Frame(sugg); btns.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(btns, text="Suggest", command=self.suggest_box_for_selected_anchor).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btns, text="Next", command=self.next_suggestion_for_selected_anchor).pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True)

        # Algorithm
        ttk.Label(left, text="Algorithm").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.method, values=["fft", "fft-refine"], state="readonly").pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(left, text="Use edge detection", variable=self.use_edges).pack(anchor="w", pady=(0, 10))

        self._spin(left, "FFT refine radius", self.refine_radius, 0, 100)

        # Output / Preview settings
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

        # ============================
        # RIGHT PANEL
        # ============================
        right = ttk.Frame(root, padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._paned = ttk.PanedWindow(right, orient=tk.HORIZONTAL)
        self._paned.pack(fill=tk.BOTH, expand=True)

        sel = ttk.Frame(self._paned, padding=0)
        self._paned.add(sel, weight=2)
        ttk.Label(sel, text="Reference frame (frame 1). Select anchor boxes here.").pack(anchor="w")

        self.canvas = tk.Canvas(sel, background="#222")
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

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

        self.bind("<Configure>", lambda e: self.redraw_selection_preview())
        self.after(120, self._set_initial_sash)
        self.after(200, self._freeze_preview_panel_sizes_once)

        # Defaults
        self.add_anchor(default_name="title", default_weight=1.4)
        self.add_anchor(default_name="tree", default_weight=1.0)
        self.anchor_list.selection_set(0)
        self.on_anchor_select()

    # ---------------- Mousewheel scrolling ----------------

    def _install_mousewheel_scrolling(self, canvas: tk.Canvas):
        def _on_mousewheel_win(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_linux_up(_event):
            canvas.yview_scroll(-3, "units")

        def _on_mousewheel_linux_down(_event):
            canvas.yview_scroll(3, "units")

        def _bind(_event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel_win)
            canvas.bind_all("<Button-4>", _on_mousewheel_linux_up)
            canvas.bind_all("<Button-5>", _on_mousewheel_linux_down)

        def _unbind(_event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind("<Enter>", _bind)
        canvas.bind("<Leave>", _unbind)

    # ---------------- Suggest config persistence ----------------

    def _bind_suggest_var_traces(self):
        def on_change(*_args):
            if self._loading_suggest_to_ui:
                return
            self._save_suggest_ui_to_selected_anchor()

        for v in (
            self.suggest_region,
            self.suggest_box_w,
            self.suggest_box_h,
            self.suggest_stride,
            self.suggest_topk,
            self.suggest_lambda,
            self.suggest_use_edges,
        ):
            v.trace_add("write", on_change)

    def _save_suggest_ui_to_selected_anchor(self):
        if self.selected_anchor_idx is None:
            return
        idx = self.selected_anchor_idx
        if not (0 <= idx < len(self.anchors)):
            return
        try:
            self.anchors[idx].suggest = SuggestUIState(
                region=str(self.suggest_region.get() or "full"),
                box_w=int(self.suggest_box_w.get()),
                box_h=int(self.suggest_box_h.get()),
                stride=int(self.suggest_stride.get()),
                topk=int(self.suggest_topk.get()),
                lam=float(self.suggest_lambda.get()),
                use_edges=bool(self.suggest_use_edges.get()),
            )
        except Exception:
            # don't break UI if user is mid-typing invalid float
            return

    def _load_suggest_ui_from_anchor(self, idx: int):
        if not (0 <= idx < len(self.anchors)):
            return
        st = self.anchors[idx].suggest
        self._loading_suggest_to_ui = True
        try:
            self.suggest_region.set(st.region or "full")
            self.suggest_box_w.set(int(st.box_w))
            self.suggest_box_h.set(int(st.box_h))
            self.suggest_stride.set(int(st.stride))
            self.suggest_topk.set(int(st.topk))
            self.suggest_lambda.set(float(st.lam))
            self.suggest_use_edges.set(bool(st.use_edges))
        finally:
            self._loading_suggest_to_ui = False

    # ---------------- Layout helpers ----------------

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
            self._paned.sashpos(0, int(w * 0.66))
        except Exception:
            pass

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

    # ---------------- Frames list / reorder ----------------

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
            self._set_status("Reference frame changed. Anchors are kept; verify they still match the new reference.")

    def on_frames_drag_end(self, _event):
        self._frames_drag_from = None

    # ---------------- Anchors ----------------

    def add_anchor(self, default_name: Optional[str] = None, default_weight: float = 1.0):
        idx = len(self.anchors)
        name = default_name if default_name is not None else f"anchor{idx+1}"

        # heuristic defaults for first two common anchors
        if idx == 0 and (default_name or "").lower() in ("title", "sequence", "sequences"):
            sug = SuggestUIState(region="top-right", box_w=520, box_h=170, stride=24, topk=8, lam=2.5, use_edges=True)
        elif idx == 1 and (default_name or "").lower() in ("tree", "bubble"):
            sug = SuggestUIState(region="bottom", box_w=420, box_h=270, stride=24, topk=8, lam=2.0, use_edges=True)
        else:
            sug = SuggestUIState()

        a = AnchorUI(name=name, weight=float(default_weight), box=None, color=_cycle_color(idx), suggest=sug)
        self.anchors.append(a)
        self.refresh_anchor_list(select_idx=idx)

    def remove_anchor(self):
        idx = self.selected_anchor_idx
        if idx is None or idx < 0 or idx >= len(self.anchors):
            return
        del self.anchors[idx]
        for i, a in enumerate(self.anchors):
            a.color = _cycle_color(i)
        self.refresh_anchor_list(select_idx=min(idx, len(self.anchors) - 1) if self.anchors else None)
        self.redraw_selection_preview()

    def clear_anchors(self):
        self.anchors.clear()
        self.selected_anchor_idx = None
        self.refresh_anchor_list(select_idx=None)
        self.redraw_selection_preview()

    def refresh_anchor_list(self, select_idx: Optional[int]):
        if not self.anchor_list:
            return
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
            if self.anchor_name:
                self.anchor_name.set(self.anchors[select_idx].name)
            if self.anchor_weight:
                self.anchor_weight.set(self.anchors[select_idx].weight)
            self._load_suggest_ui_from_anchor(select_idx)
        else:
            self.selected_anchor_idx = None
            if self.anchor_name:
                self.anchor_name.set("")
            if self.anchor_weight:
                self.anchor_weight.set(1.0)

    def on_anchor_select(self, _event=None):
        if not self.anchor_list:
            return
        sel = self.anchor_list.curselection()
        if not sel:
            self.selected_anchor_idx = None
            return
        idx = int(sel[0])
        self.selected_anchor_idx = idx
        a = self.anchors[idx]
        if self.anchor_name:
            self.anchor_name.set(a.name)
        if self.anchor_weight:
            self.anchor_weight.set(a.weight)
        self._load_suggest_ui_from_anchor(idx)
        self.redraw_selection_preview()

    def apply_anchor_edits(self):
        idx = self.selected_anchor_idx
        if idx is None or idx < 0 or idx >= len(self.anchors):
            return
        if self.anchor_name:
            self.anchors[idx].name = self.anchor_name.get().strip() or self.anchors[idx].name
        if self.anchor_weight:
            try:
                self.anchors[idx].weight = float(self.anchor_weight.get())
            except Exception:
                messagebox.showerror("Invalid weight", "Weight must be a number.")
                return
        self.refresh_anchor_list(select_idx=idx)
        self.redraw_selection_preview()

    # ---------------- Suggestion actions ----------------

    def _validate_for_suggest(self) -> bool:
        if not self.images:
            messagebox.showwarning("Missing input", "Load images first.")
            return False
        if self.selected_anchor_idx is None:
            messagebox.showwarning("No anchor selected", "Select an anchor first.")
            return False
        return True

    def suggest_box_for_selected_anchor(self):
        if not self._validate_for_suggest():
            return
        self._suggest_index = 0
        self._run_suggest_and_apply(index=0)

    def next_suggestion_for_selected_anchor(self):
        if not self._validate_for_suggest():
            return
        self._suggest_index += 1
        self._run_suggest_and_apply(index=self._suggest_index)

    def _run_suggest_and_apply(self, index: int):
        self._set_busy(True, "Suggesting anchor box...")

        region = (self.suggest_region.get() or "full").strip()
        region_opt = None if region in ("", "full") else region

        cfg = SuggestConfig(
            box_size=(int(self.suggest_box_w.get()), int(self.suggest_box_h.get())),
            stride=int(self.suggest_stride.get()),
            topk=int(self.suggest_topk.get()),
            lambda_stability=float(self.suggest_lambda.get()),
            use_edges=bool(self.suggest_use_edges.get()),
            region=region_opt,
        )

        def worker():
            try:
                boxes = suggest_anchor_boxes(self.images, cfg)
                if not boxes:
                    raise RuntimeError("No suggestions produced (try different size/stride/region).")
                i = index % len(boxes)
                box = boxes[i]
                self._ui_apply_suggested_box(box, i, len(boxes))
            except Exception as e:
                self._ui_done_err(str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _ui_apply_suggested_box(self, box: Box, idx: int, total: int):
        def done():
            self._set_busy(False, "")
            if self.selected_anchor_idx is None:
                return
            self.anchors[self.selected_anchor_idx].box = box
            # suggestion settings are already auto-saved by var traces
            self.refresh_anchor_list(select_idx=self.selected_anchor_idx)
            self.redraw_selection_preview()
            self._set_status(f"Suggestion applied: #{idx+1}/{total}  box={box}")

        self.after(0, done)

    # ---------------- Load images ----------------

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
        if self.before_label:
            self.before_label.config(text="(click Preview)", image="")
        if self.after_label:
            self.after_label.config(text="(click Preview)", image="")
        self._set_status("Images loaded. Drag frames to reorder (top is reference).")
        self.refresh_frames_list()
        self.after(120, self._freeze_preview_panel_sizes_once)

    # ---------------- Selection canvas ----------------

    def redraw_selection_preview(self):
        if self.preview_img is None or self.canvas is None:
            return

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        iw, ih = self.preview_img.size
        s = min(cw / iw, ch / ih)
        s = max(0.05, min(1.5, s))
        self.scale = s

        disp = self.preview_img.resize((max(1, int(iw * s)), max(1, int(ih * s))), Image.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(disp)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.preview_photo, anchor="nw")

        for i, a in enumerate(self.anchors):
            self._draw_annotated_box(a, i)

    def _draw_annotated_box(self, a: AnchorUI, idx: int):
        if not a.box or self.canvas is None:
            return
        x1, y1, x2, y2 = a.box
        s = self.scale
        sx1, sy1, sx2, sy2 = x1 * s, y1 * s, x2 * s, y2 * s

        outline = a.color
        width = 3 if idx == self.selected_anchor_idx else 2

        self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=outline, width=width)
        text = f"{idx+1}:{a.name}  w={a.weight:g}"
        self.canvas.create_rectangle(sx1, sy1, sx1 + 260, sy1 + 18, fill="#000000", outline="")
        self.canvas.create_text(sx1 + 4, sy1 + 2, anchor="nw", text=text, fill=outline, font=("TkDefaultFont", 9, "bold"))

    # ---- mouse events ----

    def on_mouse_down(self, event):
        if self.preview_img is None or self.canvas is None:
            return
        if self.selected_anchor_idx is None:
            messagebox.showwarning("No anchor selected", "Select an anchor in the list first.")
            return
        self._drag_start = (event.x, event.y)
        if self._current_rect_id:
            self.canvas.delete(self._current_rect_id)
            self._current_rect_id = None

    def on_mouse_drag(self, event):
        if not self._drag_start or self.preview_img is None or self.canvas is None:
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
        ox1 = max(ox0 + 1, min(iw, ox1))
        oy1 = max(oy0 + 1, min(ih, oy1))

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
        core: List[Anchor] = []
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

        self._freeze_preview_panel_sizes_once()
        self.stop_animation()

        self._set_busy(True, "Building preview (before/after)...")
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
            self._set_busy(False, "")

            self._freeze_preview_panel_sizes_once()
            bw, bh = self._before_box_wh or (420, 300)
            aw, ah = self._after_box_wh or (420, 300)

            before_fit = [self._fit_to_box_fixed(im, bw, bh) for im in before_pil]
            after_fit = [self._fit_to_box_fixed(im, aw, ah) for im in after_pil]

            self.before_frames_tk = [ImageTk.PhotoImage(im) for im in before_fit]
            self.after_frames_tk = [ImageTk.PhotoImage(im) for im in after_fit]

            self._anim_idx = 0
            self._tick_animation()
            self._set_status(f"Preview ready. (delay={self._anim_delay_ms}ms)")

        self.after(0, done)

    def _tick_animation(self):
        if not self.before_frames_tk or not self.after_frames_tk:
            return
        i = self._anim_idx % len(self.before_frames_tk)
        if self.before_label:
            self.before_label.config(image=self.before_frames_tk[i], text="")
        if self.after_label:
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

        self._set_busy(True, "Running alignment + exporting GIF...")

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

    # ---------------- Busy + status helpers ----------------

    def _set_status(self, text: str):
        if self.status:
            self.status.config(text=text)

    def _set_busy(self, busy: bool, text: str):
        if busy:
            if self.progress:
                self.progress.start(10)
            self._set_status(text)
            self._set_controls_enabled(False)
        else:
            if self.progress:
                self.progress.stop()
            self._set_controls_enabled(True)

    def _ui_done_ok(self, msg: str):
        def done():
            self._set_busy(False, "")
            self._set_status(msg)
            messagebox.showinfo("Done", msg)
        self.after(0, done)

    def _ui_done_err(self, err: str):
        def done():
            self._set_busy(False, "")
            self._set_status("Error.")
            messagebox.showerror("Error", err)
        self.after(0, done)

    # ---------------- Enable/disable controls ----------------

    def _set_controls_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for child in self.winfo_children():
            self._set_state_recursive(child, state)

        if self.canvas:
            self.canvas.configure(state="normal")
        if self.frames_listbox:
            self.frames_listbox.configure(state="normal")
        if self.anchor_list:
            self.anchor_list.configure(state="normal")

    def _set_state_recursive(self, widget, state: str):
        try:
            widget.configure(state=state)
        except Exception:
            pass
        for c in widget.winfo_children():
            self._set_state_recursive(c, state)


if __name__ == "__main__":
    App().mainloop()
