from dataclasses import dataclass
from typing import Tuple, Callable, Optional

Box = Tuple[int, int, int, int]
RGBA = Tuple[int, int, int, int]


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class Anchor:
    """An anchor region used to estimate alignment shift."""
    box: Box
    weight: float = 1.0
    name: str = ""  # optional label for debugging/UI


@dataclass(frozen=True)
class AlignConfig:
    method: str = "fft"          # "fft" or "fft-refine"
    use_edges: bool = True
    bg: RGBA = (255, 255, 255, 255)

    # FFT-refine only:
    refine_radius: int = 6       # pixels
    refine_weighted_score: bool = True  # apply weights in refinement too


@dataclass
class BestMatch:
    dx: int
    dy: int
    # For FFT: score = -peak (higher peak => better; negative so "lower is better")
    # For refinement: score = weighted SAD on anchors (lower is better)
    score: float


ProgressFn = Optional[Callable[[int, int, BestMatch], None]]  # (tested, total, best)