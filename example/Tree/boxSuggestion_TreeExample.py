from pathlib import Path

from PIL import Image

from core.BoxSuggestion import SuggestConfig, suggest_anchor_boxes
from core.DataModel import Anchor
from example.Tree import frames
from utility.debug import save_debug_image


# ----------------------------
# Paths
# ----------------------------
EXPORT_PATH = Path("boxSuggestion_export")
EXPORT_PATH.mkdir(exist_ok=True)


ref = frames[0]


def boxes_to_anchors(boxes, prefix: str) -> list[Anchor]:
    return [
        Anchor(box=b, weight=1.0, name=f"{prefix}#{i+1}")
        for i, b in enumerate(boxes)
    ]


# ----------------------------
# Suggest TITLE anchor boxes
# ----------------------------
cfg_title = SuggestConfig(
    box_size=(250, 125),
    region="top-right",
    topk=8,
    stride=24,
    lambda_stability=2.5,
)

title_boxes = suggest_anchor_boxes(frames, cfg_title)
print("Title suggestions:", title_boxes[:3])

title_anchors = boxes_to_anchors(title_boxes, "TITLE")

save_debug_image(
    image=ref,
    anchors=title_anchors,
    out_path=EXPORT_PATH/"title_suggestions.png",
    title="TITLE",
    max_draw=1,
)

# ----------------------------
# Suggest TREE/BUBBLE anchor boxes
# ----------------------------
cfg_tree = SuggestConfig(
    box_size=(350, 400),
    region="bottom",
    topk=8,
    stride=24,
    lambda_stability=2.0,
)

tree_boxes = suggest_anchor_boxes(frames, cfg_tree)
print("Tree suggestions:", tree_boxes[:3])

tree_anchors = boxes_to_anchors(tree_boxes, "TREE")

save_debug_image(
    image=ref,
    anchors=tree_anchors,
    out_path= EXPORT_PATH/"tree_suggestions.png",
    title="TREE",
    max_draw=1,
)
