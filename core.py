"""
core.py — 公共数据结构与常量
=============================
Component / ModelData / EvaluationResult、位置辅助函数、全局常量。
不含解析、匹配、评分、渲染逻辑。
"""

from dataclasses import dataclass, field, asdict
from typing import Any
import json

import numpy as np


# ============================================================
# 全局常量
# ============================================================

ALL_TYPES = ["Multi-Head Attention", "Feed Forward", "Add & Norm"]

POS_GRID = {
    "top": 0, "middle": 1, "bottom": 2,
    "left": 0, "right": 1,
}
VERT_KEYS = {"top", "middle", "bottom"}
HORIZ_KEYS = {"left", "right"}


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Component:
    id: int
    label: str
    position_desc: str  # e.g. "top-right"
    side: str = ""       # "left" or "right"
    vert: str = ""       # "top" / "middle" / "bottom"
    parent: str = ""     # "encoder" / "decoder" (GT 显式提供，模型从 side 推导)
    bbox: list = field(default_factory=lambda: [0, 0, 0, 0])


@dataclass
class ModelData:
    name: str
    image_name: str = ""
    desc_components: list = field(default_factory=list)   # 来自描述文件
    bbox_components: list = field(default_factory=list)   # 来自坐标文件
    desc_counts: dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """单个模型对单张图片的完整评测结果（可 JSON 序列化）"""
    model_name: str
    image_name: str
    # 识别
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    hallucination_rate: float = 0.0
    count_score: float = 0.0
    # 结构
    order_left: float = 0.0
    order_right: float = 0.0
    order_overall: float = 0.0
    parent_acc: float = 0.0
    connection_acc: float = 0.0
    position_acc: float = 0.0
    # 定位
    region_hit_rate: float = 0.0
    avg_iou: float = 0.0
    avg_offset: float = 0.0
    norm_offset: float = 0.0
    hit_rate: float = 0.0
    # 综合
    recognition_score: float = 0.0
    localization_score: float = 0.0
    overall: float = 0.0
    # 工程
    eng_score: float = 0.0
    consistency_score: float = 0.0
    label_error_rate: float = 0.0
    bbox_error_rate: float = 0.0
    # 高级标记
    hallucination_warning: bool = False
    logic_inconsistent: bool = False
    # 分类型明细
    per_type: dict = field(default_factory=dict)

    # 内部引用（不入 JSON）
    desc_pairs: list = field(default_factory=list, repr=False)
    bbox_pairs: list = field(default_factory=list, repr=False)
    structure_sub: dict = field(default_factory=dict, repr=False)

    def to_json_dict(self) -> dict:
        """输出可 JSON 序列化的纯数据 dict（不含 pairs）"""
        d = asdict(self)
        d.pop("desc_pairs", None)
        d.pop("bbox_pairs", None)
        d.pop("structure_sub", None)
        return d

    def to_json(self, filepath) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, ensure_ascii=False, indent=2)


# ============================================================
# 几何工具
# ============================================================

def box_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def box_area(bbox):
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def box_iou(a, b):
    xi = max(a[0], b[0]); yi = max(a[1], b[1])
    xa = min(a[2], b[2]); ya = min(a[3], b[3])
    inter = max(0, xa - xi) * max(0, ya - yi)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def center_dist(a, b):
    ca, cb = box_center(a), box_center(b)
    return np.hypot(cb[0] - ca[0], cb[1] - ca[1])


def bbox_is_valid(bbox) -> bool:
    """校验 bbox 格式：[x1, y1, x2, y2] 且 x2 > x1, y2 > y1"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    return all(isinstance(v, (int, float)) for v in bbox) and x2 > x1 and y2 > y1


# ============================================================
# 位置辅助函数
# ============================================================

def normalize_position(pos: str) -> str:
    """统一位置描述：'top-right' 和 'right-top' 归一化为 'right-top'"""
    parts = sorted(pos.lower().split("-"))
    return "-".join(parts)


def position_to_grid(pos: str) -> tuple:
    """将位置描述映射到网格坐标 (vert, horiz)"""
    parts = pos.split("-")
    v = next((POS_GRID[p] for p in parts if p in VERT_KEYS), -1)
    h = next((POS_GRID[p] for p in parts if p in HORIZ_KEYS), -1)
    return (v, h)


def position_distance(a: str, b: str) -> int:
    ga = position_to_grid(normalize_position(a))
    gb = position_to_grid(normalize_position(b))
    return abs(ga[0] - gb[0]) + abs(ga[1] - gb[1])


def _side_to_parent(side: str) -> str:
    """左侧编码器，右侧解码器"""
    if side == "left":
        return "encoder"
    if side == "right":
        return "decoder"
    return ""