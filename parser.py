"""
parser.py — 数据解析与几何工具
===============================
职责：JSON 解析、标准推导、GT 加载、逻辑一致性检查。
数据结构与基础几何函数已迁移至 core.py。
"""

import json
from pathlib import Path

from core import (
    Component, ModelData, ALL_TYPES, box_center, bbox_is_valid,
    normalize_position, _side_to_parent,
)


# ============================================================
# 区域判断（依赖图片尺寸）
# ============================================================

def get_region(cx: float, cy: float, img_w: int, img_h: int, split_x: float = None) -> str:
    """根据中心点坐标确定语义区域。split_x 为动态左右分界点，默认 img_w/2。"""
    if split_x is None:
        split_x = img_w / 2
    h_third = img_h / 3
    if cy < h_third:
        vert = "top"
    elif cy < 2 * h_third:
        vert = "middle"
    else:
        vert = "bottom"

    side = "left" if cx < split_x else "right"
    return f"{vert}-{side}"


def compute_content_split_x(components: list) -> float:
    """
    从组件 bbox 中心点计算左右分界点。
    使用"最大相邻间距"法：排序后找相邻点最大间隙的中点，天然分割左右簇。
    """
    cxs = sorted([
        box_center(c.bbox)[0] for c in components
        if c.bbox and not (c.bbox[0] == 0 and c.bbox[1] == 0
                           and c.bbox[2] == 0 and c.bbox[3] == 0)
    ])
    if len(cxs) < 2:
        return cxs[0] if cxs else 0

    max_gap, gap_idx = 0.0, 0
    for i in range(len(cxs) - 1):
        gap = cxs[i + 1] - cxs[i]
        if gap > max_gap:
            max_gap = gap
            gap_idx = i

    if max_gap < 5:
        return (cxs[0] + cxs[-1]) / 2

    return (cxs[gap_idx] + cxs[gap_idx + 1]) / 2


# ============================================================
# 计数工具
# ============================================================

def compute_counts_from_components(components: list) -> dict[str, int]:
    """从组件列表重新计算类别数量（废弃 JSON 中自带的 counts 字段）"""
    counts = {}
    for t in ALL_TYPES:
        counts[t] = sum(1 for c in components if c.label == t)
    return counts


# ============================================================
# JSON 解析
# ============================================================

def parse_desc_file(filepath: Path) -> tuple[list, dict[str, int]]:
    """
    解析模型描述 JSON。
    返回 (components, counts)，其中 counts 由实际组件重新计算。
    异常时返回空列表和空 counts，不中断流水线。
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "components" not in data:
            print(f"  [解析警告] {filepath.name}: 缺少 'components' 字段或格式异常")
            return [], {t: 0 for t in ALL_TYPES}

        components = []
        for item in data["components"]:
            if not isinstance(item, dict):
                continue
            pos = item.get("position_desc", "")
            if not pos:
                continue
            parts = normalize_position(pos).split("-")
            v, h = parts[0], parts[1] if len(parts) > 1 else ""
            side_val = h if h in ("left", "right") else v
            components.append(Component(
                id=item.get("id", len(components) + 1),
                label=item.get("label", "unknown"),
                position_desc=pos,
                side=side_val,
                vert=v if v in ("top", "middle", "bottom") else h,
                parent=_side_to_parent(side_val),
            ))

        counts = compute_counts_from_components(components)
        return components, counts

    except (json.JSONDecodeError, OSError, KeyError) as ex:
        print(f"  [解析失败] {filepath.name}: {ex}")
        return [], {t: 0 for t in ALL_TYPES}


def parse_bbox_file(filepath: Path, img_w: int = 1200, img_h: int = 900,
                   split_x: float = None) -> list:
    """
    解析模型 bbox JSON。
    优先取 detections，兼容 core_components / components / 顶层数组。
    split_x 为动态左右分界点，None 时使用 img_w/2。
    异常时返回空列表，不中断流水线。
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as ex:
        print(f"  [解析失败] {filepath.name}: {ex}")
        return []

    try:
        if isinstance(data, list):
            items = [{"label": obj.get("component") or obj.get("label") or "unknown",
                       "bbox": obj.get("bounding_box") or obj.get("bbox") or [0, 0, 0, 0]}
                     for obj in data if isinstance(obj, dict)]
        elif isinstance(data, dict):
            items = (data.get("detections")
                     or data.get("core_components")
                     or data.get("components")
                     or [])
        else:
            items = []
    except Exception as ex:
        print(f"  [解析警告] {filepath.name}: 数据格式异常 - {ex}")
        items = []

    components = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        label = item.get("label") or item.get("name") or item.get("component") or "unknown"
        bbox = item.get("bbox") or item.get("bounding_box") or [0, 0, 0, 0]
        cx, cy = box_center(bbox)
        region = get_region(cx, cy, img_w, img_h, split_x)
        parts = region.split("-")
        components.append(Component(
            id=item.get("id", i + 1), label=label,
            position_desc=region,
            side=parts[1], vert=parts[0],
            parent=_side_to_parent(parts[1]),
            bbox=bbox,
        ))
    return components


# ============================================================
# 标准数据推导
# ============================================================

def derive_standard(gt_file: Path, img_w: int = 1200, img_h: int = 900) -> tuple:
    """
    从 GT JSON 的显式 schema 读取 GT 结构。
    GT JSON 已包含 parent / side / vert / order_in_side，不再从 bbox 推导。
    仅 split_x 仍从 bbox 计算（用于模型 bbox 的区域判断）。
    返回 (ModelData, split_x)。
    """
    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_name = data.get("image", gt_file.stem)
    desc_components = []
    bbox_components = []

    for item in data["components"]:
        label = item["name"]
        bbox = item["bbox"]
        parent = item.get("parent", "")
        side = item.get("side", "")
        vert = item.get("vert", "")
        pos = f"{vert}-{side}" if vert and side else ""

        c = Component(
            id=len(desc_components) + 1,
            label=label,
            position_desc=pos,
            side=side,
            vert=vert,
            parent=parent,
            bbox=bbox,
        )
        desc_components.append(c)
        bbox_components.append(c)

    # 按 side + vert 排序
    vert_order = {"top": 0, "middle": 1, "bottom": 2}
    desc_components.sort(key=lambda c: (
        0 if c.side == "left" else 1,
        vert_order.get(c.vert, 99),
    ))
    for i, c in enumerate(desc_components):
        c.id = i + 1

    split_x = compute_content_split_x(bbox_components)
    counts = compute_counts_from_components(desc_components)

    gt_data = ModelData(
        name="GT",
        image_name=image_name,
        desc_components=desc_components,
        bbox_components=bbox_components,
        desc_counts=counts,
    )
    return gt_data, split_x


# ============================================================
# 工程诊断：逻辑一致性校验
# ============================================================

def check_counts_consistency(filepath: Path, parsed_counts: dict) -> dict:
    """
    对比 JSON 中声明的 counts 与从组件列表实际计算的 counts。
    返回一致性指标：完全一致则 score=1.0。
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        declared = data.get("counts", {})
    except (json.JSONDecodeError, KeyError, OSError):
        return {"declared_counts": {}, "parsed_counts": parsed_counts,
                "match": False, "score": 0.0}

    match = True
    total_diff = 0
    total_declared = sum(declared.values())
    for t in ALL_TYPES:
        d = declared.get(t, 0)
        p = parsed_counts.get(t, 0)
        if d != p:
            match = False
        total_diff += abs(d - p)

    score = max(0.0, 1.0 - total_diff / max(total_declared, 1))

    return {
        "declared_counts": declared,
        "parsed_counts": parsed_counts,
        "match": match,
        "score": score,
    }