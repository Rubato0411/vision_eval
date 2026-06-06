"""
matcher.py — 匹配引擎
=====================
组件间贪心匹配（描述级和 bbox 级）。
不包含指标计算。
"""

from core import position_distance, center_dist, box_iou, box_center


def greedy_match(preds: list, gts: list) -> list[dict]:
    """
    同类组件间贪心匹配（基于 position_desc 距离）。
    返回 [{pred, gt, distance}, ...]
    """
    unmatched_gt = list(gts)
    pairs = []

    for pred in preds:
        candidates = [g for g in unmatched_gt if g.label == pred.label]
        if not candidates:
            pairs.append({"pred": pred, "gt": None, "distance": 999})
            continue
        best = min(candidates, key=lambda g: position_distance(pred.position_desc, g.position_desc))
        unmatched_gt.remove(best)
        pairs.append({
            "pred": pred,
            "gt": best,
            "distance": position_distance(pred.position_desc, best.position_desc),
        })
    return pairs


def greedy_match_bbox(preds: list, gts: list) -> list[dict]:
    """基于 bbox 中心距离的贪心匹配（同类组件）"""
    unmatched_gt = list(gts)
    pairs = []

    for pred in preds:
        candidates = [g for g in unmatched_gt if g.label == pred.label]
        if not candidates:
            pairs.append({"pred": pred, "gt": None, "distance": 999, "iou": 0, "offset": None})
            continue
        best = min(candidates, key=lambda g: center_dist(pred.bbox, g.bbox))
        unmatched_gt.remove(best)
        pairs.append({
            "pred": pred,
            "gt": best,
            "distance": center_dist(pred.bbox, best.bbox),
            "iou": box_iou(pred.bbox, best.bbox),
            "offset": (
                box_center(pred.bbox)[0] - box_center(best.bbox)[0],
                box_center(pred.bbox)[1] - box_center(best.bbox)[1],
            ),
        })
    return pairs