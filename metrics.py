"""
metrics.py — 指标计算
=====================
职责：检测指标、结构理解、定位能力、工程诊断。
匹配引擎已迁移至 matcher.py，综合评分已迁移至 scorer.py。
"""

import numpy as np
from core import (
    Component, ALL_TYPES, box_center, bbox_is_valid,
)
from matcher import greedy_match


# ============================================================
# 一、识别准确率
# ============================================================

def compute_detection_metrics(pairs: list[dict], gt_total: int) -> dict:
    """整体 TP/FP/FN 和 Precision/Recall/F1"""
    tp = sum(1 for p in pairs if p["gt"] is not None)
    fp = sum(1 for p in pairs if p["gt"] is None)
    fn = gt_total - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    total_pred = tp + fp
    hallucination_rate = fp / total_pred if total_pred > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "hallucination_rate": hallucination_rate}


def compute_per_type_metrics(preds: list, gts: list) -> dict:
    """按组件类型分别计算 precision/recall/f1"""
    results = {}
    for t in ALL_TYPES:
        pred_t = [c for c in preds if c.label == t]
        gt_t = [c for c in gts if c.label == t]
        pairs = greedy_match(pred_t, gt_t)
        m = compute_detection_metrics(pairs, len(gt_t))
        m["pred_count"] = len(pred_t)
        m["gt_count"] = len(gt_t)
        results[t] = m
    return results


def compute_count_score(pred_counts: dict, gt_counts: dict) -> float:
    """
    数量一致性评分。
    score = 1 - sum(|pred - gt|) / total_gt
    """
    total_gt = sum(gt_counts.values())
    if total_gt == 0:
        return 1.0
    total_diff = sum(abs(pred_counts.get(t, 0) - gt_counts.get(t, 0)) for t in ALL_TYPES)
    score = max(0.0, 1.0 - total_diff / total_gt)
    return score


# ============================================================
# 二、结构理解能力
# ============================================================

def get_component_sequence(components: list, side: str) -> list[str]:
    """提取某侧组件从上到下的类型序列"""
    vert_order = {"top": 0, "middle": 1, "bottom": 2}
    side_comps = [c for c in components if c.side == side]
    side_comps.sort(key=lambda c: vert_order.get(c.vert, 99))
    return [c.label for c in side_comps]


def compute_sequence_similarity(pred_seq: list[str], gt_seq: list[str]) -> float:
    """基于编辑距离的序列相似度 (0~1)"""
    m, n = len(pred_seq), len(gt_seq)
    if m == 0 and n == 0:
        return 1.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred_seq[i - 1] == gt_seq[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return 1.0 - dp[m][n] / max(m, n)


def extract_connections(seq: list[str]) -> set:
    """从序列提取相邻连接关系（有序对）"""
    return {(seq[i], seq[i + 1]) for i in range(len(seq) - 1)}


def compute_structure_metrics(preds: list, gts: list, desc_pairs: list[dict]) -> dict:
    """结构理解能力：组件顺序、父子层级、连接关系、位置匹配"""
    results = {}

    for side in ["left", "right"]:
        pred_seq = get_component_sequence(preds, side)
        gt_seq = get_component_sequence(gts, side)
        results[f"order_{side}"] = compute_sequence_similarity(pred_seq, gt_seq)

    results["order_overall"] = (results.get("order_left", 0) + results.get("order_right", 0)) / 2

    matched = [p for p in desc_pairs if p["gt"] is not None]
    if matched:
        correct_parent = sum(1 for p in matched if p["pred"].parent == p["gt"].parent)
        results["parent_acc"] = correct_parent / len(matched)
    else:
        results["parent_acc"] = 0.0

    gt_conns = set()
    pred_conns = set()
    for side in ["left", "right"]:
        gt_conns |= extract_connections(get_component_sequence(gts, side))
        pred_conns |= extract_connections(get_component_sequence(preds, side))

    correct_conns = len(gt_conns & pred_conns)
    results["connection_acc"] = correct_conns / len(gt_conns) if gt_conns else 0.0

    if matched:
        position_correct = sum(
            1 for p in matched
            if p["pred"].side == p["gt"].side and p["pred"].vert == p["gt"].vert
        )
        results["position_acc"] = position_correct / len(matched)
    else:
        results["position_acc"] = 0.0

    results["structure_sub"] = {
        "order_overall": results["order_overall"],
        "parent_acc": results["parent_acc"],
        "connection_acc": results["connection_acc"],
        "position_acc": results["position_acc"],
    }

    return results


# ============================================================
# 三、定位能力
# ============================================================

def compute_localization_metrics(bbox_pairs: list[dict],
                                 img_w: int = 1200, img_h: int = 900,
                                 split_x: float = None) -> dict:
    """
    定位能力：IoU、中心偏移、归一化偏移、bbox 命中率、区域命中率。
    使用动态图像尺寸与 split_x，消除硬编码。
    """
    if split_x is None:
        split_x = img_w / 2

    valid = [p for p in bbox_pairs if p["gt"] is not None]
    if not valid:
        return {"avg_iou": 0, "avg_offset": 0, "norm_offset": 0,
                "hit_rate": 0, "region_hit_rate": 0}

    ious = [p["iou"] for p in valid]
    avg_iou = np.mean(ious)

    offsets = [np.hypot(p["offset"][0], p["offset"][1]) for p in valid]
    avg_offset = np.mean(offsets)

    diag = np.hypot(img_w, img_h)
    norm_offset = avg_offset / diag if diag > 0 else 0.0

    hit_rate = sum(1 for iou in ious if iou > 0.3) / len(ious) if ious else 0.0

    h_third = img_h / 3
    region_hits = 0
    for p in valid:
        gt_cx, gt_cy = box_center(p["gt"].bbox)
        pred_cx, pred_cy = box_center(p["pred"].bbox)
        gt_side = "left" if gt_cx < split_x else "right"
        pred_side = "left" if pred_cx < split_x else "right"
        gt_vert = "top" if gt_cy < h_third else ("middle" if gt_cy < 2 * h_third else "bottom")
        pred_vert = "top" if pred_cy < h_third else ("middle" if pred_cy < 2 * h_third else "bottom")
        if gt_side == pred_side and gt_vert == pred_vert:
            region_hits += 1

    region_hit_rate = region_hits / len(valid) if valid else 0.0

    return {
        "avg_iou": avg_iou,
        "avg_offset": avg_offset,
        "norm_offset": norm_offset,
        "hit_rate": hit_rate,
        "region_hit_rate": region_hit_rate,
    }


# ============================================================
# 工程诊断
# ============================================================

def compute_engineering_score(components: list, consistency: dict = None,
                              file_label: str = "") -> dict:
    """
    诊断 JSON 输出质量：
    - label_error_rate: label 不在 ALL_TYPES 中的比例
    - bbox_error_rate: bbox 格式不合规的比例
    - consistency_score: 声明 counts 与实际解析 counts 的一致性
    - eng_score: 综合工程得分
    """
    total = len(components)
    if total == 0:
        return {"parse_success": 1.0, "label_error_rate": 0.0,
                "bbox_error_rate": 0.0, "consistency_score": 1.0,
                "eng_score": 1.0}

    label_errors = sum(1 for c in components if c.label not in ALL_TYPES)
    bbox_errors = sum(1 for c in components if not bbox_is_valid(c.bbox))

    label_error_rate = label_errors / total
    bbox_error_rate = bbox_errors / total

    consistency_score = consistency.get("score", 1.0) if consistency else 1.0

    eng_score = max(0.0, 1.0
                    - label_error_rate * 0.20
                    - bbox_error_rate * 0.20
                    - (1.0 - consistency_score) * 0.60)

    return {
        "parse_success": 1.0,
        "label_error_rate": label_error_rate,
        "bbox_error_rate": bbox_error_rate,
        "consistency_score": consistency_score,
        "eng_score": eng_score,
    }