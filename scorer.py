"""
scorer.py — 综合评分
====================
将各维度分项聚合为 recognition_score / localization_score / final_score。
不包含指标计算和报告生成。
"""


def compute_overall_score(det_metrics: dict, count_score: float,
                          struct_metrics: dict, loc_metrics: dict) -> tuple:
    """
    综合评分（识别主导）：
      recognition_score = det_f1 * 0.55 + count_score * 0.20 + structure_score * 0.25
        where structure_score = order_overall*0.35 + parent_acc*0.20
                                + connection_acc*0.15 + position_acc*0.30
      localization_score = region_hit_rate * 0.55 + norm_offset_bonus * 0.30
                           + hit_rate * 0.10 + avg_iou * 0.05
      final_score = recognition_score * 0.90 + localization_score * 0.10

    返回 (final_score, recognition_score, localization_score, structure_sub)
    """
    det_f1 = det_metrics.get("f1", 0)

    structure_sub = {
        "order_overall": struct_metrics.get("order_overall", 0),
        "parent_acc": struct_metrics.get("parent_acc", 0),
        "connection_acc": struct_metrics.get("connection_acc", 0),
        "position_acc": struct_metrics.get("position_acc", 0),
    }

    structure_score = (
        structure_sub["order_overall"] * 0.35 +
        structure_sub["parent_acc"] * 0.20 +
        structure_sub["connection_acc"] * 0.15 +
        structure_sub["position_acc"] * 0.30
    )

    recognition_score = det_f1 * 0.55 + count_score * 0.20 + structure_score * 0.25

    # 定位分：区域一致性主导，IoU 降至极低权重
    localization_score = (
        loc_metrics.get("region_hit_rate", 0) * 0.55 +
        max(0.0, 1.0 - min(loc_metrics.get("norm_offset", 0), 0.5) * 2) * 0.30 +
        loc_metrics.get("hit_rate", 0) * 0.10 +
        loc_metrics.get("avg_iou", 0) * 0.05
    )

    final_score = recognition_score * 0.90 + localization_score * 0.10
    return final_score, recognition_score, localization_score, structure_sub