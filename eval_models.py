"""
eval_models.py — 多图片批量评测主入口
=====================================
职责：组装 parser / matcher / metrics / scorer / report，协调评估流程。
JSON 为核心数据输出；txt/png 报告为辅助渲染。
"""

import json
import traceback
from pathlib import Path
from PIL import Image

from core import ModelData, EvaluationResult, ALL_TYPES
from parser import (
    parse_desc_file, parse_bbox_file, derive_standard,
    check_counts_consistency,
)
from matcher import greedy_match, greedy_match_bbox
from metrics import (
    compute_detection_metrics, compute_per_type_metrics, compute_count_score,
    compute_structure_metrics, compute_localization_metrics,
    compute_engineering_score,
)
from scorer import compute_overall_score
from report import generate_text_report, generate_visual_report, generate_json_report


# ============================================================
# 配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

COLORS = {
    "GT":        "#D4A017",
    "DB":        "#2ECC71",
    "DeepSeek":  "#E74C3C",
    "GPT-5.5":   "#3498DB",
    "Gemini":    "#9B59B6",
}

# 多图片批量配置：每项为一个图片的完整评测配置
BATCH_CONFIG = [
    {
        "label": "figure1",
        "image": "figure1.png",
        "gt": "standard-fi.json",
        "descs": {
            "db-des-f1.json":    "DB",
            "ds-des-f1.json":    "DeepSeek",
            "gpt-des-f1.json":   "GPT-5.5",
            "ge-des-fi.json":    "Gemini",
        },
        "bboxes": {
            "db-f1-output.json":      "DB",
            "ds-f1-output.json":      "DeepSeek",
            "gpt5.5-f1-output.json":  "GPT-5.5",
            "gemini-f1-output.json":  "Gemini",
        },
    },
    # 追加更多图片：
    # {"label": "figure2", "image": "figure2.png", "gt": "standard-f2.json",
    #  "descs": {...}, "bboxes": {...}},
]


# ============================================================
# 单图片评估
# ============================================================

def evaluate_single_image(cfg: dict) -> list[EvaluationResult]:
    """
    对单张图片的所有模型进行评测，返回 EvaluationResult 列表。
    cfg 包含: label, image, gt, descs, bboxes
    """
    image_label = cfg["label"]
    image_path = BASE_DIR / cfg["image"]
    gt_path = BASE_DIR / cfg["gt"]

    print(f"\n{'#' * 60}\n  图片: {image_label}\n{'#' * 60}")

    # ---- 动态图像尺寸 ----
    img_w, img_h = 1200, 900
    try:
        if image_path.exists():
            img = Image.open(image_path)
            img_w, img_h = img.size
            print(f"  图像尺寸: {img_w} x {img_h}")
        else:
            print(f"  [警告] {cfg['image']} 不存在，使用默认尺寸 {img_w}x{img_h}")
    except Exception as ex:
        print(f"  [警告] 无法读取图像: {ex}，使用默认尺寸 {img_w}x{img_h}")

    # ---- 加载 GT ----
    std, split_x = derive_standard(gt_path, img_w, img_h)
    print(f"  GT: {len(std.desc_components)} 个组件, counts={std.desc_counts}")
    print(f"  动态左右分界点: x={split_x:.1f}")

    # ---- 加载模型数据 ----
    all_model_data: dict[str, ModelData] = {}

    for desc_file, name in cfg["descs"].items():
        fp = BASE_DIR / desc_file
        if not fp.exists():
            print(f"  [警告] 找不到 {desc_file}，跳过")
            continue
        comps, counts = parse_desc_file(fp)
        all_model_data[name] = ModelData(
            name=name, image_name=image_label,
            desc_components=comps, bbox_components=[], desc_counts=counts,
        )

    for bbox_file, name in cfg["bboxes"].items():
        fp = BASE_DIR / bbox_file
        if not fp.exists():
            print(f"  [警告] 找不到 {bbox_file}，跳过")
            continue
        if name in all_model_data:
            all_model_data[name].bbox_components = parse_bbox_file(
                fp, img_w, img_h, split_x,
            )

    # ---- 逐模型评估 ----
    results: list[EvaluationResult] = []

    for name, md in all_model_data.items():
        print(f"\n  {'─' * 50}\n    [{image_label}] [{name}]\n  {'─' * 50}")

        try:
            # 逻辑一致性检查
            desc_fp_candidates = [k for k, v in cfg["descs"].items() if v == name]
            desc_fp = BASE_DIR / desc_fp_candidates[0] if desc_fp_candidates else None
            consistency = (check_counts_consistency(desc_fp, md.desc_counts)
                           if desc_fp else {"match": True, "score": 1.0,
                                             "declared_counts": {},
                                             "parsed_counts": md.desc_counts})

            # 工程诊断
            eng = compute_engineering_score(md.bbox_components, consistency, name)

            # 一、识别准确率
            pairs_desc = greedy_match(md.desc_components, std.desc_components)
            det = compute_detection_metrics(pairs_desc, len(std.desc_components))
            per_type = compute_per_type_metrics(md.desc_components, std.desc_components)
            count_score = compute_count_score(md.desc_counts, std.desc_counts)

            print(f"    识别: P={det['precision']:.3f}  R={det['recall']:.3f}  "
                  f"F1={det['f1']:.3f}  幻觉率={det.get('hallucination_rate', 0):.3f}")

            # 二、结构理解
            struct = compute_structure_metrics(md.desc_components, std.desc_components, pairs_desc)
            print(f"    结构: order(L)={struct['order_left']:.3f}  order(R)={struct['order_right']:.3f}  "
                  f"parent={struct['parent_acc']:.3f}  position={struct.get('position_acc', 0):.3f}")

            # 三、定位能力
            bbox_pairs = greedy_match_bbox(md.bbox_components, std.bbox_components)
            loc = compute_localization_metrics(bbox_pairs, img_w, img_h, split_x)
            print(f"    定位: 区域命中={loc['region_hit_rate']:.3f}  IoU={loc['avg_iou']:.4f}  "
                  f"偏移={loc['avg_offset']:.1f}px")

            # 四、综合评分
            overall, rec_score, loc_score, structure_sub = compute_overall_score(
                det, count_score, struct, loc)
            print(f"    评分: recognition={rec_score:.4f}  localization={loc_score:.4f}  "
                  f"overall={overall:.4f}  engineering={eng['eng_score']:.4f}")

            # 五、幻觉/一致性问题标记
            hall_rate = det.get("hallucination_rate", 0)
            total_pred = det["tp"] + det["fp"]
            hall_warn = hall_rate > 0.20 or total_pred > len(std.desc_components) * 1.5
            logic_bad = not consistency.get("match", True)

            if hall_warn:
                print(f"    [高幻觉风险] 幻觉率={hall_rate:.1%}  "
                      f"预测总数={total_pred} (GT={len(std.desc_components)})")
            if logic_bad:
                print(f"    [逻辑不一致] 声明={consistency.get('declared_counts', {})} "
                      f"实际={consistency.get('parsed_counts', {})}")

            # 打包为 EvaluationResult
            r = EvaluationResult(
                model_name=name,
                image_name=image_label,
                precision=det["precision"],
                recall=det["recall"],
                f1=det["f1"],
                tp=det["tp"],
                fp=det["fp"],
                fn=det["fn"],
                hallucination_rate=hall_rate,
                count_score=count_score,
                order_left=struct["order_left"],
                order_right=struct["order_right"],
                order_overall=struct["order_overall"],
                parent_acc=struct["parent_acc"],
                connection_acc=struct["connection_acc"],
                position_acc=struct.get("position_acc", 0),
                region_hit_rate=loc["region_hit_rate"],
                avg_iou=loc["avg_iou"],
                avg_offset=loc["avg_offset"],
                norm_offset=loc["norm_offset"],
                hit_rate=loc["hit_rate"],
                recognition_score=rec_score,
                localization_score=loc_score,
                overall=overall,
                eng_score=eng["eng_score"],
                consistency_score=consistency.get("score", 1.0),
                label_error_rate=eng["label_error_rate"],
                bbox_error_rate=eng["bbox_error_rate"],
                hallucination_warning=hall_warn,
                logic_inconsistent=logic_bad,
                per_type=per_type,
                desc_pairs=pairs_desc,
                bbox_pairs=bbox_pairs,
                structure_sub=structure_sub,
            )
            results.append(r)

        except Exception as ex:
            print(f"    [致命错误] {name} 评估失败: {ex}")
            traceback.print_exc()

    return results


# ============================================================
# 主流程
# ============================================================

def main():
    all_results: list[EvaluationResult] = []

    for cfg in BATCH_CONFIG:
        try:
            results = evaluate_single_image(cfg)
            all_results.extend(results)
        except Exception as ex:
            print(f"[致命错误] 图片 {cfg.get('label', '?')} 评估失败: {ex}")
            traceback.print_exc()

    if not all_results:
        print("没有成功评估任何模型，退出。")
        return

    # ---- JSON 报告（核心数据，始终输出） ----
    json_path = BASE_DIR / "eval_results.json"
    generate_json_report(all_results, json_path)

    # ---- 汇总控制台输出 ----
    print(f"\n{'=' * 60}")
    print(f"评测汇总: {len(all_results)} 个结果 "
          f"({len({r.image_name for r in all_results})} 图片, "
          f"{len({r.model_name for r in all_results})} 模型)")
    model_order = sorted({r.model_name for r in all_results})
    for mn in model_order:
        img_results = [r for r in all_results if r.model_name == mn]
        avg_overall = sum(r.overall for r in img_results) / len(img_results)
        print(f"  [{mn}] 图片数={len(img_results)}  平均综合分={avg_overall:.4f}")

    # ---- 辅助渲染：txt / png（仅当有第一张图片的 GT 信息时） ----
    # txt/png 需要内部 pairs 数据，这里按图片分组渲染
    generate_aux_reports = True  # 可设为 False 跳过辅助渲染

    if generate_aux_reports:
        out_txt = BASE_DIR / "eval_report.txt"
        out_png = BASE_DIR / "eval_report.png"

        # 按图片分组
        by_image: dict[str, list[EvaluationResult]] = {}
        for r in all_results:
            by_image.setdefault(r.image_name, []).append(r)

        for image_name, img_results in by_image.items():
            # 加载该图片的 GT 数据用于报告
            cfg = next((c for c in BATCH_CONFIG if c["label"] == image_name), None)
            if cfg is None:
                continue
            image_path = BASE_DIR / cfg["image"]
            gt_path = BASE_DIR / cfg["gt"]
            try:
                img = Image.open(image_path) if image_path.exists() else None
                img_w, img_h = img.size if img else (1200, 900)
            except Exception:
                img_w, img_h = 1200, 900

            try:
                std, _ = derive_standard(gt_path, img_w, img_h)
            except Exception:
                print(f"[警告] 无法加载 {image_name} 的 GT，跳过 txt/png 渲染")
                continue

            # 组装 all_eval 格式（兼容 report.py 的旧接口）
            all_eval = {}
            for r in img_results:
                all_eval[r.model_name] = {
                    "detection": {
                        "precision": r.precision, "recall": r.recall,
                        "f1": r.f1, "tp": r.tp, "fp": r.fp, "fn": r.fn,
                        "hallucination_rate": r.hallucination_rate,
                    },
                    "per_type": r.per_type,
                    "count_score": r.count_score,
                    "structure": {
                        "order_left": r.order_left, "order_right": r.order_right,
                        "order_overall": r.order_overall,
                        "parent_acc": r.parent_acc,
                        "connection_acc": r.connection_acc,
                        "position_acc": r.position_acc,
                    },
                    "localization": {
                        "region_hit_rate": r.region_hit_rate,
                        "avg_iou": r.avg_iou, "avg_offset": r.avg_offset,
                        "norm_offset": r.norm_offset, "hit_rate": r.hit_rate,
                    },
                    "overall": r.overall,
                    "recognition_score": r.recognition_score,
                    "localization_score": r.localization_score,
                    "structure_sub": r.structure_sub,
                    "engineering": {
                        "eng_score": r.eng_score,
                        "consistency_score": r.consistency_score,
                        "label_error_rate": r.label_error_rate,
                        "bbox_error_rate": r.bbox_error_rate,
                        "parse_success": 1.0,
                    },
                    "desc_pairs": r.desc_pairs,
                    "bbox_pairs": r.bbox_pairs,
                }

            model_order_for_img = [r.model_name for r in img_results]
            try:
                generate_text_report(std, all_eval, model_order_for_img, out_txt)
                print(f"已生成文本报告: {out_txt}")
            except Exception as ex:
                print(f"[错误] 文本报告生成失败: {ex}")

            try:
                generate_visual_report(std, all_eval, model_order_for_img,
                                       image_path, COLORS, out_png)
                print(f"已生成可视化报告: {out_png}")
            except Exception as ex:
                print(f"[错误] 可视化报告生成失败: {ex}")
                traceback.print_exc()


if __name__ == "__main__":
    main()