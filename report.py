"""
report.py — 报告生成
====================
职责：文本报告、可视化报告（matplotlib）。
不包含解析和指标计算。
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from core import Component, box_center, ALL_TYPES, EvaluationResult


# ============================================================
# 绘图组件
# ============================================================

def draw_boxes(ax, img, components: list, color: str, show_label=True):
    ax.imshow(img)
    ax.axis("off")
    for c in components:
        x1, y1, x2, y2 = c.bbox
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            continue
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2.2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        if show_label:
            short = c.label.replace("Multi-Head Attention", "MHA") \
                           .replace("Feed Forward", "FF") \
                           .replace("Add & Norm", "A&N")
            ax.text(x1 + 2, y1 - 4, short, fontsize=5.5, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.85,
                              edgecolor=color, linewidth=0.5))


def draw_offset_heatmap(ax, img, gt_comps, all_bbox_pairs, colors, model_names):
    ax.imshow(img)
    ax.axis("off")
    for g in gt_comps:
        x1, y1, x2, y2 = g.bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=1.6, edgecolor=colors["GT"],
                                 facecolor="none", linestyle="-")
        ax.add_patch(rect)
        cx, cy = box_center(g.bbox)
        ax.plot(cx, cy, "o", color=colors["GT"], markersize=3.5)

    from matplotlib.lines import Line2D
    handles = [patches.Patch(edgecolor=colors["GT"], facecolor="none", lw=2, label="GT")]
    for model_name, pairs in all_bbox_pairs.items():
        color = colors[model_name]
        handles.append(Line2D([0], [0], color=color, lw=2, label=model_name))
        for p in pairs:
            if p["gt"] is None:
                continue
            gc = box_center(p["gt"].bbox)
            pc = box_center(p["pred"].bbox)
            ax.annotate("", xy=pc, xytext=gc,
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5,
                                        alpha=0.8, connectionstyle="arc3,rad=0.15"))
    ax.legend(handles=handles, loc="lower right", fontsize=6, framealpha=0.85)


def render_table(ax, title: str, headers: list, rows: list,
                 col_widths: list, x_start=0, row_height=0.45, fontsize=8.5):
    """通用表格渲染"""
    ax.axis("off")
    ax.set_xlim(0, sum(col_widths) + 1)
    ax.set_ylim(0, (len(rows) + 2) * row_height)

    ax.text(x_start + sum(col_widths) / 2, (len(rows) + 1.5) * row_height,
            title, fontsize=11, fontweight="bold", ha="center")

    x_positions = [x_start]
    for w in col_widths[:-1]:
        x_positions.append(x_positions[-1] + w)

    for xi, header in zip(x_positions, headers):
        ax.text(xi + 0.15, len(rows) * row_height, header,
                fontsize=fontsize, fontweight="bold", va="center")

    for row_idx, row in enumerate(rows):
        y = (len(rows) - row_idx - 1) * row_height
        for xi, cell in zip(x_positions, row):
            ax.text(xi + 0.15, y, str(cell), fontsize=fontsize - 0.5, va="center")

    for i in range(len(rows) + 1):
        y = i * row_height + row_height * 0.3
        ax.axhline(y=y, xmin=0.02, xmax=0.98, color="gray", lw=0.4, ls="--")


# ============================================================
# 文本报告
# ============================================================

def generate_text_report(std, all_eval: dict, model_order: list, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("模型评估报告\n" + "=" * 70 + "\n\n")
        f.write(f"标准: {len(std.desc_components)} 个组件\n")
        f.write(f"类型分布: {std.desc_counts}\n\n")

        for mn in model_order:
            if mn not in all_eval:
                continue
            e = all_eval[mn]
            d = e["detection"]
            s = e["structure"]
            l = e["localization"]
            eng = e.get("engineering", {})

            f.write(f"\n{'─'*50}\n[{mn}]\n{'─'*50}\n")

            # 一、识别准确率
            f.write("一、识别准确率\n")
            f.write(f"  整体 P={d['precision']:.3f}  R={d['recall']:.3f}  F1={d['f1']:.3f}\n")
            f.write(f"  TP={d['tp']}  FP={d['fp']}  FN={d['fn']}\n")
            f.write(f"  幻觉率 (FP/total_pred) = {d.get('hallucination_rate', 0):.3f}\n")
            f.write(f"  数量一致性评分: {e['count_score']:.4f}\n")
            for t in ALL_TYPES:
                m = e["per_type"][t]
                f.write(f"  [{t:25s}] P={m['precision']:.3f}  R={m['recall']:.3f}  "
                        f"F1={m['f1']:.3f}  pred={m['pred_count']}  gt={m['gt_count']}\n")

            # 二、结构理解
            f.write("\n二、结构理解能力\n")
            f.write(f"  顺序准确率(L)={s['order_left']:.3f}  (R)={s['order_right']:.3f}\n")
            f.write(f"  父子模块准确率 (parent) ={s['parent_acc']:.3f}\n")
            f.write(f"  连接关系准确率={s['connection_acc']:.3f}\n")
            f.write(f"  位置匹配准确率 (side+vert) ={s.get('position_acc', 0):.3f}\n")

            # 三、定位能力
            f.write("\n三、定位能力 (区域一致性主导)\n")
            f.write(f"  区域命中率={l['region_hit_rate']:.3f}  (核心定位指标)\n")
            f.write(f"  Avg IoU={l['avg_iou']:.4f}  (参考)\n")
            f.write(f"  中心点偏移={l['avg_offset']:.1f}px\n")
            f.write(f"  归一化偏移={l['norm_offset']:.4f}\n")
            f.write(f"  Bbox 命中率={l['hit_rate']:.3f}\n")

            # 四、综合评分
            f.write(f"\n四、综合评分\n")
            f.write(f"  ---- recognition_score 公式拆解 ----\n")
            f.write(f"  det_f1          = {d['f1']:.4f}  x 0.55 = {d['f1']*0.55:.4f}\n")
            f.write(f"  count_score     = {e['count_score']:.4f}  x 0.20 = {e['count_score']*0.20:.4f}\n")
            sub = e.get("structure_sub", {})
            struct_score = (
                sub.get("order_overall", 0) * 0.35 +
                sub.get("parent_acc", 0) * 0.20 +
                sub.get("connection_acc", 0) * 0.15 +
                sub.get("position_acc", 0) * 0.30
            )
            f.write(f"  structure_score = {struct_score:.4f}  x 0.25 = {struct_score*0.25:.4f}\n")
            f.write(f"    order_overall  = {sub.get('order_overall', 0):.4f}  x 0.35\n")
            f.write(f"    parent_acc     = {sub.get('parent_acc', 0):.4f}  x 0.20\n")
            f.write(f"    connection_acc = {sub.get('connection_acc', 0):.4f}  x 0.15\n")
            f.write(f"    position_acc   = {sub.get('position_acc', 0):.4f}  x 0.30\n")
            f.write(f"  ---- 汇总 ----\n")
            f.write(f"  识别主导分 (recognition_score) = {e.get('recognition_score', 0):.4f}\n")
            f.write(f"  定位分 (localization_score) = {e.get('localization_score', 0):.4f}\n")
            f.write(f"  最终综合得分 (final_score) = {e['overall']:.4f}\n")
            f.write(f"  （final = recognition x 0.90 + localization x 0.10）\n")

            # 五、工程诊断 + 幻觉警告
            f.write("\n五、工程诊断\n")
            f.write(f"  解析成功率={eng.get('parse_success', 1):.3f}\n")
            f.write(f"  Label 拼写错误率={eng.get('label_error_rate', 0):.3f}\n")
            f.write(f"  Bbox 格式错误率={eng.get('bbox_error_rate', 0):.3f}\n")
            f.write(f"  逻辑一致性={eng.get('consistency_score', 1):.3f}\n")
            f.write(f"  工程得分={eng.get('eng_score', 1):.4f}\n")

            # 幻觉高亮警告
            hall_rate = d.get("hallucination_rate", 0)
            total_pred = d["tp"] + d["fp"]
            gt_total = len(std.desc_components)
            if hall_rate > 0.20 or total_pred > gt_total * 1.5:
                f.write(f"\n  ⚠ [高幻觉风险] 幻觉率={hall_rate:.1%}  "
                        f"预测总数={total_pred} (GT={gt_total})\n")


# ============================================================
# 可视化报告
# ============================================================

def generate_visual_report(std, all_eval: dict, model_order: list,
                           image_path: Path, colors: dict, out_path: Path):
    img_np = None
    if image_path.exists():
        img_np = np.array(Image.open(image_path).convert("RGB"))

    fig = plt.figure(figsize=(38, 38))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.2, 0.02, 0.9, 1.0, 1.8],
                          hspace=0.35, wspace=0.22)

    # Row 1: GT + Heatmap
    if img_np is not None:
        ax_gt = fig.add_subplot(gs[0, 0])
        ax_gt.set_title("Ground Truth (标准)", fontsize=14, fontweight="bold", color=colors["GT"])
        draw_boxes(ax_gt, img_np, std.bbox_components, colors["GT"])

        ax_heat = fig.add_subplot(gs[0, 1])
        ax_heat.set_title("偏移热力图 (GT → 各模型预测)", fontsize=14, fontweight="bold")
        all_bbox_pairs = {mn: all_eval[mn]["bbox_pairs"] for mn in model_order if mn in all_eval}
        draw_offset_heatmap(ax_heat, img_np, std.bbox_components,
                            all_bbox_pairs, colors, model_order)
    else:
        for j in range(2):
            ax = fig.add_subplot(gs[0, j])
            ax.text(0.5, 0.5, "figure1.png 未找到", ha="center", va="center", fontsize=16)
            ax.axis("off")

    # Row 2: separator
    for j in range(2):
        ax = fig.add_subplot(gs[1, j])
        ax.axis("off")

    # Row 3: 组件类型分布饼图 (使用 subgridspec 避免 get_position 失效)
    models_with_data = [mn for mn in model_order if mn in all_eval]
    n_pies = min(len(models_with_data) + 1, 5)  # GT + up to 4 models
    pie_gs = gs[2, :].subgridspec(1, n_pies, wspace=0.15)
    pie_colors_list = ["#FFD700", "#E74C3C", "#2ECC71", "#3498DB", "#9B59B6"]
    short_labels = ["MHA", "FF", "A&N"]

    # GT pie
    ax_pie_gt = fig.add_subplot(pie_gs[0, 0])
    gt_counts = [std.desc_counts.get(t, 0) for t in ALL_TYPES]
    wedges_gt, texts_gt, autotexts_gt = ax_pie_gt.pie(
        gt_counts, labels=short_labels, autopct="%1.0f%%",
        colors=pie_colors_list[:len(ALL_TYPES)], startangle=90,
        textprops={"fontsize": 7},
    )
    for at in autotexts_gt:
        at.set_fontweight("bold")
    ax_pie_gt.set_title("GT (标准)", fontsize=9, fontweight="bold", color=colors["GT"])

    # Model pies
    for idx, mn in enumerate(models_with_data):
        ax_pie = fig.add_subplot(pie_gs[0, idx + 1])
        e = all_eval[mn]
        pred_counts = [e["per_type"][t]["pred_count"] for t in ALL_TYPES]
        wedges, texts, autotexts = ax_pie.pie(
            pred_counts, labels=short_labels, autopct="%1.0f%%",
            colors=pie_colors_list[:len(ALL_TYPES)], startangle=90,
            textprops={"fontsize": 7},
        )
        for at in autotexts:
            at.set_fontweight("bold")
        # 高亮幻觉类型：pred != gt 时红色边框 + 斜线
        for i, t in enumerate(ALL_TYPES):
            gt_c = std.desc_counts.get(t, 0)
            pred_c = e["per_type"][t]["pred_count"]
            if pred_c != gt_c:
                wedges[i].set_edgecolor("red")
                wedges[i].set_linewidth(2.5)
                wedges[i].set_hatch("//")
        # 高幻觉率模型标题标红
        hall_rate = e["detection"].get("hallucination_rate", 0)
        total_pred = e["detection"]["tp"] + e["detection"]["fp"]
        title_color = "red" if (hall_rate > 0.20 or total_pred > len(std.desc_components) * 1.5) else colors.get(mn, "#333")
        ax_pie.set_title(mn, fontsize=9, fontweight="bold", color=title_color)

    # Row 4: 识别准确率 + 结构理解
    ax_det = fig.add_subplot(gs[3, 0])
    det_headers = ["", "P", "R", "F1", "TP", "FP", "FN", "幻觉率", "CntScore"]
    det_rows = []
    for mn in model_order:
        if mn not in all_eval:
            continue
        d = all_eval[mn]["detection"]
        det_rows.append([
            mn, f"{d['precision']:.3f}", f"{d['recall']:.3f}", f"{d['f1']:.3f}",
            str(d['tp']), str(d['fp']), str(d['fn']),
            f"{d.get('hallucination_rate', 0):.3f}",
            f"{all_eval[mn]['count_score']:.3f}",
        ])
    render_table(ax_det, "一、识别准确率", det_headers, det_rows,
                 [1.8, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.2])

    ax_struct = fig.add_subplot(gs[3, 1])
    struct_headers = ["", "顺序(L)", "顺序(R)", "父子模块", "连接Acc", "位置匹配"]
    struct_rows = []
    for mn in model_order:
        if mn not in all_eval:
            continue
        s = all_eval[mn]["structure"]
        struct_rows.append([
            mn, f"{s['order_left']:.3f}", f"{s['order_right']:.3f}",
            f"{s['parent_acc']:.3f}", f"{s['connection_acc']:.3f}",
            f"{s.get('position_acc', 0):.3f}",
        ])
    render_table(ax_struct, "二、结构理解能力", struct_headers, struct_rows,
                 [1.8, 1.2, 1.2, 1.2, 1.2, 1.2])

    # Row 5: 定位能力 + 综合评分柱状图
    ax_loc = fig.add_subplot(gs[4, 0])
    loc_headers = ["", "区域命中", "Avg IoU", "偏移(px)", "归一化偏移", "Bbox命中"]
    loc_rows = []
    for mn in model_order:
        if mn not in all_eval:
            continue
        l = all_eval[mn]["localization"]
        loc_rows.append([
            mn, f"{l['region_hit_rate']:.3f}", f"{l['avg_iou']:.4f}",
            f"{l['avg_offset']:.1f}", f"{l['norm_offset']:.4f}",
            f"{l['hit_rate']:.3f}",
        ])
    render_table(ax_loc, "三、定位能力 (区域一致性主导)", loc_headers, loc_rows,
                 [1.8, 1.3, 1.3, 1.3, 1.5, 1.3])

    # 综合评分柱状图 (recognition_score + final_score + localization_score + eng_score)
    ax_overall = fig.add_subplot(gs[4, 1])
    colors_list = [colors.get(mn, "#333") for mn in models_with_data]
    x = np.arange(len(models_with_data))
    width = 0.18

    final_scores = [all_eval[mn]["overall"] for mn in models_with_data]
    rec_scores = [all_eval[mn].get("recognition_score", 0) for mn in models_with_data]
    loc_scores = [all_eval[mn].get("localization_score", 0) for mn in models_with_data]
    eng_scores = [all_eval[mn].get("engineering", {}).get("eng_score", 1) for mn in models_with_data]

    bars1 = ax_overall.bar(x - 1.5 * width, rec_scores, width, label="识别主导分",
                           color=colors_list, alpha=0.85, edgecolor="black", linewidth=0.8)
    bars2 = ax_overall.bar(x - 0.5 * width, final_scores, width, label="综合得分",
                           color=colors_list, alpha=0.50, edgecolor="black", linewidth=0.8, hatch="//")
    bars3 = ax_overall.bar(x + 0.5 * width, loc_scores, width, label="定位分",
                           color=colors_list, alpha=0.30, edgecolor="black", linewidth=0.8, hatch="\\\\")
    bars4 = ax_overall.bar(x + 1.5 * width, eng_scores, width, label="工程得分",
                           color=colors_list, alpha=0.20, edgecolor="black", linewidth=0.8, hatch="..")

    for bar, score in zip(bars1, rec_scores):
        ax_overall.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{score:.3f}", ha="center", fontsize=7, fontweight="bold")
    for bar, score in zip(bars2, final_scores):
        ax_overall.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{score:.3f}", ha="center", fontsize=7)

    ax_overall.set_xticks(x)
    ax_overall.set_xticklabels(models_with_data)
    ax_overall.set_ylim(0, 1.15)
    ax_overall.set_title("四、综合评分 (识别主导 90% + 定位 10%)", fontsize=12, fontweight="bold")
    ax_overall.set_ylabel("Score")
    ax_overall.legend(loc="upper right", fontsize=6)
    ax_overall.grid(axis="y", alpha=0.3)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# JSON 报告（核心数据）
# ============================================================

def generate_json_report(results: list, out_path: Path):
    """
    输出结构化 JSON 评测结果（核心数据，非渲染）。
    results 为 EvaluationResult 对象列表。
    """
    json_data = {
        "meta": {
            "total_images": len({r.image_name for r in results}),
            "total_models": len({r.model_name for r in results}),
            "total_evaluations": len(results),
        },
        "evaluations": [r.to_json_dict() for r in results],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        import json
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"已生成 JSON 报告: {out_path}")