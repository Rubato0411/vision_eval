"""
st_app.py — Streamlit 评测面板
===============================
上传接口 + 评分仪表盘 + 原始数据下钻 + bbox 可视化。
依赖：streamlit, matplotlib, numpy, pillow
"""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ---- 复用现有评测管线 ----
from core import Component, ModelData, EvaluationResult, ALL_TYPES, box_center, normalize_position, _side_to_parent
from matcher import greedy_match, greedy_match_bbox
from metrics import (
    compute_detection_metrics, compute_per_type_metrics, compute_count_score,
    compute_structure_metrics, compute_localization_metrics,
    compute_engineering_score,
)
from scorer import compute_overall_score

# ============================================================
# In-memory 解析器（替代 parser.py 中依赖文件路径的函数）
# ============================================================

def _parse_gt_from_dict(data: dict) -> tuple[ModelData, float]:
    """从 dict 解析 GT，返回 (ModelData, split_x)。"""
    components = []
    for item in data["components"]:
        label = item["name"]
        bbox = item["bbox"]
        parent = item.get("parent", "")
        side = item.get("side", "")
        vert = item.get("vert", "")
        pos = f"{vert}-{side}" if vert and side else ""
        components.append(Component(
            id=len(components) + 1, label=label, position_desc=pos,
            side=side, vert=vert, parent=parent, bbox=bbox,
        ))

    vert_order = {"top": 0, "middle": 1, "bottom": 2}
    components.sort(key=lambda c: (0 if c.side == "left" else 1, vert_order.get(c.vert, 99)))
    for i, c in enumerate(components):
        c.id = i + 1

    split_x = _compute_content_split_x(components)
    counts = {t: sum(1 for c in components if c.label == t) for t in ALL_TYPES}

    return ModelData(
        name="GT", image_name="upload",
        desc_components=components, bbox_components=components, desc_counts=counts,
    ), split_x


def _compute_content_split_x(components: list[Component]) -> float:
    """最大相邻间距法求左右分界点。"""
    cxs = sorted([
        box_center(c.bbox)[0] for c in components
        if c.bbox and not all(v == 0 for v in c.bbox)
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


def _get_region(cx, cy, img_w, img_h, split_x=None):
    """根据中心点确定语义区域。"""
    if split_x is None:
        split_x = img_w / 2
    h_third = img_h / 3
    vert = "top" if cy < h_third else ("middle" if cy < 2 * h_third else "bottom")
    side = "left" if cx < split_x else "right"
    return f"{vert}-{side}"


def _parse_desc_from_dict(data: dict) -> tuple[list[Component], dict[str, int]]:
    """从 dict 解析模型描述 JSON。"""
    components = []
    for item in data.get("components", []):
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
    counts = {t: sum(1 for c in components if c.label == t) for t in ALL_TYPES}
    return components, counts


def _parse_bbox_from_dict(data: dict, img_w: int, img_h: int,
                          split_x: float) -> list[Component]:
    """从 dict 解析模型 bbox JSON。"""
    if isinstance(data, list):
        items = [{"label": obj.get("component") or obj.get("label") or "unknown",
                   "bbox": obj.get("bounding_box") or obj.get("bbox") or [0, 0, 0, 0]}
                 for obj in data if isinstance(obj, dict)]
    elif isinstance(data, dict):
        items = (data.get("detections") or data.get("core_components")
                 or data.get("components") or [])
    else:
        items = []

    components = []
    for i, item in enumerate(items):
        label = item.get("label") or item.get("name") or item.get("component") or "unknown"
        bbox = item.get("bbox") or item.get("bounding_box") or [0, 0, 0, 0]
        cx, cy = box_center(bbox)
        region = _get_region(cx, cy, img_w, img_h, split_x)
        parts = region.split("-")
        components.append(Component(
            id=item.get("id", i + 1), label=label, position_desc=region,
            side=parts[1], vert=parts[0], parent=_side_to_parent(parts[1]), bbox=bbox,
        ))
    return components


def _check_counts_consistency_from_dict(data: dict, parsed_counts: dict) -> dict:
    """对比 JSON 声明的 counts 与解析结果。"""
    declared = data.get("counts", {})
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
    return {"declared_counts": declared, "parsed_counts": parsed_counts,
            "match": match, "score": score}


# ============================================================
# 单模型评估（复用现有管线）
# ============================================================

def evaluate_one_model(name: str, desc_dict: dict, bbox_dict: dict | None,
                       gt: ModelData, img_w: int, img_h: int,
                       split_x: float) -> EvaluationResult:
    """对单个模型执行完整评测，返回 EvaluationResult。"""
    desc_comps, desc_counts = _parse_desc_from_dict(desc_dict)
    bbox_comps = _parse_bbox_from_dict(bbox_dict, img_w, img_h, split_x) if bbox_dict else []

    consistency = _check_counts_consistency_from_dict(desc_dict, desc_counts)
    eng = compute_engineering_score(bbox_comps, consistency, name)

    pairs_desc = greedy_match(desc_comps, gt.desc_components)
    det = compute_detection_metrics(pairs_desc, len(gt.desc_components))
    per_type = compute_per_type_metrics(desc_comps, gt.desc_components)
    count_score = compute_count_score(desc_counts, gt.desc_counts)

    struct = compute_structure_metrics(desc_comps, gt.desc_components, pairs_desc)

    bbox_pairs = greedy_match_bbox(bbox_comps, gt.bbox_components)
    loc = compute_localization_metrics(bbox_pairs, img_w, img_h, split_x)

    overall, rec_score, loc_score, structure_sub = compute_overall_score(
        det, count_score, struct, loc)

    hall_rate = det.get("hallucination_rate", 0)
    total_pred = det["tp"] + det["fp"]
    hall_warn = hall_rate > 0.20 or total_pred > len(gt.desc_components) * 1.5
    logic_bad = not consistency.get("match", True)

    return EvaluationResult(
        model_name=name, image_name="upload",
        precision=det["precision"], recall=det["recall"], f1=det["f1"],
        tp=det["tp"], fp=det["fp"], fn=det["fn"],
        hallucination_rate=hall_rate, count_score=count_score,
        order_left=struct["order_left"], order_right=struct["order_right"],
        order_overall=struct["order_overall"],
        parent_acc=struct["parent_acc"], connection_acc=struct["connection_acc"],
        position_acc=struct.get("position_acc", 0),
        region_hit_rate=loc["region_hit_rate"], avg_iou=loc["avg_iou"],
        avg_offset=loc["avg_offset"], norm_offset=loc["norm_offset"],
        hit_rate=loc["hit_rate"],
        recognition_score=rec_score, localization_score=loc_score,
        overall=overall, eng_score=eng["eng_score"],
        consistency_score=consistency.get("score", 1.0),
        label_error_rate=eng["label_error_rate"],
        bbox_error_rate=eng["bbox_error_rate"],
        hallucination_warning=hall_warn, logic_inconsistent=logic_bad,
        per_type=per_type, desc_pairs=pairs_desc, bbox_pairs=bbox_pairs,
        structure_sub=structure_sub,
    )


# ============================================================
# 可视化工具
# ============================================================

MODEL_COLORS = [
    "#2ECC71", "#E74C3C", "#3498DB", "#9B59B6", "#F39C12",
    "#1ABC9C", "#E91E63", "#00BCD4", "#FF5722", "#607D8B",
]


def draw_bbox_overlay(fig, ax, img_array, components, color, label_prefix="",
                      alpha=0.9, linewidth=2):
    """在一张图上绘制 bbox 叠加。"""
    ax.imshow(img_array)
    ax.axis("off")
    for c in components:
        x1, y1, x2, y2 = c.bbox
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            continue
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=linewidth, edgecolor=color,
                                 facecolor="none", alpha=alpha)
        ax.add_patch(rect)
        short = (c.label.replace("Multi-Head Attention", "MHA")
                 .replace("Feed Forward", "FF")
                 .replace("Add & Norm", "A&N"))
        label_text = f"{label_prefix}{short}" if label_prefix else short
        ax.text(x1 + 2, y1 - 4, label_text, fontsize=6, color=color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                          alpha=0.85, edgecolor=color, linewidth=0.5))


def build_bbox_figure(img_array, gt_comps, model_bbox_map, model_colors):
    """生成 GT + 各模型 bbox 叠加图。"""
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_bbox_overlay(fig, ax, img_array, gt_comps, "#D4A017",
                      label_prefix="GT:", alpha=0.7, linewidth=3)

    from matplotlib.lines import Line2D
    handles = [patches.Patch(edgecolor="#D4A017", facecolor="none",
                             lw=2.5, label="GT")]
    for model_name, comps in model_bbox_map.items():
        if not comps:
            continue
        color = model_colors.get(model_name, "#333333")
        draw_bbox_overlay(fig, ax, img_array, comps, color,
                          label_prefix=f"{model_name}:", alpha=0.55, linewidth=1.8)
        handles.append(patches.Patch(edgecolor=color, facecolor="none",
                                     lw=2, label=model_name))
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    return fig


# ============================================================
# Streamlit 页面
# ============================================================


def main():
    if "gt" not in st.session_state:
        st.session_state.gt = None  # 或者根据你的业务逻辑赋予初始值
    
    if "results" not in st.session_state:
        st.session_state.results = []
    st.set_page_config(page_title="Vision Model Evaluator", layout="wide")
    st.title("Vision Model Evaluator")
    st.caption("上传 GT 标注 + 参考图片 + 模型输出 JSON，自动评测并下钻分析")

    # ---- 初始化 session_state ----
    defaults = {
        "gt_data": None, "gt_dict": None,
        "img_array": None, "img_w": 1200, "img_h": 900,
        "model_slots": [],
        "results": None,
        "run_clicked": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ============================================================
    # 侧边栏：上传区
    # ============================================================

    with st.sidebar:
        st.header("1. Ground Truth")
        gt_file = st.file_uploader("GT JSON (standard-fi.json)", type=["json"],
                                   key="gt_upload",
                                   help="包含 components 数组，每项有 name/bbox/parent/side/vert")

        img_file = st.file_uploader("参考图片 (figure1.png)", type=["png", "jpg", "jpeg"],
                                    key="img_upload")

        st.divider()
        st.header("2. 模型输出")

        # 动态模型槽位管理
        if st.button("+ 添加模型"):
            st.session_state.model_slots.append({"name": f"Model {len(st.session_state.model_slots) + 1}",
                                                  "desc": None, "bbox": None})

        remove_idx = None
        for i, slot in enumerate(st.session_state.model_slots):
            with st.expander(f"模型 #{i + 1}: {slot.get('name', '')}", expanded=(i < 3)):
                slot["name"] = st.text_input("模型名称", value=slot.get("name", ""),
                                             key=f"name_{i}", label_visibility="collapsed",
                                             placeholder="输入模型名称")
                slot["desc"] = st.file_uploader("描述 JSON (含 position_desc)", type=["json"],
                                                key=f"desc_{i}")
                slot["bbox"] = st.file_uploader("Bbox JSON (可选)", type=["json"],
                                                key=f"bbox_{i}")
                if st.button("删除此模型", key=f"del_{i}"):
                    remove_idx = i

        if remove_idx is not None:
            st.session_state.model_slots.pop(remove_idx)
            st.rerun()

        st.divider()
        run_btn = st.button("▶ 运行评测", type="primary", use_container_width=True)

    # ============================================================
    # 执行评测
    # ============================================================

    if run_btn:
        errors = []
        if not gt_file:
            errors.append("请上传 GT JSON")
        if not img_file:
            errors.append("请上传参考图片")

        valid_slots = [s for s in st.session_state.model_slots
                       if s.get("name", "").strip() and s.get("desc")]
        if not valid_slots:
            errors.append("至少需要一个模型（名称 + 描述 JSON）")

        if errors:
            for e in errors:
                st.error(e)
        else:
            with st.spinner("正在评测..."):
                try:
                    # 解析 GT
                    gt_dict = json.loads(gt_file.read())
                    img = Image.open(img_file)
                    img_w, img_h = img.size
                    img_array = np.array(img.convert("RGB"))
                    gt, split_x = _parse_gt_from_dict(gt_dict)

                    # 逐模型评测
                    results = []
                    for slot in valid_slots:
                        desc_dict = json.loads(slot["desc"].read())
                        bbox_dict = json.loads(slot["bbox"].read()) if slot["bbox"] else None
                        r = evaluate_one_model(
                            slot["name"].strip(), desc_dict, bbox_dict,
                            gt, img_w, img_h, split_x,
                        )
                        results.append(r)

                    st.session_state.results = results
                    st.session_state.gt = gt
                    st.session_state.gt_dict = gt_dict
                    st.session_state.img_array = img_array
                    st.session_state.img_w = img_w
                    st.session_state.img_h = img_h
                    st.session_state.split_x = split_x
                    st.session_state.run_clicked = True

                    st.success(f"评测完成：{len(results)} 个模型")
                    st.rerun()

                except Exception:
                    st.error(f"评测失败：\n```\n{traceback.format_exc()}\n```")

    # ============================================================
    # 结果展示
    # ============================================================

    results: list[EvaluationResult] | None = st.session_state.results
    if not results:
        st.info("上传 GT + 模型文件后点击「运行评测」")
        st.stop()

    gt = st.session_state.gt
    gt_dict = st.session_state.gt_dict
    img_array = st.session_state.img_array
    img_w = st.session_state.img_w
    img_h = st.session_state.img_h
    split_x = st.session_state.split_x

    model_names = [r.model_name for r in results]
    model_colors = {mn: MODEL_COLORS[i % len(MODEL_COLORS)] for i, mn in enumerate(model_names)}

    # ---- 顶部概览卡片 ----
    st.header("评测概览")

    cols = st.columns(min(len(results), 5))
    for i, r in enumerate(results):
        col_idx = i % len(cols)
        with cols[col_idx]:
            color = model_colors[r.model_name]
            st.metric(
                label=f"{r.model_name}",
                value=f"{r.overall:.3f}",
                delta=f"F1={r.f1:.3f} | Rec={r.recognition_score:.3f}",
            )
            if r.hallucination_warning:
                st.warning(f"幻觉率 {r.hallucination_rate:.1%}")
            if r.logic_inconsistent:
                st.warning("逻辑不一致")

    # ---- Tab 导航 ----
    tab_overview, tab_recognition, tab_structure, tab_localization, tab_drilldown, tab_bbox = st.tabs([
        "总览", "识别准确率", "结构理解", "定位能力", "原始数据下钻", "Bbox 可视化",
    ])

    # ============================================================
    # Tab 1: 总览表格
    # ============================================================

    with tab_overview:
        st.subheader("综合指标总览")
        rows = []
        for r in results:
            rows.append({
                "模型": r.model_name,
                "Overall": f"{r.overall:.4f}",
                "Recognition": f"{r.recognition_score:.4f}",
                "Localization": f"{r.localization_score:.4f}",
                "Engineering": f"{r.eng_score:.4f}",
                "F1": f"{r.f1:.3f}",
                "P": f"{r.precision:.3f}",
                "R": f"{r.recall:.3f}",
                "TP": r.tp, "FP": r.fp, "FN": r.fn,
                "Halluc%": f"{r.hallucination_rate:.1%}",
                "Count": f"{r.count_score:.3f}",
                "Consistency": f"{r.consistency_score:.3f}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True,
                     column_config={
                         "Overall": st.column_config.NumberColumn(format="%.4f"),
                         "F1": st.column_config.NumberColumn(format="%.3f"),
                     })

    # ============================================================
    # Tab 2: 识别准确率（分类型）
    # ============================================================

    with tab_recognition:
        st.subheader("分类型 Precision / Recall / F1")

        # 按模型展开
        for r in results:
            with st.expander(f"{r.model_name}  —  F1={r.f1:.3f}  P={r.precision:.3f}  R={r.recall:.3f}", expanded=(len(results) <= 2)):
                type_rows = []
                for t in ALL_TYPES:
                    if t in r.per_type:
                        m = r.per_type[t]
                        type_rows.append({
                            "类型": t, "P": f"{m['precision']:.3f}", "R": f"{m['recall']:.3f}",
                            "F1": f"{m['f1']:.3f}", "Pred": m["pred_count"], "GT": m["gt_count"],
                            "TP": m["tp"], "FP": m["fp"], "FN": m["fn"],
                        })
                st.dataframe(type_rows, use_container_width=True, hide_index=True)

        # 跨模型对比柱状图
        st.subheader("F1 跨模型对比")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        x = np.arange(len(ALL_TYPES))
        width = 0.8 / max(len(results), 1)
        for i, r in enumerate(results):
            f1s = [r.per_type[t]["f1"] if t in r.per_type else 0 for t in ALL_TYPES]
            ax.bar(x + i * width - (len(results) - 1) * width / 2, f1s, width,
                   label=r.model_name, color=model_colors[r.model_name], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("Multi-Head Attention", "MHA").replace("Feed Forward", "FF").replace("Add & Norm", "A&N") for t in ALL_TYPES])
        ax.set_ylabel("F1"); ax.set_ylim(0, 1.1); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # ============================================================
    # Tab 3: 结构理解
    # ============================================================

    with tab_structure:
        st.subheader("结构理解能力")
        struct_rows = []
        for r in results:
            struct_rows.append({
                "模型": r.model_name,
                "顺序(L)": f"{r.order_left:.3f}",
                "顺序(R)": f"{r.order_right:.3f}",
                "顺序Overall": f"{r.order_overall:.3f}",
                "父子Acc": f"{r.parent_acc:.3f}",
                "连接Acc": f"{r.connection_acc:.3f}",
                "位置Acc": f"{r.position_acc:.3f}",
            })
        st.dataframe(struct_rows, use_container_width=True, hide_index=True)

        # 结构分柱状图
        fig, ax = plt.subplots(figsize=(8, 3.5))
        x = np.arange(len(model_names))
        metrics_labels = ["顺序Overall", "父子Acc", "连接Acc", "位置Acc"]
        sub = {mn: [getattr(r, f) for f in ["order_overall", "parent_acc", "connection_acc", "position_acc"]]
               for r, mn in zip(results, model_names)}
        width = 0.18
        for j, (label, idx) in enumerate(zip(metrics_labels, range(4))):
            vals = [sub[mn][idx] for mn in model_names]
            ax.bar(x + j * width - 1.5 * width, vals, width, label=label, alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(model_names)
        ax.set_ylabel("Score"); ax.set_ylim(0, 1.1); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # ============================================================
    # Tab 4: 定位能力
    # ============================================================

    with tab_localization:
        st.subheader("定位能力")
        loc_rows = []
        for r in results:
            loc_rows.append({
                "模型": r.model_name,
                "区域命中": f"{r.region_hit_rate:.3f}",
                "Avg IoU": f"{r.avg_iou:.4f}",
                "中心偏移(px)": f"{r.avg_offset:.1f}",
                "归一化偏移": f"{r.norm_offset:.4f}",
                "Bbox命中": f"{r.hit_rate:.3f}",
            })
        st.dataframe(loc_rows, use_container_width=True, hide_index=True)

    # ============================================================
    # Tab 5: 原始数据下钻
    # ============================================================

    with tab_drilldown:
        st.subheader("完整评测 JSON（可展开）")

        # 全部结果 JSON
        with st.expander("eval_results.json (完整)", expanded=False):
            json_str = json.dumps({
                "meta": {"total_models": len(results), "total_evaluations": len(results)},
                "evaluations": [r.to_json_dict() for r in results],
            }, ensure_ascii=False, indent=2)
            st.download_button("下载 JSON", json_str, "eval_results.json", "application/json")
            st.code(json_str, language="json")

        # 逐模型匹配明细
        st.subheader("逐组件匹配明细")
        for r in results:
            with st.expander(f"{r.model_name} — 匹配明细"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**描述级匹配 (desc_pairs)**")
                    dp_rows = []
                    for p in r.desc_pairs:
                        pred = p["pred"]
                        gt_c = p["gt"]
                        dp_rows.append({
                            "Pred": f"{pred.label} @ {pred.position_desc}",
                            "GT": f"{gt_c.label} @ {gt_c.position_desc}" if gt_c else "—",
                            "Dist": p["distance"],
                            "Status": "✓" if gt_c else "FP",
                        })
                    st.dataframe(dp_rows, use_container_width=True, hide_index=True)
                with c2:
                    st.markdown("**Bbox 级匹配 (bbox_pairs)**")
                    bp_rows = []
                    for p in r.bbox_pairs:
                        pred = p["pred"]
                        gt_c = p["gt"]
                        bp_rows.append({
                            "Pred": f"{pred.label}",
                            "GT": f"{gt_c.label}" if gt_c else "—",
                            "IoU": f"{p['iou']:.4f}",
                            "Off(px)": f"{p['distance']:.1f}" if p["distance"] != 999 else "—",
                            "Status": "✓" if gt_c else "FP",
                        })
                    st.dataframe(bp_rows, use_container_width=True, hide_index=True)

        # GT 结构展示
        with st.expander("GT 标准结构"):
            st.json(gt_dict)

    # ============================================================
    # Tab 6: Bbox 可视化
    # ============================================================

    with tab_bbox:
        st.subheader("Bbox 叠加可视化")

        selected_models = st.multiselect(
            "选择要显示的模型", model_names, default=model_names[:min(4, len(model_names))],
            key="bbox_model_select",
        )

        if img_array is not None and selected_models:
            model_bbox_map = {}
            for r in results:
                if r.model_name in selected_models:
                    bbox_comps = [p["pred"] for p in r.bbox_pairs]
                    model_bbox_map[r.model_name] = bbox_comps

            fig = build_bbox_figure(img_array, gt.bbox_components, model_bbox_map, model_colors)
            st.pyplot(fig)
            plt.close(fig)
        elif img_array is None:
            st.warning("请先上传参考图片")

if __name__ == "__main__":
    main()
