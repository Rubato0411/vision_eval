import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ============================================================
# 配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
IMAGE_FILE = BASE_DIR / "figure1.png"
OUTPUT_FILE = BASE_DIR / "figure1_comparison.png"

# 四个 JSON 文件及其对应的模型名称
MODELS = {
    "ds-f1-output.json":      "DeepSeek",
    "db-f1-output.json":      "DB",
    "gpt5.5-f1-output.json":  "GPT-5.5",
    "gemini-f1-output.json":  "Gemini",
}

# 每个模型一种主色
COLORS = {
    "DeepSeek":  "#E74C3C",
    "DB":        "#2ECC71",
    "GPT-5.5":   "#3498DB",
    "Gemini":    "#9B59B6",
}

# 每个组件一种线型
COMPONENT_STYLES = {
    "Multi-Head Attention": {"linestyle": "-",  "linewidth": 2.5},
    "Feed Forward":         {"linestyle": "--", "linewidth": 2.5},
    "Add & Norm":           {"linestyle": ":",  "linewidth": 3.0},
}


# ============================================================
# JSON 解析 —— 四种不同结构归一化
# ============================================================

def parse_json(filepath: Path) -> list[dict]:
    """
    返回统一格式: [{"label": str, "bbox": [x1, y1, x2, y2]}, ...]
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []

    # gpt5.5: 顶层就是 list
    if isinstance(data, list):
        for obj in data:
            items.append({
                "label": obj["component"],
                "bbox":  obj["bounding_box"],
            })

    # 其他三个都是 dict，但数组 key 不同
    elif isinstance(data, dict):
        # 统一取到 components 列表
        components = data.get("core_components") or data.get("components") or []
        for obj in components:
            items.append({
                "label": obj.get("name") or obj.get("component") or "unknown",
                "bbox":  obj.get("bbox") or obj.get("bounding_box") or [0, 0, 0, 0],
            })

    return items


# ============================================================
# 绘制
# ============================================================

def draw_one(ax, img: Image.Image, model_name: str, items: list[dict]):
    ax.imshow(img)
    ax.set_title(model_name, fontsize=13, fontweight="bold", color=COLORS[model_name])
    ax.axis("off")

    color = COLORS[model_name]

    for item in items:
        label = item["label"]
        x1, y1, x2, y2 = item["bbox"]
        w, h = x2 - x1, y2 - y1
        style = COMPONENT_STYLES.get(label, {"linestyle": "-", "linewidth": 2})

        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        # 标签文字
        ax.text(
            x1, y1 - 6, f"{label} [{x1},{y1},{x2},{y2}]",
            fontsize=7, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor=color, linewidth=0.8),
        )


def main():
    if not IMAGE_FILE.exists():
        raise FileNotFoundError(f"找不到图片: {IMAGE_FILE}\n请把 figure1.png 放到脚本同目录。")

    img = Image.open(IMAGE_FILE).convert("RGB")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    for ax, (json_file, model_name) in zip(axes, MODELS.items()):
        filepath = BASE_DIR / json_file
        if not filepath.exists():
            print(f"[警告] 找不到 {json_file}，跳过 {model_name}")
            ax.axis("off")
            continue

        items = parse_json(filepath)
        draw_one(ax, img, model_name, items)

    # 全局图例
    handles = []
    for comp, style in COMPONENT_STYLES.items():
        handles.append(
            patches.Patch(
                edgecolor="black",
                facecolor="none",
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                label=comp,
            )
        )
    for model_name, color in COLORS.items():
        handles.append(
            patches.Patch(edgecolor=color, facecolor="none", linewidth=2.5, label=model_name)
        )

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已生成: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()