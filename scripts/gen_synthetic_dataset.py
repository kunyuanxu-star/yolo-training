"""
生成合成网球检测数据集（无需下载，立即可用）。

用于快速验证训练流程，可用真实数据替换。

运行方式：
    python scripts/gen_synthetic_dataset.py
"""
from __future__ import annotations

import random
import math
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "-q"])
    from PIL import Image, ImageDraw

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("dataset/Tennis Ball Obj Det.v1i.yolov8")
IMG_SIZE = 640
TRAIN_N = 300
VAL_N = 80
TEST_N = 40
RANDOM_SEED = 42

BG_COLORS = [
    (34, 139, 34),   # 草地绿
    (0, 100, 0),     # 深绿草地
    (165, 42, 42),   # 红土场
    (139, 90, 43),   # 棕色场地
    (200, 200, 200), # 灰色硬地
    (50, 120, 200),  # 蓝色硬地
    (255, 255, 255), # 白色背景
]
# 网球颜色范围（荧光黄绿）
BALL_COLORS = [
    (204, 255, 0),
    (180, 220, 0),
    (220, 255, 30),
    (200, 250, 10),
]


def random_bg(size: int) -> Image.Image:
    """生成随机背景（纯色或简单纹理）"""
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    base = random.choice(BG_COLORS)
    # 添加随机噪声
    for y in range(0, size, 8):
        for x in range(0, size, 8):
            noise = random.randint(-20, 20)
            r = max(0, min(255, base[0] + noise))
            g = max(0, min(255, base[1] + noise))
            b = max(0, min(255, base[2] + noise))
            draw.rectangle([x, y, x + 8, y + 8], fill=(r, g, b))
    return img


def draw_ball(
    img: Image.Image, cx: float, cy: float, radius: float
) -> tuple[float, float, float, float]:
    """在图像上绘制网球，返回 YOLO 格式 (cx_n, cy_n, w_n, h_n)"""
    draw = ImageDraw.Draw(img)
    size = img.width

    color = random.choice(BALL_COLORS)
    # 主球体
    x0, y0 = cx - radius, cy - radius
    x1, y1 = cx + radius, cy + radius
    draw.ellipse([x0, y0, x1, y1], fill=color, outline=(150, 180, 0), width=2)

    # 白色曲线（模拟接缝）
    seam_color = (255, 255, 255, 180)
    draw.arc([x0 + 4, y0 + 4, x1 - 4, y1 - 4],
             start=30, end=150, fill=(255, 255, 255), width=2)
    draw.arc([x0 + 4, y0 + 4, x1 - 4, y1 - 4],
             start=210, end=330, fill=(255, 255, 255), width=2)

    # YOLO 归一化坐标
    cx_n = cx / size
    cy_n = cy / size
    w_n = (2 * radius) / size
    h_n = (2 * radius) / size
    return cx_n, cy_n, w_n, h_n


def generate_sample(img_size: int) -> tuple[Image.Image, list[tuple]]:
    """生成一张图和对应标签"""
    img = random_bg(img_size)
    labels = []
    n_balls = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]

    for _ in range(n_balls):
        radius = random.randint(15, 60)
        cx = random.randint(radius + 5, img_size - radius - 5)
        cy = random.randint(radius + 5, img_size - radius - 5)
        box = draw_ball(img, cx, cy, radius)
        labels.append(box)

    return img, labels


def write_split(split_dir: Path, n: int) -> None:
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        img, labels = generate_sample(IMG_SIZE)
        stem = f"tennis_{i:05d}"
        img.save(img_dir / f"{stem}.jpg", quality=90)
        with open(lbl_dir / f"{stem}.txt", "w") as f:
            for cx, cy, w, h in labels:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    print(f"  {split_dir.name}: {n} 张图 -> {img_dir}")


def main() -> None:
    random.seed(RANDOM_SEED)
    print(f"生成合成网球数据集到: {DATASET_ROOT.resolve()}")

    write_split(DATASET_ROOT / "train", TRAIN_N)
    write_split(DATASET_ROOT / "valid", VAL_N)
    write_split(DATASET_ROOT / "test", TEST_N)

    print(f"\n完成！共 {TRAIN_N + VAL_N + TEST_N} 张图像。")
    print("提示：这是合成数据集，用于验证流程。训练效果有限，")
    print("      建议替换为真实网球图像以获得更好结果。")


if __name__ == "__main__":
    main()
