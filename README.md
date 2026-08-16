# YOLO Training Platform

一个完整的 YOLO 目标检测训练平台，提供 CLI 命令行工具和 Web 界面两种使用方式。支持数据标注管理、离线数据增强、训练预设、超参调优（HPO）、模型对比、模型量化导出等功能。仓库自带 11 个已经训练好的 YOLOv8n 单类别检测模型（`.pt` 与 `.onnx`），可直接用于推理。

## 已训练模型（11 类 YOLOv8n）

本仓库包含 11 个独立的 YOLOv8n 单类别检测模型（`models/best_<class>.pt`，同时提供同名 `.onnx` 导出）。

| 类别 | 权重 (pt) | 导出 (onnx) | mAP50 | 训练图/验证图 |
| --- | --- | --- | --- | --- |
| apple | `models/best_apple.pt` | `models/best_apple.onnx` | 0.917 | 160 / 40 |
| banana | `models/best_banana.pt` | `models/best_banana.onnx` | 0.634 | 351 / 64 |
| paper_cup | `models/best_paper_cup.pt` | `models/best_paper_cup.onnx` | 0.965 | 171 / 30 |
| orange | `models/best_orange.pt` | `models/best_orange.onnx` | 0.786 | 469 / 64 |
| doll | `models/best_doll.pt` | `models/best_doll.onnx` | 0.830 | 500 / 64 |
| plastic_bottle | `models/best_plastic_bottle.pt` | `models/best_plastic_bottle.onnx` | 0.995 | 523 / 92 |
| book | `models/best_book.pt` | `models/best_book.onnx` | 0.712 | 500 / 64 |
| chair | `models/best_chair.pt` | `models/best_chair.onnx` | 0.831 | 500 / 64 |
| cup | `models/best_cup.pt` | `models/best_cup.onnx` | 0.932 | 425 / 64 |
| computer_keyboard | `models/best_computer_keyboard.pt` | `models/best_computer_keyboard.onnx` | 0.929 | 500 / 64 |
| laptop | `models/best_laptop.pt` | `models/best_laptop.onnx` | 0.903 | 500 / 64 |

数据来源与说明见 [datasets/README.md](datasets/README.md)。数据集图片体积较大，本仓库仅随库分发 `data.yaml` 与标签（`datasets/<class>/labels/`），训练图片可从 Open Images 等公开来源按 `data.yaml` 中的类别重建。

## 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | ≥ 3.10 | 训练引擎 + 后端 |
| Node.js | ≥ 20 | 前端（Vite + React，仅 Web 界面需要） |
| Docker（可选） | ≥ 24 | 一键启动全部服务 |
| GPU（可选） | - | 训练/加速推理；无 GPU 时用 CPU |

## 快速开始

### 方式一：Docker 开发模式（推荐，一键启动全部服务）

```bash
docker compose -f docker-compose.dev.yml up -d
```

首次启动会自动构建镜像（pip install + npm install），完成后访问：
- Web 界面: http://localhost:3000
- 后端 API: http://localhost:8000/docs

### 方式二：Docker 生产模式

```bash
docker compose up -d
```

前后端统一由 Nginx 托管在 http://localhost:8000。

### 方式三：手动启动（本地开发）

后端默认使用 SQLite（零依赖），可直接从 `backend/` 启动；依赖数据库（PostgreSQL + Redis）和前端时请参考 `docker-compose.dev.yml` 中的服务定义。

## 本地运行

### 环境配置

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-gpu.txt   # CUDA 12.6；无 GPU 用 requirements.txt
```

### 本地推理

```bash
# 图片推理（demo.jpg 为仓库自带橙子样例图，可换成任意图片/视频）
python scripts/infer.py --class orange --source demo.jpg --save

# 十一个模型都跑一遍
python scripts/infer.py --all --source path/to/image_or_video.mp4 --save

# 摄像头实时检测
python scripts/infer.py --class plastic_bottle --source 0 --device 0

# ONNX 推理（无需 torch）
python scripts/infer_onnx.py --class laptop --source path/to/image.jpg --save

# 直接用 Ultralytics API
python - <<'PY'
from ultralytics import YOLO
model = YOLO("models/best_orange.pt")
results = model("demo.jpg", conf=0.25, device=0)
print(results[0].boxes)
PY
```

`scripts/infer.py` 支持 `--class <name>`（apple / banana / paper_cup / orange / doll / plastic_bottle / book / chair / cup / computer_keyboard / laptop）与 `--all` 两种模式，`--device` 可传 `cpu`、`0`、`cuda:0`。

### 本地训练

```bash
python -m training_engine.train \
  --model yolov8n.pt \
  --data datasets/banana/data.yaml \
  --epochs 100 --imgsz 640 --batch 16 --device 0
```

### 本地 Docker 复现训练

```bash
docker build -f docker/Dockerfile.gpu -t yolo-gpu .
docker run --gpus all -it --rm \
  -e YOLO_EPOCHS=100 -e YOLO_BATCH=16 -e YOLO_IMGSZ=640 \
  -v "$(pwd)/runs:/workspace/runs" \
  yolo-gpu
```

## 常用操作速查

```bash
# 训练
python -m training_engine.train --model yolov8n.pt --data datasets/banana/data.yaml --epochs 100

# 小数据集训练
python -m training_engine.train --model yolov8n.pt --data datasets/cup/data.yaml --preset aggressive

# 超参调优
python -m training_engine.tune --data datasets/cup/data.yaml --epochs 30 --iterations 100

# 推理
python -m training_engine.predict --model models/best_apple.pt --source datasets/apple/val/images/apple_00008.jpg --save

# 通用推理入口
python scripts/infer.py --class banana --source datasets/banana/val/images/00000.jpg --save

# 量化
python -m training_engine.quantize --model models/best_apple.pt --format int8

# 查看训练结果
cat runs/*/train*/results.csv | tail -5

# 启动 Web 界面
docker compose -f docker-compose.dev.yml up -d
```

## 目录结构

```
├── training_engine/         # Python 训练引擎（CLI）
├── backend/                 # FastAPI 后端
├── frontend/                # React + TypeScript 前端
├── scripts/                 # 训练 / 推理 / 数据脚本
├── configs/                 # 数据集配置（YAML）
├── datasets/                # 各单类数据集（data.yaml + labels）
├── models/                  # 已训练模型（best_<class>.pt / .onnx）
├── docs/                    # 文档
├── rknn/                    # RKNN 芯片部署工具
├── docker-compose.yml       # 生产部署
└── docker-compose.dev.yml   # 开发环境
```

## License

[MIT](LICENSE)
