# YOLO Training Platform

A complete YOLO object detection training platform with both a CLI and a web UI. It supports dataset annotation, offline augmentation, training presets, hyperparameter optimization, model comparison, and model export. The repo ships 11 pre-trained YOLOv8n single-class detection models (`.pt` and `.onnx`) ready for inference.

## Trained Models (11 YOLOv8n classes)

Each model is single-class (`models/best_<class>.pt` with a matching `.onnx` export).

| Class | Weights (pt) | Export (onnx) | mAP50 | train/val images |
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

See [datasets/README.md](datasets/README.md) for data provenance. Dataset images are large, so this repo ships only `data.yaml` and labels (`datasets/<class>/labels/`); images can be rebuilt from public sources (e.g. Open Images) per the classes in `data.yaml`.

## Requirements

- Python >= 3.10
- Node.js >= 20 (web UI only)
- Docker >= 24 (optional, one-command services)
- GPU (optional; CPU works for inference)

## Quick Start

### Docker dev mode (recommended, one command for everything)

```bash
docker compose -f docker-compose.dev.yml up -d
```

- Web UI: http://localhost:3000
- API docs: http://localhost:8000/docs

### Docker production mode

```bash
docker compose up -d
```

Served by Nginx at http://localhost:8000.

### Manual local start

The backend defaults to SQLite (zero extra services). When you need PostgreSQL + Redis and the frontend, follow the service definitions in `docker-compose.dev.yml`.

## Local Usage

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-gpu.txt   # or requirements.txt for CPU
```

### Inference

```bash
python scripts/infer.py --class orange --source demo.jpg --save
python scripts/infer.py --all --source path/to/image_or_video.mp4 --save
python scripts/infer.py --class plastic_bottle --source 0 --device 0
python scripts/infer_onnx.py --class laptop --source path/to/image.jpg --save
```

### Training

```bash
python -m training_engine.train \
  --model yolov8n.pt \
  --data datasets/banana/data.yaml \
  --epochs 100 --imgsz 640 --batch 16 --device 0
```

### Docker training (reproducible)

```bash
docker build -f docker/Dockerfile.gpu -t yolo-gpu .
docker run --gpus all -it --rm \
  -e YOLO_EPOCHS=100 -e YOLO_BATCH=16 -e YOLO_IMGSZ=640 \
  -v "$(pwd)/runs:/workspace/runs" \
  yolo-gpu
```

## License

[MIT](LICENSE)
