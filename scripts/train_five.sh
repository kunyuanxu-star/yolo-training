#!/bin/sh
set -eu

CLASSES="book chair cup computer_keyboard laptop"

for cls in $CLASSES; do
  echo "=== train $cls ==="
  rm -rf "runs/$cls"
  yolo train \
    model=yolov8n.pt \
    data="datasets/$cls/data.yaml" \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    project="$(pwd)/runs/$cls" \
    name=train \
    exist_ok=True
done

echo "=== export onnx ==="
python - <<'PY'
import shutil
from pathlib import Path

from ultralytics import YOLO

CLASSES = ["book", "chair", "cup", "computer_keyboard", "laptop"]
for cls in CLASSES:
    src_dir = Path("runs") / cls / "train" / "weights"
    model = YOLO(src_dir / "best.pt")
    model.export(format="onnx", imgsz=640, dynamic=False, simplify=True, opset=17)
    shutil.copy2(src_dir / "best.pt", src_dir / f"best_{cls}.pt")
    shutil.copy2(src_dir / "best.onnx", src_dir / f"best_{cls}.onnx")
    print(f"exported {cls}: best_{cls}.pt + best_{cls}.onnx")
PY
