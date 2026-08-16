"""Run YOLOv8n inference with the six trained single-class models.

The six models live in ``models/`` and are named ``best_<class>.pt``:

    apple, banana, paper_cup, orange, doll, plastic_bottle

Examples:
    # Run every model on one image
    python scripts/infer.py --source demo.jpg --all

    # Run a single class model
    python scripts/infer.py --class apple --source apple.jpg

    # Live camera with the plastic bottle model
    python scripts/infer.py --class plastic_bottle --source 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


CLASSES = [
    "apple", "banana", "paper_cup", "orange", "doll", "plastic_bottle",
    "book", "chair", "cup", "computer_keyboard", "laptop",
]
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def model_path(cls: str) -> Path:
    p = MODELS_DIR / f"best_{cls}.pt"
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    return p


def run_one(cls: str, source: str, conf: float, device: str, save: bool) -> None:
    print(f"[{cls}] loading {model_path(cls)}")
    model = YOLO(model_path(cls))
    results = model.predict(
        source=source,
        conf=conf,
        device=device,
        save=save,
        project=str(ROOT / "runs" / "infer"),
        name=cls,
        exist_ok=True,
    )
    for r in results:
        names = r.names
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            print(f"[{cls}] no detections")
            continue
        for box in boxes:
            label = names.get(int(box.cls), "object")
            conf_val = float(box.conf)
            xyxy = [round(float(v), 1) for v in box.xyxy[0]]
            print(f"[{cls}] {label} conf={conf_val:.3f} xyxy={xyxy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with six YOLOv8n models")
    parser.add_argument("--source", required=True, help="Image, video, folder, or camera id.")
    parser.add_argument("--class", dest="cls", choices=CLASSES, help="Which class model to run.")
    parser.add_argument("--all", action="store_true", help="Run all six models.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", default="", help="Device: cpu, 0, cuda:0... (default auto).")
    parser.add_argument("--save", action="store_true", help="Save annotated outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all and args.cls:
        raise SystemExit("Use either --class or --all, not both.")
    targets = CLASSES if args.all else [args.cls]
    if not args.all and not args.cls:
        raise SystemExit("Specify --class <name> or --all.")
    for cls in targets:
        run_one(cls, args.source, args.conf, args.device, args.save)


if __name__ == "__main__":
    main()
