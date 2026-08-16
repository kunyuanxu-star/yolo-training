"""Hyperparameter optimization CLI using Ultralytics' genetic algorithm.

Usage:
    python -m training_engine.tune --data configs/tennis_ball.yaml --epochs 30 --iterations 100
    python -m training_engine.tune --preset aggressive --data configs/tennis_ball.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .adapter import ModelAdapter
from .presets import merge_preset_with_overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for YOLO using genetic algorithm."
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained model.")
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Epochs per HPO iteration (keep low, default: 30).")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of HPO iterations (default: 100).")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size.")
    parser.add_argument("--device", default="", help="Device.")
    parser.add_argument("--project", default="runs/hpo", help="Output directory.")
    parser.add_argument("--name", default="tune", help="Run name.")
    parser.add_argument("--preset", default=None,
                        help="Use a preset config as base (conservative/aggressive/balanced).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    adapter = ModelAdapter(args.model)

    # Resolve imgsz / batch: use preset values if available, else defaults
    imgsz = args.imgsz or 640
    batch = args.batch or 16

    if args.preset:
        preset_cfg = merge_preset_with_overrides(args.preset, {
            "imgsz": args.imgsz,
            "batch": args.batch,
        })
        imgsz = preset_cfg.get("imgsz", imgsz)
        batch = preset_cfg.get("batch", batch)
        print(f"Using preset '{args.preset}': {preset_cfg.get('description', '')}")
        print(f"  imgsz={imgsz}, batch={batch}")

    print(f"Starting HPO: {args.iterations} iterations, {args.epochs} epochs each...")
    print(f"  Data: {data_path}")
    print(f"  Output: {args.project}/{args.name}")

    results = adapter.tune(
        data=str(data_path),
        epochs=args.epochs,
        iterations=args.iterations,
        imgsz=imgsz,
        batch=batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    print("\nHPO complete!")
    print(results)


if __name__ == "__main__":
    main()
