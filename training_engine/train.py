from __future__ import annotations

import argparse
from pathlib import Path

from .adapter import ModelAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tennis detector with YOLO26.")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained model or config path.")
    parser.add_argument(
        "--data",
        default="configs/tennis_ball.yaml",
        help="Dataset yaml path.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="", help="Device string, e.g. cpu, 0, 0,1.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--project", default="runs/tennis", help="Output root directory.")
    parser.add_argument("--name", default="train", help="Run name.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip training and only run validation.",
    )
    # --- Small-dataset / fine-tuning hyperparameters ---
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate (default: ultralytics default).")
    parser.add_argument("--lrf", type=float, default=None, help="Final learning rate factor (lr0 * lrf).")
    parser.add_argument("--momentum", type=float, default=None, help="SGD momentum.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay.")
    parser.add_argument("--warmup-epochs", type=float, default=None, help="Warmup epochs.")
    parser.add_argument("--cos-lr", action="store_true", default=None, help="Use cosine LR scheduler.")
    parser.add_argument("--optimizer", default=None, help="Optimizer: auto, SGD, Adam, AdamW.")
    parser.add_argument("--freeze", type=int, default=None, help="Freeze first N layers.")
    parser.add_argument("--close-mosaic", type=int, default=None, help="Epoch to close mosaic augmentation (0=never).")
    parser.add_argument("--no-augment", action="store_true", default=None, help="Disable data augmentation.")
    parser.add_argument("--single-cls", action="store_true", default=None, help="Treat all classes as a single class.")
    parser.add_argument("--extra-args", default=None, help="JSON string of extra kwargs to pass to model.train().")
    parser.add_argument("--preset", default=None,
                        help="Use a training preset (conservative/aggressive/balanced). Overrides defaults but not explicit args.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    adapter = ModelAdapter(args.model)

    if args.validate_only:
        results = adapter.validate(
            data=str(data_path),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
        print(results)
        return

    # Parse --extra-args JSON if provided
    extra_args = None
    if args.extra_args:
        import json
        extra_args = json.loads(args.extra_args)

    # Apply preset if specified (preset values fill in defaults, explicit CLI args override)
    lr0 = args.lr0
    lrf = args.lrf
    momentum = args.momentum
    weight_decay = args.weight_decay
    warmup_epochs = args.warmup_epochs
    cos_lr = args.cos_lr
    optimizer = args.optimizer
    freeze = args.freeze
    close_mosaic = args.close_mosaic
    augment = False if args.no_augment else None
    single_cls = args.single_cls if args.single_cls is not None else False
    epochs = args.epochs
    imgsz = args.imgsz
    batch = args.batch

    if args.preset:
        from .presets import merge_preset_with_overrides
        preset_cfg = merge_preset_with_overrides(args.preset, {
            "lr0": lr0, "lrf": lrf, "momentum": momentum,
            "weight_decay": weight_decay, "warmup_epochs": warmup_epochs,
            "cos_lr": cos_lr, "optimizer": optimizer,
            "freeze": freeze, "close_mosaic": close_mosaic,
            "augment": augment, "single_cls": single_cls,
            "epochs": epochs, "imgsz": imgsz, "batch": batch,
        })
        lr0 = preset_cfg.get("lr0")
        lrf = preset_cfg.get("lrf")
        momentum = preset_cfg.get("momentum")
        weight_decay = preset_cfg.get("weight_decay")
        warmup_epochs = preset_cfg.get("warmup_epochs")
        cos_lr = preset_cfg.get("cos_lr")
        optimizer = preset_cfg.get("optimizer")
        freeze = preset_cfg.get("freeze")
        close_mosaic = preset_cfg.get("close_mosaic")
        augment = preset_cfg.get("augment")
        single_cls = preset_cfg.get("single_cls", single_cls)
        epochs = preset_cfg.get("epochs", epochs)
        imgsz = preset_cfg.get("imgsz", imgsz)
        batch = preset_cfg.get("batch", batch)
        print(f"Using preset '{args.preset}': {preset_cfg.get('description', '')}")

    results = adapter.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        single_cls=single_cls,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        cos_lr=cos_lr,
        optimizer=optimizer,
        freeze=freeze,
        close_mosaic=close_mosaic,
        augment=augment,
        extra_args=extra_args,
    )
    print(results)


if __name__ == "__main__":
    main()
