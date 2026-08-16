"""Training presets for small datasets — Roboflow-style quick configs.

Each preset is a set of hyperparameters optimized for a specific scenario.
Use with: python -m training_engine.train --preset aggressive
"""

from __future__ import annotations

# ── Small-dataset detection presets ─────────────────────────────────────────
# These are designed for datasets with 10–200 annotated images.

SMALL_DATASET_PRESETS: dict[str, dict] = {
    "conservative": {
        "description": "快速基线：冻结 backbone，轻度增强，适合快速验证",
        "epochs": 100,
        "imgsz": 320,
        "batch": 4,
        "lr0": 0.001,
        "lrf": 0.01,
        "cos_lr": True,
        "freeze": 10,
        "close_mosaic": 10,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "single_cls": False,
    },
    "aggressive": {
        "description": "强增强：全程 mosaic，极低学习率，适合极少数据（10-50张）",
        "epochs": 200,
        "imgsz": 320,
        "batch": 4,
        "lr0": 0.0005,
        "lrf": 0.01,
        "cos_lr": True,
        "freeze": 10,
        "close_mosaic": 0,
        "weight_decay": 0.001,
        "warmup_epochs": 5,
        "single_cls": False,
    },
    "balanced": {
        "description": "均衡配置：中等学习率和增强，适合 50-200 张数据集",
        "epochs": 150,
        "imgsz": 416,
        "batch": 4,
        "lr0": 0.001,
        "lrf": 0.01,
        "cos_lr": True,
        "freeze": 5,
        "close_mosaic": 5,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "single_cls": False,
    },
}

# ── Full-dataset presets (200+ images) ──────────────────────────────────────

STANDARD_PRESETS: dict[str, dict] = {
    "default": {
        "description": "Ultralytics 默认配置",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "lr0": 0.01,
        "lrf": 0.01,
        "cos_lr": False,
        "freeze": None,
        "close_mosaic": 10,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
    },
    "high_accuracy": {
        "description": "高精度：更大分辨率，更长训练",
        "epochs": 300,
        "imgsz": 640,
        "batch": 8,
        "lr0": 0.01,
        "lrf": 0.01,
        "cos_lr": True,
        "freeze": None,
        "close_mosaic": 15,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
    },
}

# ── Merge all presets ───────────────────────────────────────────────────────

ALL_PRESETS = {**SMALL_DATASET_PRESETS, **STANDARD_PRESETS}


def get_preset(name: str) -> dict | None:
    """Look up a preset by name. Returns None if not found."""
    return ALL_PRESETS.get(name)


def list_presets() -> dict[str, dict]:
    """Return all available presets."""
    return {
        name: {"description": cfg["description"]}
        for name, cfg in ALL_PRESETS.items()
    }


def merge_preset_with_overrides(preset_name: str, overrides: dict) -> dict:
    """Load a preset and apply user overrides on top.

    Args:
        preset_name: Name of the preset to load.
        overrides: Dict of parameter overrides (only non-None values are applied).

    Returns:
        Merged config dict.

    Raises:
        ValueError: If preset_name is not found.
    """
    preset = get_preset(preset_name)
    if preset is None:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(ALL_PRESETS.keys())}")

    merged = dict(preset)  # copy
    for k, v in overrides.items():
        if v is not None:
            merged[k] = v
    return merged
