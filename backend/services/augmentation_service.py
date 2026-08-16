"""Offline dataset augmentation service — Roboflow-style preprocessing.

Uses Albumentations to apply image-level transforms while automatically
adjusting bounding boxes for geometric changes.
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
from PIL import Image

from backend.store import db
from backend.services.storage_service import storage_service
from backend.config import settings

THUMBNAIL_SIZE = getattr(settings, "THUMBNAIL_SIZE", 256)

# ── Augmentation presets ────────────────────────────────────────────────────

AUGMENTATION_REGISTRY: dict[str, dict[str, Any]] = {
    "horizontal_flip": {
        "label": "水平翻转",
        "description": "左右镜像翻转图片，适合大多数场景",
        "category": "几何",
    },
    "vertical_flip": {
        "label": "垂直翻转",
        "description": "上下镜像翻转，适合航拍/显微场景",
        "category": "几何",
    },
    "rotate": {
        "label": "旋转 ±15°",
        "description": "小幅随机旋转，模拟拍摄角度变化",
        "category": "几何",
    },
    "brightness_contrast": {
        "label": "亮度/对比度",
        "description": "随机调整亮度和对比度，模拟光照变化",
        "category": "像素",
    },
    "hue_saturation": {
        "label": "色调/饱和度",
        "description": "随机调整色调、饱和度和明度",
        "category": "像素",
    },
    "gaussian_blur": {
        "label": "高斯模糊",
        "description": "模拟对焦不准或运动模糊",
        "category": "像素",
    },
    "gauss_noise": {
        "label": "高斯噪声",
        "description": "添加随机噪声，模拟低光照/传感器噪声",
        "category": "像素",
    },
    "cutout": {
        "label": "随机遮挡 (Cutout)",
        "description": "随机遮挡部分区域，提高对遮挡的鲁棒性",
        "category": "遮挡",
    },
}

# ── Recommended presets for small datasets ──────────────────────────────────

RECOMMENDED_PRESETS = {
    "basic": ["horizontal_flip", "brightness_contrast"],
    "moderate": ["horizontal_flip", "rotate", "brightness_contrast", "gaussian_blur"],
    "aggressive": ["horizontal_flip", "rotate", "brightness_contrast",
                   "hue_saturation", "gaussian_blur", "gauss_noise", "cutout"],
}


def _build_pipeline(augmentation_names: list[str]) -> A.Compose:
    """Build an Albumentations Compose pipeline from named augmentations."""
    transforms: list[A.BasicTransform] = []

    for name in augmentation_names:
        if name == "horizontal_flip":
            transforms.append(A.HorizontalFlip(p=0.5))
        elif name == "vertical_flip":
            transforms.append(A.VerticalFlip(p=0.3))
        elif name == "rotate":
            transforms.append(A.Rotate(limit=15, border_mode=0, p=0.7))
        elif name == "brightness_contrast":
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.7))
        elif name == "hue_saturation":
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.5))
        elif name == "gaussian_blur":
            transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=0.5))
        elif name == "gauss_noise":
            transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.5))
        elif name == "cutout":
            transforms.append(A.CoarseDropout(
                max_holes=4, max_height=32, max_width=32,
                min_holes=1, min_height=8, min_width=8,
                fill_value="random", p=0.5))

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",  # normalized [cx, cy, w, h]
            label_fields=["class_ids"],
            min_visibility=0.3,  # drop bboxes that are <30% visible after transform
        ),
    )


def _get_annotated_images(dataset_id: str) -> list[dict]:
    """Return all annotated images in a dataset."""
    return [
        i for i in db["images"].filter(lambda i: i["dataset_id"] == dataset_id)
        if i.get("status") == "annotated"
    ]


def _get_annotations_for_image(image_id: str) -> list[dict]:
    """Return annotations for an image."""
    return db["annotations"].filter(lambda a: a["image_id"] == image_id)


def _get_yolo_index_for_class(class_id: str, dataset_id: str) -> int:
    """Look up the yolo_index for a class."""
    classes = db["label_classes"].filter(lambda c: c["dataset_id"] == dataset_id)
    for c in classes:
        if c["id"] == class_id:
            return c.get("yolo_index", 0)
    return 0


def augment_dataset(
    dataset_id: str,
    augmentation_names: list[str],
    multiplier: int = 3,
    output_mode: str = "expand",
    cancel_event: Any = None,
) -> dict:
    """Apply offline augmentations to a dataset, generating new training samples.

    Args:
        dataset_id: Target dataset ID.
        augmentation_names: List of augmentation keys to apply.
        multiplier: Number of augmented variants per source image.
        output_mode: 'expand' (keep originals + add augmented) or
                     'replace' (delete originals, keep only augmented).
        cancel_event: Optional threading.Event to signal cancellation.

    Returns:
        dict with 'generated' count, 'errors' list, and 'total_images'.
    """
    if not augmentation_names:
        return {"generated": 0, "errors": ["No augmentations specified"], "total_images": 0}

    # Validate augmentation names
    invalid = [n for n in augmentation_names if n not in AUGMENTATION_REGISTRY]
    if invalid:
        return {"generated": 0, "errors": [f"Unknown augmentations: {invalid}"], "total_images": 0}

    pipeline = _build_pipeline(augmentation_names)
    images = _get_annotated_images(dataset_id)

    if not images:
        return {"generated": 0, "errors": ["No annotated images found"], "total_images": 0}

    # Build class_id → yolo_index map for creating annotations
    classes = db["label_classes"].filter(lambda c: c["dataset_id"] == dataset_id)
    idx_to_class: dict[int, dict] = {}
    for c in classes:
        idx_to_class[c.get("yolo_index", 0)] = c

    generated = 0
    errors: list[str] = []
    source_ids_to_delete: list[str] = []  # for 'replace' mode

    for img_record in images:
        # Check cancellation
        if cancel_event is not None and cancel_event.is_set():
            errors.append("Augmentation cancelled by user")
            break

        # Read source image
        src_path = storage_service.backend._full_path(img_record["storage_path"])
        try:
            pil_img = Image.open(src_path).convert("RGB")
        except Exception as e:
            errors.append(f"Cannot open {img_record['filename']}: {e}")
            continue

        img_array = np.array(pil_img)
        orig_h, orig_w = img_array.shape[:2]

        # Gather annotations
        anns = _get_annotations_for_image(img_record["id"])
        if not anns:
            continue  # skip images without annotations

        bboxes = []
        class_ids = []
        for a in anns:
            yolo_idx = _get_yolo_index_for_class(a["class_id"], dataset_id)
            bboxes.append([a["x_center"], a["y_center"], a["width"], a["height"]])
            class_ids.append(yolo_idx)

        # Generate N augmented variants
        for _ in range(multiplier):
            if cancel_event is not None and cancel_event.is_set():
                break

            # ── Transform (with error handling) ──
            try:
                transformed = pipeline(
                    image=img_array,
                    bboxes=bboxes,
                    class_ids=class_ids,
                )
            except Exception as e:
                errors.append(f"Transform failed for {img_record['filename']}: {e}")
                continue

            new_img_array = transformed["image"]
            new_bboxes = transformed["bboxes"]
            new_class_ids = transformed["class_ids"]

            # Skip if all bboxes were removed by the transform
            if len(new_bboxes) == 0:
                continue

            # ── Save augmented image & create DB records (with error handling) ──
            try:
                new_img = Image.fromarray(new_img_array)
            except Exception:
                # Handle edge case: array is not uint8
                new_img = Image.fromarray(new_img_array.astype(np.uint8))

            image_uuid = str(uuid.uuid4())
            ext = Path(img_record["filename"]).suffix.lower() or ".jpg"
            rel_path = f"datasets/{dataset_id}/{image_uuid}{ext}"
            abs_path = storage_service.backend._full_path(rel_path)

            try:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                new_img.save(abs_path, quality=90)
            except Exception as e:
                errors.append(f"Failed to save image for {img_record['filename']}: {e}")
                continue

            # Generate thumbnail (project-standard size)
            thumb_rel_path = f"datasets/{dataset_id}/thumbnails/{image_uuid}_thumb.jpg"
            thumb_full = storage_service.backend._full_path(thumb_rel_path)
            try:
                thumb_full.parent.mkdir(parents=True, exist_ok=True)
                thumb_img = new_img.copy()
                thumb_img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.LANCZOS)
                thumb_img.save(thumb_full, "JPEG", quality=80)
            except Exception as e:
                errors.append(f"Failed to save thumbnail for {img_record['filename']}: {e}")
                # Clean up the full image if thumbnail failed
                try:
                    if abs_path.exists():
                        abs_path.unlink()
                except Exception:
                    pass
                continue

            new_h, new_w = new_img_array.shape[:2]

            # Create Image DB record
            try:
                new_image_record = db["images"].create({
                    "dataset_id": dataset_id,
                    "filename": f"aug_{image_uuid}{ext}",
                    "storage_path": rel_path,
                    "thumbnail_path": thumb_rel_path,
                    "width": new_w,
                    "height": new_h,
                    "file_size_bytes": abs_path.stat().st_size,
                    "status": "annotated",
                })
            except Exception as e:
                errors.append(f"Failed to create image record for {img_record['filename']}: {e}")
                # Clean up files
                for p in (abs_path, thumb_full):
                    try:
                        if p.exists():
                            p.unlink()
                    except Exception:
                        pass
                continue

            # Create Annotation DB records + YOLO .txt label file
            try:
                lines = []
                for bbox, cls_idx in zip(new_bboxes, new_class_ids):
                    cls_info = idx_to_class.get(cls_idx, {})
                    db["annotations"].create({
                        "image_id": new_image_record["id"],
                        "class_id": cls_info.get("id", ""),
                        "x_center": round(float(bbox[0]), 6),
                        "y_center": round(float(bbox[1]), 6),
                        "width": round(float(bbox[2]), 6),
                        "height": round(float(bbox[3]), 6),
                    })
                    lines.append(f"{cls_idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")

                # Write YOLO .txt file
                lbl_dir = storage_service.storage_root / "datasets" / dataset_id / "labels"
                lbl_dir.mkdir(parents=True, exist_ok=True)
                (lbl_dir / f"{image_uuid}.txt").write_text("\n".join(lines) + "\n")
            except Exception as e:
                errors.append(f"Failed to create annotations for augmented image of {img_record['filename']}: {e}")
                # Clean up: remove the image DB record and files
                try:
                    db["images"].delete(new_image_record["id"])
                except Exception:
                    pass
                for p in (abs_path, thumb_full):
                    try:
                        if p.exists():
                            p.unlink()
                    except Exception:
                        pass
                continue

            generated += 1

        # Mark source for deletion in 'replace' mode
        if output_mode == "replace":
            source_ids_to_delete.append(img_record["id"])

    # ── Handle 'replace' mode: delete original images ──
    if output_mode == "replace" and source_ids_to_delete:
        deleted_count = 0
        for img_id in source_ids_to_delete:
            try:
                img = db["images"].get(img_id)
                if not img:
                    continue
                # Remove annotation DB records
                for ann in db["annotations"].filter(lambda a: a["image_id"] == img_id):
                    db["annotations"].delete(ann["id"])
                # Remove image files from disk
                for key in ("storage_path", "thumbnail_path"):
                    fp = img.get(key)
                    if fp:
                        try:
                            p = storage_service.backend._full_path(fp)
                            if p.exists():
                                p.unlink()
                        except Exception:
                            pass
                # Remove YOLO label file
                lbl_dir = storage_service.storage_root / "datasets" / dataset_id / "labels"
                lbl_name = Path(img["filename"]).stem + ".txt"
                lbl_path = lbl_dir / lbl_name
                try:
                    if lbl_path.exists():
                        lbl_path.unlink()
                except Exception:
                    pass
                # Delete DB record
                db["images"].delete(img_id)
                deleted_count += 1
            except Exception as e:
                errors.append(f"Failed to delete source image {img_id}: {e}")

    # Update dataset image count
    ds = db["datasets"].get(dataset_id)
    if ds:
        current_count = len(_get_annotated_images(dataset_id))
        db["datasets"].update(dataset_id, {"image_count": current_count})

    return {
        "generated": generated,
        "errors": errors,
        "total_images": len(_get_annotated_images(dataset_id)),
    }
