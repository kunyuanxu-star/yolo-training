"""Dataset service — file-based."""
from pathlib import Path
from fastapi import UploadFile
from backend.store import db
from backend.services.storage_service import storage_service

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_image(file: UploadFile) -> str | None:
    if not file.filename: return "Missing filename"
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS: return f"Unsupported format: {ext}"
    return None


def upload_images(dataset_id: str, files: list[UploadFile]) -> dict:
    ds = db["datasets"].get(dataset_id)
    if not ds: return {"uploaded": 0, "errors": ["Dataset not found"]}
    uploaded, errors = 0, []
    for file in files:
        err = validate_image(file)
        if err: errors.append({"filename": file.filename, "error": err}); continue
        try:
            meta = storage_service.save_dataset_image(file, dataset_id)
            db["images"].create(dict(dataset_id=dataset_id, **meta, status="uploaded"))
            uploaded += 1
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})
    ds["image_count"] = ds.get("image_count", 0) + uploaded
    db["datasets"].update(dataset_id, {"image_count": ds["image_count"]})
    return {"uploaded": uploaded, "errors": errors}


def get_or_create_default_class(dataset_id: str) -> dict:
    classes = db["label_classes"].filter(lambda c: c["dataset_id"] == dataset_id)
    if not classes:
        return db["label_classes"].create({"dataset_id": dataset_id, "name": "object", "yolo_index": 0, "color": "#00FF00"})
    return classes[0]


# ── YOLO label file storage ──

def _label_path_for_image(image: dict) -> Path:
    """Get the .txt label path for an image from its storage_path."""
    sp = Path(image.get("storage_path", ""))
    ds_id = image.get("dataset_id", "")
    stem = sp.stem  # UUID without extension
    return storage_service.storage_root / "datasets" / ds_id / "labels" / f"{stem}.txt"


def save_yolo_labels(image_id: str) -> Path | None:
    """Write YOLO-format label .txt for an image. Returns path or None if empty."""
    image = db["images"].get(image_id)
    if not image: return None

    anns = db["annotations"].filter(lambda a: a["image_id"] == image_id)
    if not anns:
        # Remove label file if no annotations
        p = _label_path_for_image(image)
        if p.exists(): p.unlink()
        return None

    # Build class_id → yolo_index map
    ds_id = image.get("dataset_id", "")
    classes = {c["id"]: c.get("yolo_index", 0) for c in db["label_classes"].filter(lambda c: c["dataset_id"] == ds_id)}

    lines = []
    for a in anns:
        yolo_idx = classes.get(a["class_id"], 0)
        lines.append(f"{yolo_idx} {a['x_center']:.6f} {a['y_center']:.6f} {a['width']:.6f} {a['height']:.6f}")

    p = _label_path_for_image(image)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    return p


def import_yolo_dataset(dataset_id: str, folder_path: str) -> dict:
    """Import a YOLO-format dataset from a local folder.

    The folder must contain:
      data.yaml          — with 'names' and optionally 'nc'
      train/images/      — training images
      train/labels/      — YOLO .txt labels (same stem as image)
      valid/images/      — (optional) validation images
      valid/labels/      — (optional)
      test/images/       — (optional) test images
      test/labels/       — (optional)

    Returns: dict with 'imported', 'errors', 'classes_created'
    """
    import os
    import shutil
    import uuid
    from PIL import Image

    root = Path(folder_path)
    if not root.is_dir():
        return {"imported": 0, "errors": [f"Folder not found: {folder_path}"], "classes_created": 0}

    # 1. Parse data.yaml
    yaml_path = root / "data.yaml"
    yaml_names: list[str] = []
    if yaml_path.is_file():
        import yaml as _yaml
        try:
            with open(yaml_path) as f:
                cfg = _yaml.safe_load(f) or {}
            yaml_names = cfg.get("names", [])
            # Normalize names: ultralytics sometimes uses {0: 'class0', 1: 'class1'} format
            if isinstance(yaml_names, dict):
                yaml_names = [yaml_names[i] for i in sorted(yaml_names.keys())]
        except Exception as e:
            return {"imported": 0, "errors": [f"Failed to parse data.yaml: {e}"], "classes_created": 0}
    else:
        return {"imported": 0, "errors": ["data.yaml not found in folder"], "classes_created": 0}

    # 2. Create label classes from yaml names
    existing_classes = db["label_classes"].filter(lambda c: c["dataset_id"] == dataset_id)
    name_to_cls: dict[str, dict] = {c["name"]: c for c in existing_classes}
    next_yolo_idx = max([c.get("yolo_index", 0) for c in existing_classes], default=0)
    classes_created = 0

    for name in yaml_names:
        if name not in name_to_cls:
            cls = db["label_classes"].create({
                "dataset_id": dataset_id,
                "name": str(name),
                "yolo_index": next_yolo_idx,
                "color": "#00FF00",
            })
            name_to_cls[name] = cls
            next_yolo_idx += 1
            classes_created += 1

    # 3. Collect image/label pairs from train, valid, test
    imported = 0
    errors: list[str] = []

    for split in ("train", "valid", "test"):
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        if not images_dir.is_dir():
            continue

        for img_file in sorted(images_dir.iterdir()):
            if img_file.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue

            # Find matching label file
            label_file = labels_dir / (img_file.stem + ".txt")
            label_annotations: list[tuple[int, float, float, float, float]] = []
            if label_file.is_file():
                try:
                    for line in label_file.read_text().strip().splitlines():
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        yolo_idx = int(parts[0])
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        label_annotations.append((yolo_idx, xc, yc, w, h))
                except Exception as e:
                    errors.append(f"Failed to parse {label_file}: {e}")
                    continue

            # Copy image to storage
            image_uuid = str(uuid.uuid4())
            ext = img_file.suffix.lower()
            rel_path = f"datasets/{dataset_id}/{image_uuid}{ext}"
            abs_path = storage_service.backend._full_path(rel_path)
            abs_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copy2(str(img_file), str(abs_path))
            except Exception as e:
                errors.append(f"Failed to copy {img_file.name}: {e}")
                continue

            # Get image dimensions
            try:
                with Image.open(abs_path) as pil:
                    w, h = pil.size
            except Exception:
                w, h = 0, 0

            # Generate thumbnail
            thumb_rel_path = f"datasets/{dataset_id}/thumbnails/{image_uuid}_thumb.jpg"
            thumb_full = storage_service.backend._full_path(thumb_rel_path)
            thumb_full.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(abs_path) as pil:
                    pil.thumbnail((256, 256), Image.LANCZOS)
                    pil.save(thumb_full, "JPEG", quality=80)
            except Exception:
                thumb_rel_path = ""  # no thumbnail

            # Create image DB record
            file_size = abs_path.stat().st_size
            img_record = db["images"].create({
                "dataset_id": dataset_id,
                "filename": img_file.name,
                "storage_path": rel_path,
                "thumbnail_path": thumb_rel_path,
                "width": w,
                "height": h,
                "file_size_bytes": file_size,
                "status": "annotated" if label_annotations else "uploaded",
            })

            # Create annotation records
            for yolo_idx, xc, yc, bw, bh in label_annotations:
                # Map yolo_index back to class by name
                if yolo_idx < len(yaml_names):
                    cls_name = yaml_names[yolo_idx]
                    cls = name_to_cls.get(str(cls_name))
                else:
                    # Try to find by yolo_index directly
                    cls = None
                    for c in name_to_cls.values():
                        if c.get("yolo_index") == yolo_idx:
                            cls = c
                            break
                if cls:
                    db["annotations"].create({
                        "image_id": img_record["id"],
                        "class_id": cls["id"],
                        "x_center": round(xc, 6),
                        "y_center": round(yc, 6),
                        "width": round(bw, 6),
                        "height": round(bh, 6),
                    })

            # Write YOLO label file
            if label_annotations:
                lbl_dir = storage_service.storage_root / "datasets" / dataset_id / "labels"
                lbl_dir.mkdir(parents=True, exist_ok=True)
                lines_out = []
                for yolo_idx, xc, yc, bw, bh in label_annotations:
                    lines_out.append(f"{yolo_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                (lbl_dir / f"{image_uuid}.txt").write_text("\n".join(lines_out) + "\n")

            imported += 1

    # Update dataset image count
    ds = db["datasets"].get(dataset_id)
    if ds:
        all_imgs = len(db["images"].filter(lambda i: i["dataset_id"] == dataset_id))
        db["datasets"].update(dataset_id, {"image_count": all_imgs})

    return {"imported": imported, "errors": errors, "classes_created": classes_created}


def load_yolo_labels(image_id: str) -> list[dict]:
    """Read annotations from YOLO .txt file. Falls back to JSON store."""
    image = db["images"].get(image_id)
    if not image: return []

    p = _label_path_for_image(image)
    if not p.exists():
        # Fallback: read from JSON store
        return db["annotations"].filter(lambda a: a["image_id"] == image_id)

    ds_id = image.get("dataset_id", "")
    classes_by_idx: dict[int, dict] = {}
    for c in db["label_classes"].filter(lambda c: c["dataset_id"] == ds_id):
        classes_by_idx[c.get("yolo_index", 0)] = c

    anns = []
    for line in p.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5: continue
        try:
            yolo_idx = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue
        cls = classes_by_idx.get(yolo_idx, {})
        anns.append({
            "class_id": cls.get("id", ""),
            "class_name": cls.get("name", ""),
            "x_center": xc, "y_center": yc, "width": w, "height": h,
        })
    return anns
