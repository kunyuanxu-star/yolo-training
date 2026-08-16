"""Dataset routes — file-based storage."""
import threading
from pathlib import Path
from fastapi import APIRouter, Depends, Header, HTTPException, status, Query, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from backend.store import db
from backend.schemas.dataset import DatasetCreate, AnnotationBulkUpdate, AnnotationCreate, LabelClassCreate
from backend.dependencies import get_current_user, resolve_project_dataset
from backend.services.dataset_service import upload_images, get_or_create_default_class, save_yolo_labels, load_yolo_labels, import_yolo_dataset
from backend.services.yolo_export_service import generate_yolo_dataset
from backend.services.storage_service import storage_service
from backend.services.augmentation_service import augment_dataset, AUGMENTATION_REGISTRY, RECOMMENDED_PRESETS

router = APIRouter(prefix="/api/v1", tags=["datasets"])

# Per-dataset cancellation events for background augmentation
_augment_jobs: dict[str, dict] = {}

def _own_project(pid: str, user: dict) -> dict:
    p = db["projects"].get(pid)
    if not p or str(p.get("user_id")) != str(user.get("id")):
        raise HTTPException(404, detail="Project not found")
    return p

def _own_ds(did: str, user: dict) -> dict:
    ds = db["datasets"].get(did)
    if not ds: raise HTTPException(404, detail="Dataset not found")
    p = db["projects"].get(ds["project_id"])
    if not p or str(p.get("user_id")) != str(user.get("id")): raise HTTPException(404, detail="Project not found")
    return ds

def _own_img(iid: str, user: dict) -> dict:
    img = db["images"].get(iid)
    if not img: raise HTTPException(404, detail="Image not found")
    _own_ds(img["dataset_id"], user)
    return img

def _resolve_user_for_img(user: str = "", x_user: str = "") -> dict:
    """Resolve user from query param or header for <img> tag requests."""
    username = user or x_user or "dev"
    return get_current_user(username)

# Dataset CRUD
@router.get("/projects/{project_id}/datasets")
def list_datasets(project_id: str, user: dict = Depends(get_current_user)):
    return db["datasets"].filter(lambda d: d["project_id"] == project_id)

@router.post("/projects/{project_id}/datasets", status_code=201)
def create_dataset(project_id: str, data: DatasetCreate, user: dict = Depends(get_current_user)):
    ds = db["datasets"].create({"project_id": project_id, "name": data.name, "description": data.description, "current_version": 1, "image_count": 0})
    get_or_create_default_class(ds["id"])
    return ds

@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, user: dict = Depends(get_current_user)):
    return _own_ds(dataset_id, user)

@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, user: dict = Depends(get_current_user)):
    ds = _own_ds(dataset_id, user)
    for img in db["images"].filter(lambda i: i["dataset_id"] == dataset_id):
        for ann in db["annotations"].filter(lambda a: a["image_id"] == img["id"]):
            db["annotations"].delete(ann["id"])
        db["images"].delete(img["id"])
    db["label_classes"].delete(dataset_id)
    db["datasets"].delete(dataset_id)

# Uploads
@router.post("/datasets/{dataset_id}/upload")
def upload(dataset_id: str, files: list[UploadFile] = File(...), user: dict = Depends(get_current_user)):
    _own_ds(dataset_id, user)
    return upload_images(dataset_id, files)

@router.get("/datasets/{dataset_id}/images")
def list_images(dataset_id: str, page: int = Query(1, ge=1), per_page: int = Query(50, ge=1, le=200), status_filter: str = Query("", alias="status"), user: dict = Depends(get_current_user)):
    _own_ds(dataset_id, user)
    imgs = db["images"].filter(lambda i: i["dataset_id"] == dataset_id)
    if status_filter: imgs = [i for i in imgs if i.get("status") == status_filter]
    total = len(imgs)
    start = (page - 1) * per_page
    result = []
    for i in imgs[start:start + per_page]:
        i = dict(i)
        i["thumbnail_url"] = f"/api/v1/images/{i['id']}/thumbnail"
        i["image_url"] = f"/api/v1/images/{i['id']}/file"
        result.append(i)
    return {"items": result, "total": total, "page": page, "per_page": per_page}

@router.get("/images/{image_id}/file")
def get_file(image_id: str, user: str = Query("", alias="user"), x_user: str = Header(default="", alias="X-User")):
    img = _own_img(image_id, _resolve_user_for_img(user, x_user))
    path = storage_service.backend._full_path(img["storage_path"])
    return FileResponse(path) if path.exists() else HTTPException(404, detail="File not found")

@router.get("/images/{image_id}/thumbnail")
def get_thumb(image_id: str, user: str = Query("", alias="user"), x_user: str = Header(default="", alias="X-User")):
    img = _own_img(image_id, _resolve_user_for_img(user, x_user))
    thumb = img.get("thumbnail_path")
    path = storage_service.backend._full_path(thumb) if thumb else None
    if path and path.exists():
        return FileResponse(path, media_type="image/jpeg")
    # Fallback to full image
    fp = storage_service.backend._full_path(img["storage_path"])
    if fp.exists():
        return FileResponse(fp, media_type="image/jpeg")
    raise HTTPException(404, detail="Not found")

@router.get("/images/{image_id}")
def get_image_detail(image_id: str, user: dict = Depends(get_current_user)):
    img = _own_img(image_id, user)
    anns = load_yolo_labels(image_id)
    return {"image": img, "annotations": anns}

@router.delete("/images/{image_id}", status_code=204)
def delete_image(image_id: str, user: dict = Depends(get_current_user)):
    img = _own_img(image_id, user)
    ds = db["datasets"].get(img["dataset_id"])
    if ds: db["datasets"].update(ds["id"], {"image_count": max(0, ds["image_count"] - 1)})

    # Delete annotations from DB
    for ann in db["annotations"].filter(lambda a: a["image_id"] == image_id):
        db["annotations"].delete(ann["id"])

    # Delete YOLO label file from disk
    from backend.services.dataset_service import _label_path_for_image
    label_path = _label_path_for_image(img)
    if label_path.exists():
        label_path.unlink()

    # Delete image file and thumbnail from disk
    import os
    from pathlib import Path
    from backend.services.storage_service import storage_service

    for key in ["storage_path", "thumbnail_path"]:
        fp = img.get(key)
        if fp:
            try:
                p = storage_service.backend._full_path(fp)
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    # Delete DB record
    db["images"].delete(image_id)

# Annotations
@router.get("/images/{image_id}/annotations")
def get_annotations(image_id: str, user: dict = Depends(get_current_user)):
    _own_img(image_id, user)
    return load_yolo_labels(image_id)

@router.put("/images/{image_id}/annotations")
def bulk_update_annotations(image_id: str, data: AnnotationBulkUpdate, user: dict = Depends(get_current_user)):
    img = _own_img(image_id, user)
    for ann in db["annotations"].filter(lambda a: a["image_id"] == image_id):
        db["annotations"].delete(ann["id"])
    new_anns = []
    for ad in data.annotations:
        a = db["annotations"].create({"image_id": image_id, "class_id": ad.class_id, "x_center": ad.x_center, "y_center": ad.y_center, "width": ad.width, "height": ad.height, "created_by": user["id"]})
        new_anns.append(a)
    db["images"].update(image_id, {"status": "annotated" if data.annotations else "uploaded"})
    # Also write YOLO-format .txt label file alongside the image
    save_yolo_labels(image_id)
    return {"annotations": new_anns}

@router.post("/images/{image_id}/annotations", status_code=201)
def create_annotation(image_id: str, data: AnnotationCreate, user: dict = Depends(get_current_user)):
    _own_img(image_id, user)
    a = db["annotations"].create({"image_id": image_id, "class_id": data.class_id, "x_center": data.x_center, "y_center": data.y_center, "width": data.width, "height": data.height, "created_by": user["id"]})
    save_yolo_labels(image_id)
    return a

@router.delete("/annotations/{annotation_id}", status_code=204)
def delete_annotation(annotation_id: str, user: dict = Depends(get_current_user)):
    ann = db["annotations"].get(annotation_id)
    if ann:
        img_id = ann["image_id"]
        _own_img(img_id, user)
        db["annotations"].delete(annotation_id)
        save_yolo_labels(img_id)

# Classes
@router.get("/datasets/{dataset_id}/classes")
def list_classes(dataset_id: str, user: dict = Depends(get_current_user)):
    _own_ds(dataset_id, user)
    return db["label_classes"].filter(lambda c: c["dataset_id"] == dataset_id)

@router.post("/datasets/{dataset_id}/classes", status_code=201)
def create_class(dataset_id: str, data: LabelClassCreate, user: dict = Depends(get_current_user)):
    _own_ds(dataset_id, user)
    existing = db["label_classes"].filter(lambda c: c["dataset_id"] == dataset_id)
    next_index = max([c["yolo_index"] for c in existing], default=-1) + 1
    return db["label_classes"].create({"dataset_id": dataset_id, "name": data.name, "yolo_index": next_index, "color": data.color})

@router.delete("/classes/{class_id}", status_code=204)
def delete_class(class_id: str, user: dict = Depends(get_current_user)):
    cls = db["label_classes"].get(class_id)
    if cls: _own_ds(cls["dataset_id"], user)
    # Check if class is in use by annotations
    ann_count = len(db["annotations"].filter(lambda a: a["class_id"] == class_id))
    if ann_count > 0:
        raise HTTPException(400, detail=f"该类别被 {ann_count} 个标注使用，请先删除标注后再删除类别")
    db["label_classes"].delete(class_id)

# Export
@router.post("/datasets/{dataset_id}/export/yolo")
def export_yolo(dataset_id: str, user: dict = Depends(get_current_user)):
    _own_ds(dataset_id, user)
    out = storage_service.exports_dir / f"dataset_{dataset_id}"
    yaml_path = generate_yolo_dataset(dataset_id, out, {"train": 0.7, "val": 0.2, "test": 0.1})
    zip_path = out.parent / f"{out.name}.zip"
    return FileResponse(zip_path, media_type="application/zip", filename=f"dataset_{dataset_id}.zip")


# ── Data Augmentation ──

class AugmentRequest(BaseModel):
    augmentation_names: list[str]
    multiplier: int = 3
    output_mode: str = "expand"


@router.get("/datasets/{dataset_id}/augmentations")
def list_augmentations(dataset_id: str, user: dict = Depends(get_current_user)):
    """List available augmentation types and recommended presets."""
    _own_ds(dataset_id, user)
    return {
        "augmentations": [
            {"key": k, **v} for k, v in AUGMENTATION_REGISTRY.items()
        ],
        "presets": RECOMMENDED_PRESETS,
    }


@router.post("/datasets/{dataset_id}/augment")
def augment_dataset_endpoint(
    dataset_id: str,
    data: AugmentRequest,
    user: dict = Depends(get_current_user),
):
    """Start offline augmentation in the background.

    Returns immediately with a job_id. Poll GET /datasets/{id}/augment/{job_id}
    for status and results.
    """
    import uuid
    _own_ds(dataset_id, user)

    if data.multiplier < 1 or data.multiplier > 10:
        raise HTTPException(400, detail="multiplier must be between 1 and 10")

    if not data.augmentation_names:
        raise HTTPException(400, detail="No augmentations specified")

    # Check no augmentation is already running for this dataset
    for jid, job in _augment_jobs.items():
        if job.get("dataset_id") == dataset_id and job.get("status") == "running":
            raise HTTPException(400, detail="An augmentation job is already running for this dataset")

    job_id = str(uuid.uuid4())
    cancel_event = threading.Event()
    _augment_jobs[job_id] = {
        "dataset_id": dataset_id,
        "status": "running",
        "cancel_event": cancel_event,
        "result": None,
    }

    def _run():
        try:
            result = augment_dataset(
                dataset_id=dataset_id,
                augmentation_names=data.augmentation_names,
                multiplier=data.multiplier,
                output_mode=data.output_mode,
                cancel_event=cancel_event,
            )
            _augment_jobs[job_id]["result"] = result
            _augment_jobs[job_id]["status"] = "completed"
        except Exception as e:
            _augment_jobs[job_id]["result"] = {"generated": 0, "errors": [str(e)], "total_images": 0}
            _augment_jobs[job_id]["status"] = "failed"

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "running", "message": "Augmentation started in background"}


@router.get("/datasets/{dataset_id}/augment/{job_id}")
def get_augment_status(
    dataset_id: str,
    job_id: str,
    user: dict = Depends(get_current_user),
):
    """Poll augmentation job status and get results when done."""
    _own_ds(dataset_id, user)
    job = _augment_jobs.get(job_id)
    if not job:
        raise HTTPException(404, detail="Augmentation job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "result": job["result"],
    }


@router.post("/datasets/{dataset_id}/augment/{job_id}/cancel")
def cancel_augment(
    dataset_id: str,
    job_id: str,
    user: dict = Depends(get_current_user),
):
    """Cancel a running augmentation job."""
    _own_ds(dataset_id, user)
    job = _augment_jobs.get(job_id)
    if not job:
        raise HTTPException(404, detail="Augmentation job not found")
    if job["status"] == "running":
        job["cancel_event"].set()
        job["status"] = "cancelled"
    return {"status": "cancelled"}


# ── Project-scoped endpoints (auto-resolve dataset from project) ──

@router.post("/projects/{project_id}/upload")
def project_upload(project_id: str, files: list[UploadFile] = File(...), user: dict = Depends(get_current_user)):
    _own_project(project_id, user)
    ds = resolve_project_dataset(project_id)
    return upload_images(ds["id"], files)


class ImportYoloRequest(BaseModel):
    folder_path: str


@router.post("/projects/{project_id}/import-yolo")
def project_import_yolo(project_id: str, data: ImportYoloRequest, user: dict = Depends(get_current_user)):
    """Import a YOLO-format dataset from a local folder path.

    The folder must contain data.yaml and train/images/ + train/labels/ directories.
    Valid and test splits are optional. Images and annotations are imported together.
    """
    _own_project(project_id, user)
    ds = resolve_project_dataset(project_id)

    if not data.folder_path.strip():
        raise HTTPException(400, detail="folder_path is required")

    result = import_yolo_dataset(ds["id"], data.folder_path.strip())

    if result["imported"] == 0 and result.get("errors"):
        raise HTTPException(400, detail="\n".join(result["errors"]))

    return result


@router.post("/projects/{project_id}/import-yolo-upload")
async def project_import_yolo_upload(
    project_id: str,
    files: list[UploadFile] = File(...),
    user: dict = Depends(get_current_user),
):
    """Import a YOLO-format dataset from uploaded files (folder picker).

    Each file's filename is the relative path within the YOLO dataset
    (e.g. 'train/images/img1.jpg', 'train/labels/img1.txt', 'data.yaml').
    Files are extracted to a temp directory and imported.
    """
    import tempfile
    import shutil

    _own_project(project_id, user)
    ds = resolve_project_dataset(project_id)

    if not files:
        raise HTTPException(400, detail="No files provided")

    # Extract files to a temp directory preserving relative paths.
    # webkitRelativePath includes the selected folder name as the root
    # (e.g. 'Tennis_Ball/train/images/img1.jpg'). Strip the common prefix.
    tmp_root = Path(tempfile.mkdtemp(prefix="yolo_import_"))
    try:
        paths = [(f.filename or "") for f in files]
        paths = [p for p in paths if p]
        if not paths:
            raise HTTPException(400, detail="No files with valid paths")

        # Strip common root prefix (the selected folder name)
        if len(paths) > 1:
            # Find common prefix of all paths
            common = paths[0].split("/")
            for p in paths[1:]:
                parts = p.split("/")
                new_common = []
                for a, b in zip(common, parts):
                    if a == b:
                        new_common.append(a)
                    else:
                        break
                common = new_common
            prefix = "/".join(common) + "/" if common else ""
        else:
            # Single file: keep only the basename part (strip folder prefix)
            parts = paths[0].rsplit("/", 2)
            prefix = parts[0] + "/" if len(parts) > 2 else ""

        for f, rel_path in zip(files, paths):
            stripped = rel_path[len(prefix):] if prefix and rel_path.startswith(prefix) else rel_path
            if not stripped:
                continue
            dest = tmp_root / stripped
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as out:
                content = await f.read()
                out.write(content)

        result = import_yolo_dataset(ds["id"], str(tmp_root))
        if result["imported"] == 0 and result.get("errors"):
            raise HTTPException(400, detail="\n".join(result["errors"]))
        return result
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


@router.post("/projects/{project_id}/capture-url")
def project_capture_url(project_id: str, url: str = Query(...), user: dict = Depends(get_current_user)):
    """Capture a single frame from an MJPEG/RTSP URL and save to the project dataset."""
    _own_project(project_id, user)
    ds = resolve_project_dataset(project_id)

    import cv2
    import uuid
    from backend.services.storage_service import storage_service

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise HTTPException(400, detail=f"Cannot open stream: {url}")

    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            raise HTTPException(400, detail="Failed to read frame from stream")
    finally:
        cap.release()

    # Save frame as JPEG
    import tempfile
    image_uuid = str(uuid.uuid4())
    rel_path = f"datasets/{ds['id']}/{image_uuid}.jpg"
    abs_path = storage_service.backend._full_path(rel_path)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(abs_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Generate thumbnail
    from PIL import Image
    thumb_rel = f"datasets/{ds['id']}/thumbnails/{image_uuid}_thumb.jpg"
    thumb_full = storage_service.backend._full_path(thumb_rel)
    thumb_full.parent.mkdir(parents=True, exist_ok=True)
    h, w = frame.shape[:2]
    with Image.open(abs_path) as img:
        img.thumbnail((256, 256), Image.LANCZOS)
        img.save(thumb_full, "JPEG", quality=80)

    file_size = abs_path.stat().st_size
    filename = f"capture_{image_uuid[:8]}.jpg"

    img_record = db["images"].create({
        "dataset_id": ds["id"],
        "filename": filename,
        "storage_path": rel_path,
        "thumbnail_path": thumb_rel,
        "width": w,
        "height": h,
        "file_size_bytes": file_size,
        "status": "uploaded",
    })

    ds["image_count"] = ds.get("image_count", 0) + 1
    db["datasets"].update(ds["id"], {"image_count": ds["image_count"]})

    return {"uploaded": 1, "errors": [], "image": img_record}


@router.get("/projects/{project_id}/images")
def project_list_images(project_id: str, page: int = Query(1, ge=1),
                         per_page: int = Query(50, ge=1, le=200),
                         status_filter: str = Query("", alias="status"),
                         user: dict = Depends(get_current_user)):
    _own_project(project_id, user)
    ds = resolve_project_dataset(project_id)
    imgs = db["images"].filter(lambda i: i["dataset_id"] == ds["id"])
    if status_filter: imgs = [i for i in imgs if i.get("status") == status_filter]
    total = len(imgs)
    start = (page - 1) * per_page
    result = []
    for i in imgs[start:start + per_page]:
        i = dict(i)
        i["thumbnail_url"] = f"/api/v1/images/{i['id']}/thumbnail"
        i["image_url"] = f"/api/v1/images/{i['id']}/file"
        result.append(i)
    return {"items": result, "total": total, "page": page, "per_page": per_page}


@router.get("/projects/{project_id}/classes")
def project_list_classes(project_id: str, user: dict = Depends(get_current_user)):
    _own_project(project_id, user)
    ds = resolve_project_dataset(project_id)
    classes = db["label_classes"].filter(lambda c: c["dataset_id"] == ds["id"])
    classes.sort(key=lambda c: c.get("yolo_index", 0))
    # Attach annotation count for each class
    result = []
    for c in classes:
        c = dict(c)
        c["annotation_count"] = len(db["annotations"].filter(lambda a: a["class_id"] == c["id"]))
        result.append(c)
    return result


@router.post("/projects/{project_id}/classes", status_code=201)
def project_create_class(project_id: str, data: LabelClassCreate, user: dict = Depends(get_current_user)):
    _own_project(project_id, user)
    ds = resolve_project_dataset(project_id)
    existing = db["label_classes"].filter(lambda c: c["dataset_id"] == ds["id"])
    next_index = max([c["yolo_index"] for c in existing], default=-1) + 1
    return db["label_classes"].create({
        "dataset_id": ds["id"], "name": data.name,
        "yolo_index": next_index, "color": data.color,
    })


@router.post("/projects/{project_id}/export/yolo")
def project_export_yolo(project_id: str, user: dict = Depends(get_current_user)):
    _own_project(project_id, user)
    ds = resolve_project_dataset(project_id)
    out = storage_service.exports_dir / f"dataset_{ds['id']}"
    generate_yolo_dataset(ds["id"], out, {"train": 0.7, "val": 0.2, "test": 0.1})
    zip_path = out.parent / f"{out.name}.zip"
    return FileResponse(zip_path, media_type="application/zip", filename=f"dataset_{ds['id']}.zip")
