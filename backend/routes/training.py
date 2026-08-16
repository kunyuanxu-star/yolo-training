"""Training routes — file-based storage."""
import threading
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from backend.store import db
from backend.schemas.training import ModelConfigCreate, TrainingJobCreate
from backend.dependencies import get_current_user, resolve_project_dataset
from backend.services.training_service import create_training_job
from backend.services.storage_service import storage_service
from backend.services.yolo_export_service import generate_yolo_dataset

router = APIRouter(prefix="/api/v1", tags=["training"])

# Per-job cancellation events for background tune threads
_tune_cancel_events: dict[str, threading.Event] = {}

def _own_project(pid: str, user: dict) -> dict:
    p = db["projects"].get(pid)
    if not p or str(p.get("user_id")) != str(user.get("id")): raise HTTPException(404, detail="Project not found")
    return p

# Configs
@router.get("/projects/{project_id}/configs")
def list_configs(project_id: str, user: dict = Depends(get_current_user)):
    _own_project(project_id, user)
    return db["model_configs"].filter(lambda c: c["project_id"] == project_id)

@router.post("/projects/{project_id}/configs", status_code=201)
def create_config(project_id: str, data: ModelConfigCreate, user: dict = Depends(get_current_user)):
    _own_project(project_id, user)
    return db["model_configs"].create(dict(project_id=project_id, **data.model_dump()))

@router.delete("/configs/{config_id}", status_code=204)
def delete_config(config_id: str, user: dict = Depends(get_current_user)):
    cfg = db["model_configs"].get(config_id)
    if cfg: _own_project(cfg["project_id"], user)
    db["model_configs"].delete(config_id)

# Jobs
@router.post("/training/jobs", status_code=201)
def start_training(data: TrainingJobCreate, user: dict = Depends(get_current_user)):
    cfg = db["model_configs"].get(data.model_config_id)
    if not cfg: raise HTTPException(404, detail="Config not found")
    _own_project(cfg["project_id"], user)
    # Auto-resolve dataset_id from project if not provided
    dataset_id = data.dataset_id or resolve_project_dataset(cfg["project_id"])["id"]
    return create_training_job(user["id"], data.model_config_id, dataset_id, data.name, cfg["project_id"])

@router.get("/training/jobs")
def list_jobs(project_id: str = Query(""), status_filter: str = Query("", alias="status"), user: dict = Depends(get_current_user)):
    if project_id:
        _own_project(project_id, user)
        models_in_project = [m["id"] for m in db["trained_models"].filter(lambda m: m["project_id"] == project_id)]
        # Also include tune jobs (model_id="") for datasets in this project
        ds_ids = [d["id"] for d in db["datasets"].filter(lambda d: d["project_id"] == project_id)]
        jobs = [j for j in db["training_jobs"].all()
                if j["model_id"] in models_in_project or (j["model_id"] == "" and j.get("dataset_id") in ds_ids)]
    else:
        user_projects = [p["id"] for p in db["projects"].filter(lambda p: str(p.get("user_id")) == str(user.get("id")))]
        models_in_project = [m["id"] for m in db["trained_models"].all() if m["project_id"] in user_projects]
        user_ds_ids = [d["id"] for d in db["datasets"].all() if d["project_id"] in user_projects]
        jobs = [j for j in db["training_jobs"].all()
                if j["model_id"] in models_in_project or (j["model_id"] == "" and j.get("dataset_id") in user_ds_ids)]
    if status_filter: jobs = [j for j in jobs if j.get("status") == status_filter]
    return {"items": sorted(jobs, key=lambda j: j.get("created_at", ""), reverse=True), "total": len(jobs)}

@router.get("/training/jobs/{job_id}")
def get_job(job_id: str, user: dict = Depends(get_current_user)):
    return db["training_jobs"].get(job_id) or HTTPException(404, detail="Not found")

@router.post("/training/jobs/{job_id}/cancel")
def cancel_job(job_id: str, user: dict = Depends(get_current_user)):
    job = db["training_jobs"].get(job_id)
    if not job: raise HTTPException(404, detail="Not found")
    if job["status"] in ("queued", "running", "tuning"):
        db["training_jobs"].update(job_id, {"status": "cancelled"})
        # Signal background tune thread to stop, if any
        cancel_event = _tune_cancel_events.pop(job_id, None)
        if cancel_event:
            cancel_event.set()
    return {"status": "cancelled"}


# ── Training Presets ──

@router.get("/training/presets")
def list_presets(user: dict = Depends(get_current_user)):
    """Return available training presets for quick configuration."""
    from training_engine.presets import list_presets, ALL_PRESETS
    presets = list_presets()
    return {
        "presets": [
            {"name": name, "description": info["description"],
             "config": ALL_PRESETS.get(name, {})}
            for name, info in presets.items()
        ]
    }


# ── Hyperparameter Optimization (HPO) ──

class TuneRequest(BaseModel):
    dataset_id: str
    base_model: str = "yolov8n.pt"
    epochs: int = 30
    iterations: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = ""


@router.post("/training/tune")
def start_tune(data: TuneRequest, user: dict = Depends(get_current_user)):
    """Run hyperparameter optimization using genetic algorithm.

    This is a long-running operation. Returns immediately with a job-like
    tracking object. The tune runs in a background thread with cancellation support.
    """
    import uuid
    from datetime import datetime, timezone

    ds = db["datasets"].get(data.dataset_id)
    if not ds:
        raise HTTPException(404, detail="Dataset not found")
    _own_project(ds["project_id"], user)

    # Use a unique output directory per tune run to avoid races with
    # concurrent tune requests on the same dataset.
    job_id = str(uuid.uuid4())
    out = storage_service.exports_dir / f"tune_dataset_{job_id[:8]}"
    yaml_path = generate_yolo_dataset(ds["id"], out, {"train": 0.7, "val": 0.2, "test": 0.1})

    # Create a tracking record
    db["training_jobs"].create({
        "id": job_id,
        "model_id": "",                     # tune jobs have no trained model
        "config_id": "",
        "dataset_id": data.dataset_id,
        "project_id": ds["project_id"],
        "status": "tuning",
        "progress": 0,
        "total_epochs": data.epochs,
        "current_metric": {"type": "tune", "iterations": data.iterations},
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    # Create a cancellation event for this tune run
    cancel_event = threading.Event()
    _tune_cancel_events[job_id] = cancel_event

    def _run_tune():
        try:
            from training_engine.adapter import ModelAdapter

            # Check cancellation before starting expensive work
            if cancel_event.is_set():
                db["training_jobs"].update(job_id, {"status": "cancelled"})
                return

            adapter = ModelAdapter(data.base_model)
            results = adapter.tune(
                data=str(yaml_path),
                epochs=data.epochs,
                iterations=data.iterations,
                imgsz=data.imgsz,
                batch=data.batch,
                device=data.device,
                project=str(storage_service.models_dir / "hpo"),
                name=f"tune_{job_id[:8]}",
            )

            # Check if cancelled during tuning
            if cancel_event.is_set():
                db["training_jobs"].update(job_id, {"status": "cancelled"})
                return

            best_fitness = results.get("best_fitness")
            best_params = results.get("best_params", {})
            db["training_jobs"].update(job_id, {
                "status": "completed",
                "progress": 100,
                "current_metric": {
                    "type": "tune_complete",
                    "iterations": data.iterations,
                    "best_fitness": best_fitness,
                    "best_params": best_params,
                },
            })
        except Exception as e:
            db["training_jobs"].update(job_id, {
                "status": "failed",
                "error_message": str(e),
            })
        finally:
            _tune_cancel_events.pop(job_id, None)

    threading.Thread(target=_run_tune, daemon=True).start()
    return {"job_id": job_id, "status": "tuning", "message": "HPO started in background"}
