"""WebSocket routes for real-time training progress and inference."""

import os
os.environ["ULTRALYTICS_NO_AUTO_UPDATE"] = "1"  # prevent pip install at runtime

import json, io, struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from ..store import db
from ..services.storage_service import storage_service

router = APIRouter(tags=["websocket"])

# Cached model adapters for WebSocket inference
_ws_adapter_cache: dict[str, "ModelAdapter"] = {}

# Bright, distinct palette for per-class colors
_PALETTE = [
    (0, 255, 0),     # green
    (255, 50, 50),   # red
    (0, 180, 255),   # blue
    (255, 200, 0),   # yellow
    (255, 0, 255),   # magenta
    (0, 255, 255),   # cyan
    (255, 130, 0),   # orange
    (180, 0, 255),   # purple
    (0, 255, 130),   # spring green
    (255, 80, 180),  # hot pink
]
_class_colors: dict[str, tuple[int, int, int]] = {}
_color_idx = 0

def _class_color(name: str) -> tuple[int, int, int]:
    global _color_idx
    if name not in _class_colors:
        _class_colors[name] = _PALETTE[_color_idx % len(_PALETTE)]
        _color_idx += 1
    return _class_colors[name]

def _get_ws_adapter(weights_path: str):
    if weights_path not in _ws_adapter_cache:
        from training_engine.adapter import ModelAdapter
        _ws_adapter_cache[weights_path] = ModelAdapter(weights_path)
    return _ws_adapter_cache[weights_path]


@router.websocket("/ws/training/{job_id}")
async def training_progress_ws(
    websocket: WebSocket,
    job_id: str,
):
    """WebSocket endpoint for real-time training progress updates."""
    await websocket.accept()
    import asyncio
    try:
        while True:
            job = db["training_jobs"].get(job_id)
            if not job:
                await websocket.send_json({"type": "error", "message": "Job not found"})
                break
            await websocket.send_json({
                "type": "progress",
                "status": job.get("status", ""),
                "progress": job.get("progress", 0),
                "current_epoch": job.get("current_epoch", 0),
                "current_metric": job.get("current_metric"),
            })
            if job.get("status") in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass


def _resolve_ws_model(model_id: str) -> dict | None:
    """Resolve model for WebSocket inference. Supports pretrained_ prefix."""
    m = db["trained_models"].get(model_id)
    if m:
        return m

    # Pretrained model: pretrained_yolov8n -> storage/models/pretrained/yolov8n.pt
    if model_id.startswith("pretrained_"):
        name = model_id[len("pretrained_"):]
        pt_path = storage_service.storage_root / "models" / "pretrained" / f"{name}.pt"
        if pt_path.exists():
            return {
                "id": model_id, "name": name,
                "weights_path": str(pt_path),
                "format_type": "pretrained",
            }

    return None


@router.websocket("/ws/inference/{model_id}")
async def inference_ws(websocket: WebSocket, model_id: str, conf: float = 0.25):
    """WebSocket real-time inference: receive JPEG frames, return annotated JPEG."""
    m = _resolve_ws_model(model_id)
    if not m:
        await websocket.close(code=4004, reason="Model not found")
        return
    # Prefer PT weights; fallback to ONNX
    weights = m.get("weights_path")
    if not weights or not Path(weights).exists():
        for key in ["int8_onnx_path", "fp16_onnx_path", "onnx_path"]:
            p = m.get(key)
            if p and Path(p).exists():
                weights = p
                break
    if not weights or not Path(weights).exists():
        await websocket.close(code=4004, reason="Weights not available")
        return

    adapter = _get_ws_adapter(weights)
    await websocket.accept()

    try:
        while True:
            frame_bytes = await websocket.receive_bytes()

            # Decode JPEG in-memory
            pil_img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            img_bgr = np.array(pil_img)[..., ::-1]  # RGB -> BGR for model

            try:
                results = adapter.predict(source=img_bgr, conf=conf, save=False)
                r = results[0] if len(results) > 0 else None

                detections = []
                if r is not None and hasattr(r, 'boxes') and r.boxes:
                    names = getattr(r, 'names', {})
                    boxes = r.boxes
                    for i in range(len(boxes.cls)):
                        cls_id = int(boxes.cls[i].item()) if hasattr(boxes.cls[i], 'item') else int(boxes.cls[i])
                        detections.append({
                            "class": names.get(cls_id, str(cls_id)),
                            "class_id": cls_id,
                            "confidence": round(float(boxes.conf[i]), 4),
                            "bbox": [round(float(x), 1) for x in boxes.xyxy[i].tolist()],
                        })

                # Fast PIL drawing — different color per class
                draw = ImageDraw.Draw(pil_img)
                W, H = pil_img.size
                if detections:
                    for d in detections:
                        color = _class_color(d["class"])
                        x1, y1, x2, y2 = [max(0, int(v)) for v in d["bbox"]]
                        x2 = min(x2, W - 1)
                        y2 = min(y2, H - 1)
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                        label = f"{d['class']} {d['confidence']:.2f}"
                        tw, th = len(label) * 6, 14
                        ty = y1 - th if y1 - th > 0 else y1 + 2
                        draw.rectangle([x1, ty, x1 + tw, ty + th], fill=color)
                        draw.text((x1 + 1, ty + 1), label, fill=(255, 255, 255))

                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=55)
                result_jpeg = buf.getvalue()

                meta = json.dumps({"detections": detections, "count": len(detections)}).encode()
                print(f"  WS frame: {W}x{H}, detections={len(detections)}, jpeg={len(result_jpeg)}B", end="\r")
                await websocket.send_bytes(struct.pack("!I", len(meta)) + meta + result_jpeg)
            except Exception as e:
                # Send error frame so frontend doesn't hang
                error_meta = json.dumps({"error": str(e), "detections": [], "count": 0}).encode()
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=55)
                await websocket.send_bytes(struct.pack("!I", len(error_meta)) + error_meta + buf.getvalue())

    except WebSocketDisconnect:
        pass
