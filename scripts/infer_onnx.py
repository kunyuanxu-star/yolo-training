"""Run inference with the exported ONNX models using onnxruntime only.

This lets you run detection without PyTorch/Ultralytics installed.

Examples:
    python scripts/infer_onnx.py --class orange --source orange.jpg --save
    python scripts/infer_onnx.py --class apple --source 0 --camera
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


CLASSES = [
    "apple", "banana", "paper_cup", "orange", "doll", "plastic_bottle",
    "book", "chair", "cup", "computer_keyboard", "laptop",
]
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def letterbox(img: np.ndarray, new_shape: int = INPUT_SIZE):
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    pad_h, pad_w = new_shape - nh, new_shape - nw
    top, left = pad_h // 2, pad_w // 2
    padded = cv2.copyMakeBorder(
        resized,
        top,
        pad_h - top,
        left,
        pad_w - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, scale, (left, top)


def postprocess(
    output: np.ndarray,
    names: dict[int, str],
    conf: float,
    iou: float,
    orig_shape: tuple[int, int],
    scale: float,
    pad: tuple[int, int],
) -> list[tuple[int, str, float, tuple[float, float, float, float]]]:
    if output.ndim == 3:
        if output.shape[1] < output.shape[2]:
            output = output[0].T
    if output.shape[1] < 6:
        # YOLOv8 single-class export: [cx, cy, w, h, class0_score]
        scores = output[:, 4]
        cls_ids = np.zeros(len(output), dtype=int)
        final_scores = scores
    else:
        scores = output[:, 4]
        cls_ids = np.argmax(output[:, 5:], axis=1)
        cls_scores = output[:, 5:][np.arange(len(output)), cls_ids]
        final_scores = scores * cls_scores
    mask = final_scores > conf
    filtered = output[mask]
    if len(filtered) == 0:
        return []
    cx, cy, w, h = (
        filtered[:, 0],
        filtered[:, 1],
        filtered[:, 2],
        filtered[:, 3],
    )
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    confs = final_scores[mask]
    cids = cls_ids[mask]
    indices = cv2.dnn.NMSBoxes(
        bboxes=[[float(x1[i]), float(y1[i]), float(w[i]), float(h[i])] for i in range(len(x1))],
        scores=[float(c) for c in confs],
        score_threshold=conf,
        nms_threshold=iou,
    )
    left, top = pad
    results = []
    for i in indices:
        i = int(i)
        rx1 = (x1[i] - left) / scale
        ry1 = (y1[i] - top) / scale
        rx2 = (x2[i] - left) / scale
        ry2 = (y2[i] - top) / scale
        results.append(
            (int(cids[i]), names.get(int(cids[i]), "object"), float(confs[i]),
             (float(rx1), float(ry1), float(rx2), float(ry2)))
        )
    return results


def draw_boxes(img: np.ndarray, boxes: list, color: tuple[int, int, int] = (0, 255, 0)):
    out = img.copy()
    for _, label, conf, (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(
            out, text, (int(x1), int(y1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
    return out


def infer_image(session: ort.InferenceSession, names: dict[int, str], img: np.ndarray):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    padded, scale, pad = letterbox(rgb, INPUT_SIZE)
    blob = padded.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis]
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    return postprocess(
        outputs[0], names, CONF_THRESHOLD, IOU_THRESHOLD,
        img.shape[:2], scale, pad,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX inference for six class models")
    parser.add_argument("--class", dest="cls", required=True, choices=CLASSES)
    parser.add_argument("--source", required=True, help="Image path or camera id (with --camera).")
    parser.add_argument("--camera", action="store_true", help="Treat --source as camera id.")
    parser.add_argument("--save", action="store_true", help="Save annotated output.")
    args = parser.parse_args()

    model_path = MODELS_DIR / f"best_{args.cls}.onnx"
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    session = ort.InferenceSession(str(model_path))
    names = {0: args.cls}

    if args.camera:
        cap = cv2.VideoCapture(int(args.source))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera {args.source}")
        print("Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            boxes = infer_image(session, names, frame)
            vis = draw_boxes(frame, boxes)
            cv2.imshow(f"ONNX {args.cls}", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    src = Path(args.source)
    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit(f"Cannot read image: {src}")
    boxes = infer_image(session, names, img)
    for _, label, conf, box in boxes:
        print(f"{label} conf={conf:.3f} xyxy={[round(float(v), 1) for v in box]}")
    if args.save:
        out_dir = ROOT / "runs" / "infer_onnx"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{src.stem}_{args.cls}.jpg"
        cv2.imwrite(str(out_path), draw_boxes(img, boxes))
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
