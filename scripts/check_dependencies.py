from __future__ import annotations

import importlib


REQUIRED = [
    "torch",
    "torchvision",
    "ultralytics",
    "PIL",
    "yaml",
    "cv2",
    "onnx",
    "onnxruntime",
]


def main() -> None:
    missing = []
    for name in REQUIRED:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)

    if missing:
        raise SystemExit(f"missing dependencies: {', '.join(missing)}")

    import torch
    import ultralytics

    print(f"torch={torch.__version__}")
    print(f"cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"ultralytics={ultralytics.__version__}")
    print("all required dependencies are available")


if __name__ == "__main__":
    main()
