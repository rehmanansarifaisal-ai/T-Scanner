from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "detection.pt"
DEFAULT_DATA_YAML = BASE_DIR / "dataset" / "yolo11 dataset" / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained brain-tumor YOLO detection model.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to trained detection .pt file.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="Path to YOLO data.yaml file.")
    parser.add_argument("--imgsz", type=int, default=512, help="Validation image size.")
    parser.add_argument("--device", default=None, help="Device id, e.g. 0, cpu, or cuda:0.")
    parser.add_argument("--json-out", default=None, help="Optional path to save the evaluation summary as JSON.")
    return parser.parse_args()


def normalize_model_names(names: dict | list) -> list[str]:
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    return [str(item) for item in names]


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    data_path = Path(args.data)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_path}")

    model = YOLO(str(model_path))
    val_kwargs: dict[str, object] = {
        "data": str(data_path),
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device is not None:
        val_kwargs["device"] = args.device
    elif torch.cuda.is_available():
        val_kwargs["device"] = 0

    metrics = model.val(**val_kwargs)
    names = normalize_model_names(getattr(model, "names", {}))
    class_map = getattr(metrics, "names", None)
    if isinstance(class_map, dict):
        names = normalize_model_names(class_map)

    summary = {
        "model_path": str(model_path),
        "data_yaml": str(data_path),
        "device": "cuda:0" if torch.cuda.is_available() and args.device is None else (args.device or "cpu"),
        "imgsz": args.imgsz,
        "class_names": names,
        "metrics": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        },
        "per_class": {},
        "speed": {
            "preprocess_ms": float(getattr(metrics.speed, "preprocess", 0.0)),
            "inference_ms": float(getattr(metrics.speed, "inference", 0.0)),
            "loss_ms": float(getattr(metrics.speed, "loss", 0.0)),
            "postprocess_ms": float(getattr(metrics.speed, "postprocess", 0.0)),
        },
    }

    if hasattr(metrics, "results_dict") and isinstance(metrics.results_dict, dict):
        summary["raw_results"] = {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in metrics.results_dict.items()
        }

    if hasattr(metrics, "box") and hasattr(metrics.box, "maps") and metrics.box.maps is not None:
        maps = list(metrics.box.maps)
        for idx, class_name in enumerate(names):
            if idx < len(maps):
                summary["per_class"][class_name] = float(maps[idx])

    print(json.dumps(summary, indent=2))
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to: {out_path}")


if __name__ == "__main__":
    main()
