from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "yolo11 dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
MODELS_DIR = BASE_DIR / "models"
FINAL_MODEL_NAME = "detection.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO brain-tumor detector.")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=512, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default=None, help="Device id, e.g. 0, cpu, or cuda:0.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO detection model.")
    parser.add_argument("--project", default=str(BASE_DIR / "runs"), help="Training output folder.")
    parser.add_argument("--name", default="brain_tumor_detector", help="Run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing detection dataset config: {DATA_YAML}")

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(DATA_YAML),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    elif torch.cuda.is_available():
        train_kwargs["device"] = 0

    results = model.train(**train_kwargs)
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training finished but best model was not found at {best_pt}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / FINAL_MODEL_NAME
    shutil.copy2(best_pt, target)
    print(f"Detection model saved to: {target}")


if __name__ == "__main__":
    main()
