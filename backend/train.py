from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
SOURCE_TRAIN_DIR = DATASET_DIR / "Train"
SOURCE_TEST_DIR = DATASET_DIR / "Test"
WORK_DIR = DATASET_DIR / "_prepared_classification"
PREPARED_TRAIN_DIR = WORK_DIR / "train"
PREPARED_VAL_DIR = WORK_DIR / "val"
PREPARED_TEST_DIR = WORK_DIR / "test"
DEFAULT_MODEL = "yolov8n-cls.pt"
FINAL_MODEL_NAME = "final.pt"
MODELS_DIR = BASE_DIR / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO brain-tumor classifier.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=224, help="Training image size.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--device", default=None, help="Device id, e.g. 0, cpu, or cuda:0.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split from Train.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base YOLO classification model.")
    parser.add_argument("--project", default=str(BASE_DIR / "runs"), help="Training output folder.")
    parser.add_argument("--name", default="brain_tumor_classifier", help="Run name.")
    return parser.parse_args()


def copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    for class_dir in source.iterdir():
        if not class_dir.is_dir():
            continue
        for image_file in class_dir.glob("*"):
            if image_file.is_file():
                dest_dir = target / class_dir.name
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_file, dest_dir / image_file.name)


def split_train_val(source_train: Path, train_target: Path, val_target: Path, val_ratio: float, seed: int) -> None:
    if train_target.exists():
        shutil.rmtree(train_target)
    if val_target.exists():
        shutil.rmtree(val_target)

    rng = random.Random(seed)
    for class_dir in source_train.iterdir():
        if not class_dir.is_dir():
            continue
        images = [p for p in class_dir.iterdir() if p.is_file()]
        if not images:
            continue
        rng.shuffle(images)
        val_count = max(1, int(len(images) * val_ratio)) if len(images) > 1 else 0
        val_images = set(images[:val_count])
        for image_file in images:
            dest_root = val_target if image_file in val_images else train_target
            dest_dir = dest_root / class_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_file, dest_dir / image_file.name)


def prepare_dataset(val_ratio: float, seed: int) -> None:
    if not SOURCE_TRAIN_DIR.exists():
        raise FileNotFoundError(f"Missing training folder: {SOURCE_TRAIN_DIR}")
    if not SOURCE_TEST_DIR.exists():
        raise FileNotFoundError(f"Missing test folder: {SOURCE_TEST_DIR}")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    split_train_val(SOURCE_TRAIN_DIR, PREPARED_TRAIN_DIR, PREPARED_VAL_DIR, val_ratio, seed)
    copy_tree(SOURCE_TEST_DIR, PREPARED_TEST_DIR)


def main() -> None:
    args = parse_args()
    prepare_dataset(args.val_ratio, args.seed)

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(WORK_DIR),
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
    final_pt = Path(results.save_dir) / "weights" / FINAL_MODEL_NAME
    if not best_pt.exists():
        raise FileNotFoundError(f"Training finished but best model was not found at {best_pt}")
    shutil.copy2(best_pt, final_pt)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, MODELS_DIR / FINAL_MODEL_NAME)
    print(f"Final model saved to: {final_pt}")


if __name__ == "__main__":
    main()
