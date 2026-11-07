from __future__ import annotations

import argparse
import json
import math
import tkinter as tk
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "final.pt"
DEFAULT_DATA_DIR = BASE_DIR / "dataset" / "yolo11 dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI evaluation tool for a trained brain tumor YOLO model.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to trained .pt file.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_DIR), help="Path to YOLO dataset folder or data.yaml.")
    return parser.parse_args()


def resolve_data_yaml(data_path: Path) -> Path:
    if data_path.is_file() and data_path.suffix.lower() in {".yaml", ".yml"}:
        return data_path
    for candidate in (
        data_path / "data.yaml",
        data_path / "dataset.yaml",
        data_path / "data.yml",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No YOLO data.yaml found in {data_path}")


def read_yaml_text(path: Path) -> dict:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "").replace("_", "")


def safe_probability(value: float) -> float:
    return min(max(float(value), 1e-9), 1.0 - 1e-9)


def list_test_images(data_yaml: Path) -> list[Path]:
    data = read_yaml_text(data_yaml)
    test_entry = data.get("test")
    if not test_entry:
        raise FileNotFoundError(f"Dataset YAML does not define a test split: {data_yaml}")
    test_path = Path(str(test_entry))
    candidates = [
        (data_yaml.parent / test_path).resolve(),
        (data_yaml.parent / test_path.name).resolve(),
        (data_yaml.parent / "test" / "images").resolve(),
        (data_yaml.parent / "test").resolve(),
    ]
    image_dir = None
    for candidate in candidates:
        if candidate.is_dir() and candidate.name == "images":
            image_dir = candidate
            break
        if candidate.is_dir() and (candidate / "images").exists():
            image_dir = candidate / "images"
            break
    if image_dir is None:
        raise FileNotFoundError(f"Test image folder not found for {data_yaml}. Tried: {', '.join(str(p) for p in candidates)}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Test image folder not found: {image_dir}")
    return sorted([p for p in image_dir.glob("*") if p.is_file()])


def resolve_label_file(image_path: Path) -> Path | None:
    candidates = [
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt",
        image_path.parent.parent.parent / "labels" / f"{image_path.stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_detection_targets(image_paths: list[Path], class_names: list[str]) -> tuple[list[int], dict[str, int], dict[int, list[str]]]:
    class_lookup = {normalize_label(name): idx for idx, name in enumerate(class_names)}
    image_targets: list[int] = []
    class_counts: Counter[str] = Counter()
    image_labels: dict[int, list[str]] = defaultdict(list)

    for index, image_path in enumerate(image_paths):
        label_file = resolve_label_file(image_path)
        target_ids: set[int] = set()
        if label_file and label_file.exists():
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    continue
                if 0 <= class_id < len(class_names):
                    target_ids.add(class_id)
                    class_name = class_names[class_id]
                    class_counts[class_name] += 1
                    image_labels[index].append(class_name)
        image_targets.append(next(iter(target_ids), -1))
    return image_targets, dict(class_counts), image_labels


def evaluate_model(model_path: Path, data_path: Path) -> dict:
    model = YOLO(str(model_path))
    device = 0 if torch.cuda.is_available() else "cpu"
    data_yaml = resolve_data_yaml(data_path)
    data = read_yaml_text(data_yaml)
    names_raw = data.get("names") or getattr(model, "names", {})
    if isinstance(names_raw, dict):
        class_names = [names_raw[key] for key in sorted(names_raw)]
    else:
        class_names = list(names_raw)

    test_images = list_test_images(data_yaml)
    if not test_images:
        raise FileNotFoundError(f"No test images found in {data_yaml}")

    label_ids, class_counts, image_labels = load_detection_targets(test_images, class_names)

    sample_preds: list[dict[str, object]] = []
    confidences: list[float] = []
    losses: list[float] = []
    true_labels_norm: list[str] = []
    pred_labels_norm: list[str] = []
    per_class_conf: dict[str, list[float]] = defaultdict(list)
    per_class_hits: dict[str, list[int]] = defaultdict(list)

    batch_size = 8
    with ThreadPoolExecutor(max_workers=1):
        for start in range(0, len(test_images), batch_size):
            batch = test_images[start:start + batch_size]
            batch_results = model.predict(source=[str(p) for p in batch], verbose=False, device=device)
            for idx, (image_path, result) in enumerate(zip(batch, batch_results), start=start):
                probs = result.probs
                boxes = result.boxes
                top1_label = None
                top1_conf = 0.0
                if probs is not None:
                    top1_index = int(probs.top1)
                    top1_conf = float(probs.top1conf)
                    top1_label = class_names[top1_index] if 0 <= top1_index < len(class_names) else str(top1_index)
                    confidences.append(top1_conf)
                    losses.append(-math.log(safe_probability(top1_conf)))
                elif boxes is not None and len(boxes) > 0:
                    best_box = boxes[0]
                    cls_index = int(best_box.cls.item()) if best_box.cls is not None else -1
                    top1_conf = float(best_box.conf.item()) if best_box.conf is not None else 0.0
                    top1_label = class_names[cls_index] if 0 <= cls_index < len(class_names) else str(cls_index)
                    confidences.append(top1_conf)
                    losses.append(-math.log(safe_probability(top1_conf)))

                true_ids = [i for i in [label_ids[idx]] if i >= 0]
                true_label = class_names[true_ids[0]] if true_ids else "Unknown"
                normalized_true = normalize_label(true_label)
                normalized_pred = normalize_label(str(top1_label or "Unknown"))
                matched = normalized_true == normalized_pred and normalized_true != "unknown"
                true_labels_norm.append(normalized_true)
                pred_labels_norm.append(normalized_pred)
                if top1_label is not None:
                    per_class_conf[true_label].append(top1_conf)
                    per_class_hits[true_label].append(int(matched))

                sample_preds.append(
                    {
                        "file": image_path.name,
                        "true": true_label,
                        "pred": top1_label or "Unknown",
                        "confidence": top1_conf,
                        "status": "Correct" if matched else "Mismatch",
                    }
                )

    label_to_index = {normalize_label(name): idx for idx, name in enumerate(class_names)}
    confusion = [[0 for _ in class_names] for _ in class_names]
    valid_true: list[int] = []
    valid_pred: list[int] = []
    for true_label, pred_label in zip(true_labels_norm, pred_labels_norm):
        true_idx = label_to_index.get(true_label, -1)
        pred_idx = label_to_index.get(pred_label, -1)
        if true_idx >= 0 and pred_idx >= 0:
            confusion[true_idx][pred_idx] += 1
            valid_true.append(true_idx)
            valid_pred.append(pred_idx)

    top1_accuracy = accuracy_score(valid_true, valid_pred) if valid_true else 0.0
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_true,
        valid_pred,
        average="macro",
        zero_division=0,
    ) if valid_true else (0.0, 0.0, 0.0, None)
    metrics_dict = {
        "metrics/precision(B)": precision,
        "metrics/recall(B)": recall,
        "box.map50": top1_accuracy,
        "box.map": top1_accuracy,
        "box.map75": top1_accuracy,
        "f1": f1,
    }

    per_class_accuracy = {}
    for class_name in class_names:
        hits = per_class_hits.get(class_name, [])
        per_class_accuracy[class_name] = sum(hits) / len(hits) if hits else 0.0

    return {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "model_path": str(model_path),
        "data_yaml": str(data_yaml),
        "class_names": class_names,
        "sample_count": len(sample_preds),
        "accuracy": top1_accuracy,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "loss_variance": _variance(losses),
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "conf_variance": _variance(confidences),
        "metrics": metrics_dict,
        "confusion_matrix": confusion,
        "sample_preds": sample_preds,
        "class_counts": class_counts,
        "per_class_conf": {key: (sum(vals) / len(vals) if vals else 0.0) for key, vals in per_class_conf.items()},
        "per_class_accuracy": per_class_accuracy,
    }


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


class EvaluationApp:
    def __init__(self, root: tk.Tk, model_path: Path, data_path: Path):
        self.root = root
        self.model_path = model_path
        self.data_path = data_path
        self.results: dict | None = None
        self.cm_canvas = None
        self.bar_canvas = None
        self.build_ui()

    def build_ui(self) -> None:
        self.root.title("Brain Tumor YOLO Evaluation")
        self.root.geometry("1360x900")
        self.root.minsize(1100, 720)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(3, weight=0)

        top = ttk.Frame(self.root, padding=12)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Model").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value=str(self.model_path))
        ttk.Entry(top, textvariable=self.model_var, width=95).grid(row=0, column=1, padx=8, sticky="ew")
        ttk.Button(top, text="Browse", command=self.browse_model).grid(row=0, column=2)

        ttk.Label(top, text="Data").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.data_var = tk.StringVar(value=str(self.data_path))
        ttk.Entry(top, textvariable=self.data_var, width=95).grid(row=1, column=1, padx=8, pady=(8, 0), sticky="ew")
        ttk.Button(top, text="Browse", command=self.browse_data).grid(row=1, column=2, pady=(8, 0))

        ttk.Button(top, text="Run Evaluation", command=self.run_evaluation).grid(row=0, column=3, rowspan=2, padx=10)
        summary = ttk.Frame(self.root, padding=12)
        summary.grid(row=1, column=0, sticky="ew")
        self.summary_var = tk.StringVar(value="Run evaluation to see metrics.")
        ttk.Label(summary, textvariable=self.summary_var, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.metrics_var = tk.StringVar(value="")
        ttk.Label(summary, textvariable=self.metrics_var).pack(anchor="w", pady=(6, 0))

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.grid(row=2, column=0, sticky="nsew", padx=12, pady=12)

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=1)

        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        text_wrap = ttk.Frame(left)
        text_wrap.grid(row=0, column=0, sticky="nsew")
        text_wrap.rowconfigure(0, weight=1)
        text_wrap.columnconfigure(0, weight=1)
        self.text = tk.Text(text_wrap, wrap="word", height=20)
        text_scroll = ttk.Scrollbar(text_wrap, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=text_scroll.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        text_scroll.grid(row=0, column=1, sticky="ns")

        chart_frame = ttk.Frame(right)
        chart_frame.pack(fill="both", expand=True)
        chart_frame.rowconfigure(0, weight=3)
        chart_frame.rowconfigure(1, weight=2)
        chart_frame.columnconfigure(0, weight=1)

        self.cm_fig = Figure(figsize=(5.3, 4.5), dpi=100)
        self.cm_ax = self.cm_fig.add_subplot(111)
        self.cm_canvas = FigureCanvasTkAgg(self.cm_fig, master=chart_frame)
        self.cm_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", pady=(0, 10))

        self.bar_fig = Figure(figsize=(5.3, 3.3), dpi=100)
        self.bar_ax = self.bar_fig.add_subplot(111)
        self.bar_canvas = FigureCanvasTkAgg(self.bar_fig, master=chart_frame)
        self.bar_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(self.root, textvariable=self.status_var, padding=12).grid(row=3, column=0, sticky="ew")

    def browse_model(self) -> None:
        path = filedialog.askopenfilename(title="Select trained model", filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
        if path:
            self.model_var.set(path)

    def browse_data(self) -> None:
        path = filedialog.askdirectory(title="Select YOLO dataset folder")
        if path:
            self.data_var.set(path)

    def run_evaluation(self) -> None:
        model_path = Path(self.model_var.get())
        data_path = Path(self.data_var.get())
        if not model_path.exists():
            messagebox.showerror("Missing model", f"Model file not found:\n{model_path}")
            return
        if not data_path.exists():
            messagebox.showerror("Missing data", f"Data folder not found:\n{data_path}")
            return

        self.status_var.set("Evaluating model in background...")
        self.root.update_idletasks()
        worker = Thread(target=self._evaluate_background, args=(model_path, data_path), daemon=True)
        worker.start()

    def _evaluate_background(self, model_path: Path, data_path: Path) -> None:
        try:
            results = evaluate_model(model_path, data_path)
        except Exception as exc:
            self.root.after(0, lambda: self._handle_error(exc))
            return
        self.root.after(0, lambda: self._handle_success(results))

    def _handle_error(self, exc: Exception) -> None:
        self.status_var.set("Evaluation failed.")
        messagebox.showerror("Evaluation error", str(exc))

    def _handle_success(self, results: dict) -> None:
        self.results = results
        self.render_results()
        self.status_var.set("Evaluation complete.")

    def render_results(self) -> None:
        if not self.results:
            return

        res = self.results
        metrics = res["metrics"]
        self.summary_var.set(
            f"Samples: {res['sample_count']} | Accuracy: {res['accuracy']:.4f} | mAP50: {metrics.get('box.map50', 0.0):.4f} | Precision: {metrics.get('metrics/precision(B)', 0.0):.4f}"
        )
        self.metrics_var.set(
            f"Device: {res['device']} | Recall: {metrics.get('metrics/recall(B)', 0.0):.4f} | mAP50-95: {metrics.get('box.map', 0.0):.4f} | Avg loss: {res['avg_loss']:.4f} | Loss variance: {res['loss_variance']:.4f}"
        )

        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, f"Model: {res['model_path']}\n")
        self.text.insert(tk.END, f"Data: {res['data_yaml']}\n")
        self.text.insert(tk.END, f"Device: {res['device']}\n")
        self.text.insert(tk.END, f"Samples: {res['sample_count']}\n")
        self.text.insert(tk.END, f"Top-1 accuracy: {res['accuracy']:.4f}\n")
        self.text.insert(tk.END, f"Average confidence: {res['avg_confidence']:.4f}\n")
        self.text.insert(tk.END, f"Confidence variance: {res['conf_variance']:.4f}\n")
        self.text.insert(tk.END, f"Average loss: {res['avg_loss']:.4f}\n")
        self.text.insert(tk.END, f"Loss variance: {res['loss_variance']:.4f}\n")
        self.text.insert(tk.END, f"Precision: {metrics.get('metrics/precision(B)', 0.0):.4f}\n")
        self.text.insert(tk.END, f"Recall: {metrics.get('metrics/recall(B)', 0.0):.4f}\n")
        self.text.insert(tk.END, f"mAP50: {metrics.get('box.map50', 0.0):.4f}\n")
        self.text.insert(tk.END, f"mAP50-95: {metrics.get('box.map', 0.0):.4f}\n\n")

        self.text.insert(tk.END, "Per-class accuracy:\n")
        for label in res["class_names"]:
            self.text.insert(tk.END, f"- {label}: {res['per_class_accuracy'].get(label, 0.0):.4f}\n")

        self.text.insert(tk.END, "\nSample predictions:\n")
        for row in res["sample_preds"][:200]:
            self.text.insert(
                tk.END,
                f"- {row['file']}: true={row['true']}, pred={row['pred']}, confidence={row['confidence']:.4f}, {row['status']}\n",
            )

        cm = res["confusion_matrix"]
        labels = res["class_names"]
        self.cm_ax.clear()
        self.cm_fig.clear()
        self.cm_ax = self.cm_fig.add_subplot(111)
        if labels and cm:
            matrix = cm.tolist() if hasattr(cm, "tolist") else cm
            im = self.cm_ax.imshow(matrix, cmap="Blues")
            self.cm_ax.set_title("Confusion Matrix")
            self.cm_ax.set_xticks(range(len(labels)))
            self.cm_ax.set_xticklabels(labels, rotation=45, ha="right")
            self.cm_ax.set_yticks(range(len(labels)))
            self.cm_ax.set_yticklabels(labels)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    value = int(matrix[i][j]) if i < len(matrix) and j < len(matrix[i]) else 0
                    self.cm_ax.text(j, i, value, ha="center", va="center", color="black", fontsize=8)
            self.cm_fig.colorbar(im, ax=self.cm_ax, fraction=0.046, pad=0.04)
            self.cm_fig.tight_layout()
        self.cm_canvas.draw()

        self.bar_ax.clear()
        per_class_counts = [res["class_counts"].get(label, 0) for label in labels]
        per_class_accuracy = [res["per_class_accuracy"].get(label, 0.0) for label in labels]
        x = range(len(labels))
        self.bar_ax.bar([i - 0.2 for i in x], per_class_counts, width=0.4, label="Samples", color="#5da4ff")
        self.bar_ax.bar([i + 0.2 for i in x], per_class_accuracy, width=0.4, label="Accuracy", color="#58d68d")
        self.bar_ax.set_xticks(list(x))
        self.bar_ax.set_xticklabels(labels, rotation=45, ha="right")
        self.bar_ax.set_title("Class Distribution and Accuracy")
        self.bar_ax.legend()
        self.bar_fig.tight_layout()
        self.bar_canvas.draw()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = EvaluationApp(root, Path(args.model), Path(args.data))
    root.mainloop()


if __name__ == "__main__":
    main()
