from __future__ import annotations

import csv
import json
import os
import random
import shutil
import socket
import sqlite3
import time
import uuid
from urllib.parse import quote
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from gemini_service import get_scan_explanation
from inference import YOLOBrainTumorEngine

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
BRAIN_ASSET_DIR = BASE_DIR / "3d" / "brain"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
SCAN_STORE = DATA_DIR / "scans"
REFERENCE_DIR = BASE_DIR / "reference_cases"
DATASET_DIR = BASE_DIR / "dataset"
DB_PATH = DATA_DIR / "tscanner.sqlite3"
DEFAULT_MODEL_CANDIDATES = [
    BASE_DIR / "models" / "final.pt",
    BASE_DIR / "runs" / "brain_tumor_classifier" / "weights" / "final.pt",
    BASE_DIR / "runs" / "brain_tumor_classifier" / "weights" / "best.pt",
    BASE_DIR / "models" / "best.pt",
    BASE_DIR / "models" / "detection.pt",
    BASE_DIR / "runs" / "brain_tumor_detector" / "weights" / "best.pt",
]
RECENT_ANALYSIS_LOGS: deque[dict[str, object]] = deque(maxlen=120)
MODEL_PATH = Path(os.getenv("MODEL_PATH", ""))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "20"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

for folder in (UPLOAD_DIR, OUTPUT_DIR, DATA_DIR, SCAN_STORE, REFERENCE_DIR, BRAIN_ASSET_DIR / "source", BRAIN_ASSET_DIR / "textures"):
    folder.mkdir(parents=True, exist_ok=True)

if not str(MODEL_PATH) or str(MODEL_PATH) == ".":
    MODEL_PATH = next((candidate for candidate in DEFAULT_MODEL_CANDIDATES if candidate.exists()), DEFAULT_MODEL_CANDIDATES[0])

engine = YOLOBrainTumorEngine(MODEL_PATH, CONF_THRESHOLD)
app = FastAPI(title="T-Scanner Brain MRI Research Simulator", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def db_connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database() -> None:
    with db_connect() as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                scan_path TEXT NOT NULL,
                predicted_type TEXT,
                tumor_detected INTEGER NOT NULL DEFAULT 0,
                best_confidence REAL,
                processing_time_ms REAL,
                result_json TEXT NOT NULL,
                explanation_json TEXT,
                notes TEXT NOT NULL DEFAULT '',
                favorite INTEGER NOT NULL DEFAULT 0
            )"""
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC)")


init_database()


def validate_upload(filename: str | None, size: int) -> Path:
    if not filename:
        raise HTTPException(400, "Missing filename")
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported image format: {extension or 'unknown'}")
    if size > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_FILE_MB} MB limit")
    return Path(filename)


async def save_upload(file: UploadFile) -> Path:
    data = await file.read()
    validate_upload(file.filename, len(data))
    target = UPLOAD_DIR / f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    target.write_bytes(data)
    return target


def persist_case(
    source_path: Path,
    original_name: str,
    result: dict,
    explanation: dict | None = None,
) -> str:
    case_id = uuid.uuid4().hex
    suffix = Path(original_name).suffix.lower() or ".png"
    saved_path = SCAN_STORE / f"{case_id}{suffix}"
    shutil.copy2(source_path, saved_path)
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with db_connect() as connection:
        connection.execute(
            """INSERT INTO analyses
            (id, created_at, filename, scan_path, predicted_type, tumor_detected,
             best_confidence, processing_time_ms, result_json, explanation_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                case_id,
                created_at,
                Path(original_name).name,
                str(saved_path),
                result.get("predicted_type"),
                int(bool(result.get("tumor_detected"))),
                result.get("best_confidence"),
                result.get("processing_time_ms"),
                json.dumps(result, ensure_ascii=False),
                json.dumps(explanation, ensure_ascii=False) if explanation else None,
            ),
        )
    return case_id


def row_to_case(row: sqlite3.Row, include_detail: bool = False) -> dict:
    item = {
        "id": row["id"],
        "created_at": row["created_at"],
        "filename": row["filename"],
        "predicted_type": row["predicted_type"],
        "tumor_detected": bool(row["tumor_detected"]),
        "best_confidence": row["best_confidence"],
        "processing_time_ms": row["processing_time_ms"],
        "notes": row["notes"],
        "favorite": bool(row["favorite"]),
        "image_url": f"/api/history/{row['id']}/image",
    }
    if include_detail:
        item["result"] = json.loads(row["result_json"] or "{}")
        item["explanation"] = json.loads(row["explanation_json"] or "null")
    return item


def append_log(result: dict) -> None:
    RECENT_ANALYSIS_LOGS.appendleft(
        {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "request_id": result.get("request_id"),
            "case_id": result.get("case_id"),
            "filename": result.get("filename"),
            "task": result.get("task", "classification"),
            "status": result.get("status", "success"),
            "tumor_detected": result.get("tumor_detected"),
            "predicted_type": result.get("predicted_type"),
            "best_confidence": result.get("best_confidence"),
            "processing_time_ms": result.get("processing_time_ms"),
            "detections": result.get("detections", []),
        }
    )


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/assets/brain", StaticFiles(directory=str(BRAIN_ASSET_DIR)), name="brain-assets")


@app.get("/api/brain-model")
def brain_model_status():
    model_file = BRAIN_ASSET_DIR / "source" / "brain.glb"
    texture_names = (
        "gltf_embedded_0.png",
        "gltf_embedded_1.png",
        "gltf_embedded_1@channels=A.png",
        "gltf_embedded_2.png",
    )
    return {
        "available": model_file.is_file(),
        "model_url": "/assets/brain/source/brain.glb",
        "textures": {
            name: (BRAIN_ASSET_DIR / "textures" / name).is_file()
            for name in texture_names
        },
    }


@app.get("/")
def home():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(404, "T-Scanner HTML interface is missing")
    return FileResponse(index)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "backend": "online",
        "model_loaded": engine.loaded,
        "model_path": str(MODEL_PATH),
        "device": engine.device,
        "interface": "static-html-vanilla-js",
        "renderer": "custom-webgl-no-threejs",
    }


@app.get("/api/model")
def model_info():
    return engine.info()


@app.get("/api/settings")
def get_settings():
    return {"confidence_threshold": engine.conf, "max_file_mb": MAX_FILE_MB}


@app.patch("/api/settings/confidence")
def set_confidence(payload: dict):
    try:
        value = float(payload.get("value"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "Confidence threshold must be a number from 0.05 to 0.90") from exc
    if not 0.05 <= value <= 0.90:
        raise HTTPException(400, "Confidence threshold must be between 0.05 and 0.90")
    engine.conf = value
    return {"status": "updated", "confidence_threshold": engine.conf}


@app.get("/api/model/logs")
def model_logs():
    return {
        "model_path": str(MODEL_PATH),
        "active_device": engine.device,
        "recent_upload_logs": list(RECENT_ANALYSIS_LOGS),
        "logs": [
            {"source": "classifier", "lines": ["classification inference", "predictions are generated by the loaded model"]},
            {"source": "detector", "lines": ["optional detector", "localization is only shown when available"]},
        ],
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    path = await save_upload(file)
    started = time.perf_counter()
    try:
        result = engine.predict(path)
        result["filename"] = Path(file.filename or path.name).name
        result["request_id"] = uuid.uuid4().hex
        result["processing_time_ms"] = round((time.perf_counter() - started) * 1000, 2)
        result["case_id"] = persist_case(path, result["filename"], result)
        append_log(result)
        return result
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(503 if "model" in str(exc).lower() else 422, str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/explain")
async def explain(file: UploadFile = File(...)):
    path = await save_upload(file)
    started = time.perf_counter()
    try:
        classification = engine.predict(path)
        explanation = get_scan_explanation(path, classification)
        request_id = uuid.uuid4().hex
        processing_time_ms = round((time.perf_counter() - started) * 1000, 2)
        filename = Path(file.filename or path.name).name
        classification["filename"] = filename
        classification["request_id"] = request_id
        classification["processing_time_ms"] = processing_time_ms
        case_id = persist_case(path, filename, classification, explanation)
        classification["case_id"] = case_id
        append_log(classification)
        return {
            "classification": classification,
            "explanation": explanation,
            "filename": filename,
            "processing_time_ms": processing_time_ms,
            "case_id": case_id,
        }
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(503 if "model" in str(exc).lower() else 422, str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/analyze/batch")
async def analyze_batch(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files supplied")
    results = []
    for file in files:
        path = await save_upload(file)
        started = time.perf_counter()
        try:
            result = engine.predict(path)
            result["filename"] = Path(file.filename or path.name).name
            result["request_id"] = uuid.uuid4().hex
            result["processing_time_ms"] = round((time.perf_counter() - started) * 1000, 2)
            result["case_id"] = persist_case(path, result["filename"], result)
            append_log(result)
            results.append(result)
        except (RuntimeError, ValueError) as exc:
            results.append({"filename": Path(file.filename or path.name).name, "status": "error", "error": str(exc)})
        finally:
            path.unlink(missing_ok=True)
    return {"count": len(results), "results": results}


@app.get("/api/history")
def history(limit: int = 100, offset: int = 0, favorites_only: bool = False):
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    query = "SELECT * FROM analyses"
    if favorites_only:
        query += " WHERE favorite = 1"
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    with db_connect() as connection:
        rows = connection.execute(query, (limit, offset)).fetchall()
        total = connection.execute("SELECT COUNT(*) FROM analyses" + (" WHERE favorite = 1" if favorites_only else "")).fetchone()[0]
    return {"count": total, "results": [row_to_case(row) for row in rows]}


@app.get("/api/history/{case_id}")
def history_detail(case_id: str):
    with db_connect() as connection:
        row = connection.execute("SELECT * FROM analyses WHERE id = ?", (case_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Case not found")
    return row_to_case(row, include_detail=True)


@app.get("/api/history/{case_id}/image")
def history_image(case_id: str):
    with db_connect() as connection:
        row = connection.execute("SELECT scan_path FROM analyses WHERE id = ?", (case_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Case not found")
    path = Path(row["scan_path"]).resolve()
    if SCAN_STORE.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "Stored MRI image not found")
    return FileResponse(path)


@app.patch("/api/history/{case_id}")
def update_history(case_id: str, payload: dict):
    notes = str(payload.get("notes", ""))[:5000]
    favorite = 1 if payload.get("favorite") else 0
    with db_connect() as connection:
        cursor = connection.execute("UPDATE analyses SET notes = ?, favorite = ? WHERE id = ?", (notes, favorite, case_id))
    if cursor.rowcount == 0:
        raise HTTPException(404, "Case not found")
    return {"status": "updated", "id": case_id, "notes": notes, "favorite": bool(favorite)}


@app.get("/api/export/history.csv")
def export_history_csv():
    with db_connect() as connection:
        rows = connection.execute(
            "SELECT id, created_at, filename, predicted_type, tumor_detected, best_confidence, processing_time_ms, notes, favorite FROM analyses ORDER BY created_at DESC"
        ).fetchall()
    def rows_stream():
        from io import StringIO
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["Case ID", "Date", "Filename", "Predicted Type", "Tumour Detected", "Confidence", "Processing Time (ms)", "Notes", "Favourite"])
        yield output.getvalue()
        output.seek(0); output.truncate(0)
        for row in rows:
            writer.writerow(list(row))
            yield output.getvalue()
            output.seek(0); output.truncate(0)
    return StreamingResponse(rows_stream(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=tscanner_analysis_history.csv"})


@app.get("/api/research/stats")
def research_stats():
    with db_connect() as connection:
        total, detected, avg_conf = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(tumor_detected), 0), AVG(best_confidence) FROM analyses"
        ).fetchone()
        groups = connection.execute(
            "SELECT COALESCE(predicted_type, 'Unknown') AS label, COUNT(*) AS count FROM analyses GROUP BY predicted_type ORDER BY count DESC"
        ).fetchall()
    return {
        "total_scans": total,
        "tumor_flagged": detected,
        "average_confidence": round(avg_conf, 4) if avg_conf is not None else None,
        "class_counts": [{"label": row["label"], "count": row["count"]} for row in groups],
    }


def normalized_category(label: str) -> str:
    value = "".join(ch for ch in label.lower() if ch.isalnum())
    aliases = {
        "glioma": "glioma", "meningioma": "meningioma", "pituitary": "pituitary",
        "notumor": "notumor", "notumour": "notumor", "normal": "notumor",
    }
    return aliases.get(value, value)


def hog_descriptor(image: np.ndarray) -> np.ndarray | None:
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        descriptor = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
        vector = descriptor.compute(resized)
        if vector is None:
            return None
        vector = vector.astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else None
    except Exception:
        return None


@app.post("/api/cases/similar")
async def similar_cases(file: UploadFile = File(...), predicted_type: str = Form(...), limit: int = Form(6)):
    """Return randomly sampled real MRI files from the predicted dataset class."""
    path = await save_upload(file)
    try:
        category = normalized_category(predicted_type)
        if category not in {"glioma", "meningioma", "pituitary", "notumor"}:
            return {
                "category": category,
                "results": [],
                "count": 0,
                "message": "The model category does not match one of the four labelled dataset folders.",
            }

        candidates: list[tuple[str, Path]] = []
        split_counts: dict[str, int] = {"Train": 0, "Test": 0}
        for split in ("Train", "Test"):
            class_dir = DATASET_DIR / split / category
            if not class_dir.is_dir():
                continue
            files = [item for item in class_dir.rglob("*") if item.is_file() and item.suffix.lower() in ALLOWED_EXTENSIONS]
            split_counts[split] = len(files)
            candidates.extend((split, item) for item in files)

        if not candidates:
            expected = f"backend/dataset/Train/{category}/ and backend/dataset/Test/{category}/"
            return {
                "category": category,
                "results": [],
                "count": 0,
                "message": f"No images were found. Add MRI files to {expected}",
                "split_counts": split_counts,
            }

        requested = max(1, min(int(limit or 6), 12))
        chosen = random.SystemRandom().sample(candidates, min(requested, len(candidates)))
        results = []
        for index, (split, reference) in enumerate(chosen, start=1):
            results.append({
                "id": f"{split.lower()}-{index:03d}",
                "display_name": f"Reference MRI {index:02d}",
                "image_url": f"/api/dataset/{category}/{split}/{quote(reference.relative_to(DATASET_DIR / split / category).as_posix(), safe='/')}",
                "category": category,
                "dataset_split": split,
                "metadata": {"description": "Randomly sampled from the same labelled dataset class."},
            })
        return {
            "category": category,
            "method": "Random sample from matching Train/Test class folders",
            "results": results,
            "count": len(results),
            "available_count": len(candidates),
            "split_counts": split_counts,
        }
    finally:
        path.unlink(missing_ok=True)


@app.get("/api/dataset/{category}/{split}/{filename:path}")
def dataset_image(category: str, split: str, filename: str):
    safe_category = normalized_category(category)
    safe_split = split.capitalize()
    relative_name = Path(filename)
    if relative_name.is_absolute() or ".." in relative_name.parts:
        raise HTTPException(404, "Dataset image not found")
    if safe_category not in {"glioma", "meningioma", "pituitary", "notumor"} or safe_split not in {"Train", "Test"}:
        raise HTTPException(404, "Dataset image not found")
    folder = (DATASET_DIR / safe_split / safe_category).resolve()
    path = (folder / relative_name).resolve()
    if folder not in path.parents or not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(404, "Dataset image not found")
    return FileResponse(path)


@app.post("/api/gradcam/{case_id}")
def generate_gradcam(case_id: str):
    with db_connect() as connection:
        row = connection.execute("SELECT scan_path, result_json FROM analyses WHERE id = ?", (case_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Case not found")
    source_path = Path(row["scan_path"])
    result = json.loads(row["result_json"] or "{}")
    if not hasattr(engine, "generate_gradcam"):
        return {"available": False, "reason": "Grad-CAM is not available in this model adapter."}
    output_path = OUTPUT_DIR / f"{case_id}_gradcam.jpg"
    generated = engine.generate_gradcam(source_path, output_path, result.get("predicted_class_id"))
    if not generated.get("available"):
        return generated
    return {**generated, "image_url": f"/api/annotated/{output_path.name}"}


@app.get("/api/annotated/{filename}")
def annotated(filename: str):
    safe = Path(filename).name
    path = (OUTPUT_DIR / safe).resolve()
    if OUTPUT_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "Annotated image not found")
    return FileResponse(path)


def find_free_port(start_port: int, host: str = "127.0.0.1", search_limit: int = 50) -> int:
    for port in range(start_port, start_port + search_limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port found from {start_port} to {start_port + search_limit - 1}")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    requested_port = int(os.getenv("PORT", "8000"))
    port = find_free_port(requested_port, host=host, search_limit=int(os.getenv("PORT_SEARCH_LIMIT", "50")))
    print(f"T-Scanner: http://{host}:{port}", flush=True)
    uvicorn.run("main:app", host=host, port=port, reload=False)
