from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from google import genai
    from google.genai import types
except ImportError:  # Gemini is optional; local explanations remain available.
    genai = None
    types = None


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
DEFAULT_MODEL = "gemini-2.5-flash-lite"


def _load_env_file() -> None:
    if not ENV_PATH.exists():
        return
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@lru_cache(maxsize=1)
def get_gemini_client():
    _load_env_file()
    if genai is None:
        raise RuntimeError("The optional google-genai package is not installed.")
    api_key = os.getenv("gemini_api") or os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API")
    if not api_key:
        raise RuntimeError("Missing Gemini API key in backend/.env or environment variables.")
    return genai.Client(api_key=api_key)


def build_medical_prompt(classification: dict[str, Any]) -> str:
    label = classification.get("predicted_type") or "unknown"
    confidence = classification.get("best_confidence")
    detected = classification.get("tumor_detected")
    return (
        "You are assisting with an educational brain MRI classifier. "
        "Write a clear, patient-friendly explanation based on the model result. "
        "Do not claim a diagnosis. Do not invent unsupported facts. "
        "Return JSON only with these keys: summary, detected_class, confidence_note, "
        "precautions (array of 5-7 short items), symptoms (array of 6-10 short items), "
        "when_to_seek_help, and disclaimer.\n\n"
        f"Model result:\n"
        f"- predicted_type: {label}\n"
        f"- confidence: {confidence}\n"
        f"- tumor_detected: {detected}\n"
    )


def get_scan_explanation(image_path: Path, classification: dict[str, Any]) -> dict[str, Any]:
    fallback = {
        "summary": f"The model classified this MRI as {classification.get('predicted_type') or 'unknown'}. This is a screening result, not a medical diagnosis.",
        "detected_class": classification.get("predicted_type") or "unknown",
        "confidence_note": f"Model confidence was {classification.get('best_confidence', 'unknown')}.",
        "precautions": [
            "Do not rely on the result alone for treatment decisions.",
            "Share the scan and report with a qualified doctor.",
            "Seek urgent care for sudden or severe neurological symptoms.",
            "Keep follow-up imaging and clinical review on schedule.",
            "Do not delay evaluation if symptoms are worsening.",
        ],
        "symptoms": [
            "Headache",
            "Nausea or vomiting",
            "Blurred or double vision",
            "Seizures",
            "Weakness or numbness",
            "Difficulty speaking or understanding",
            "Balance problems",
            "Personality or memory changes",
        ],
        "when_to_seek_help": "Seek urgent medical attention if symptoms are sudden, severe, or rapidly worsening.",
        "disclaimer": "Educational output only. This app cannot diagnose cancer or replace a radiologist or neurologist.",
    }

    try:
        client = get_gemini_client()
        model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        if model_name == "gemini-flash-lite-latest":
            model_name = DEFAULT_MODEL

        prompt = build_medical_prompt(classification)
        image_bytes = image_path.read_bytes()
        extension = image_path.suffix.lower()
        mime_type = {".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp"}.get(extension, "image/jpeg")
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        text = (response.text or "").strip()
        if text:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return {**fallback, **parsed}
    except Exception:
        pass

    return fallback
