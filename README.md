# T-Scanner — Brain MRI Exploration & Analysis Simulator

T-Scanner is a Python-powered research and educational prototype for brain MRI classification, scan inspection and case review. The interface is a **static HTML/CSS page with plain JavaScript** served directly by FastAPI.

## UI/runtime constraints

- No Node.js runtime or npm commands are required.
- No React, Vite, Electron, Three.js, React Three Fiber or other frontend framework/3D library is used.
- `frontend/index.html`, `frontend/styles.css` and `frontend/app.js` are static assets.
- The interactive schematic brain is drawn by a small custom WebGL renderer written in vanilla JavaScript.
- Python + FastAPI serves the UI and handles AI inference, research history and MRI reference retrieval.

## Run on Windows (without a virtual environment)

T-Scanner does **not** require or create a virtual environment. The project targets **Python 3.10.8** (later Python 3.10 patch releases are also accepted). Do not use Python 3.11, 3.12 or 3.13 for this project. Open PowerShell in the project root.

First, check which Python installations Windows can see:

```powershell
py -0p
```

Use the Python 3.10 interpreter explicitly. This avoids a generic `python` command accidentally selecting a stale environment or a different Python version:

```powershell
py -3.10 -m pip install -r backend\requirements.txt
```

Start the application from the project root:

```powershell
py -3.10 main.py
```

Verify that the selected interpreter is the expected version:

```powershell
py -3.10 --version
```

You should see Python 3.10.8 (or another 3.10.x maintenance release). Use `py -3.10` for both installation and launch; avoid plain `python` when it points to a stale Python installation.

Open the address printed in the terminal, normally `http://127.0.0.1:8000`.

You can also double-click `start_tscanner.bat`. It explicitly selects Python 3.10 and checks the version before installing dependencies. It does not create or activate `.venv`.

## Docker

From the project root, build and run with:

```bash
docker build -f backend/Dockerfile -t t-scanner .
docker run --rm -p 8000:8000 t-scanner
```

## Model files

The backend searches for the classifier in this order:

1. `backend/models/final.pt`
2. `backend/runs/brain_tumor_classifier/weights/final.pt`
3. `backend/runs/brain_tumor_classifier/weights/best.pt`
4. `backend/models/best.pt`
5. `backend/models/detection.pt`

The optional detector is loaded from `backend/models/detection.pt`.

## Included environments and capabilities

- Neural Laboratory: schematic interactive brain, rotate/zoom, anatomy selection, structure layers, explode view, transparency and cross-section.
- MRI Scanner Chamber: 2D scan import, single-scan analysis and batch analysis.
- Tumour Exploration Room: model result, class probabilities, adjustable confidence threshold, detection output, explanation, brightness/contrast/gamma, inversion, grayscale, zoom/pan, pixel inspection, ruler, ROI selection/export, before/after enhancement comparison and optional Grad-CAM.
- Case Comparison Laboratory: ranked reference-image search using HOG visual descriptors and cosine similarity.
- Research Observatory: SQLite case history, notes, favourites, CSV export and print-to-PDF.
- Neuro-oncology Atlas: general educational information and short anatomy/classification learning questions.

## Similar MRI reference library

The supplied project did not include labelled reference MRI cases. The comparison feature therefore starts empty rather than fabricating sample patients or similarity values.

Add real, permissioned, de-identified reference images to the relevant class folder:

```text
backend/reference_cases/
  glioma/
  meningioma/
  pituitary/
  notumor/
```

Supported image formats: JPG, JPEG, PNG, WEBP and BMP. An optional JSON file with the same filename stem can hold metadata, for example:

```json
{
  "case_id": "GLIOMA-001",
  "sequence": "T2-weighted",
  "description": "De-identified teaching reference"
}
```

The displayed percentage is a **HOG visual-descriptor cosine similarity score**, not a medical similarity judgement, diagnostic confidence or proof of clinical equivalence. The current implementation is a lightweight local retrieval baseline; it is not a pretrained medical imaging encoder.

## Important technical limits

- The brain is a procedural schematic anatomy model. It is not a licensed detailed anatomical model and is not a patient-specific 3D reconstruction.
- Tumour markers in the 3D anatomy are not inferred from the MRI. Patient-specific placement would require volumetric segmentation and spatial registration.
- The current scanner accepts 2D images. True axial, coronal and sagittal reconstruction requires DICOM/NIfTI volume support and voxel spacing; unavailable views are intentionally disabled.
- Measurement results are in pixels unless the source contains usable spatial calibration metadata. No millimetre values are invented.
- Grad-CAM is attempted only when the loaded classifier graph supports a compatible gradient path. When unsupported, the UI reports why instead of displaying a fabricated heatmap. A Grad-CAM heatmap is model attention, not tumour segmentation.
- The current HOG-based reference search uses only images manually added to the reference library.
- Gemini explanations are optional. Copy `backend/.env.example` to `backend/.env` and set `GEMINI_API_KEY`; local fallback text is returned when the service is not configured.

## API endpoints

- `GET /api/health`
- `GET /api/model`
- `GET /api/model/logs`
- `GET /api/settings`
- `PATCH /api/settings/confidence`
- `POST /api/analyze`
- `POST /api/analyze/batch`
- `POST /api/explain`
- `POST /api/cases/similar`
- `POST /api/gradcam/{case_id}`
- `GET /api/history`
- `GET /api/history/{case_id}`
- `GET /api/history/{case_id}/image`
- `PATCH /api/history/{case_id}`
- `GET /api/export/history.csv`
- `GET /api/research/stats`

## Local data and privacy

Analysis history and copied scan images are stored locally in `backend/data/`. Uploaded request files are removed from the temporary upload folder after inference. The locally archived cases remain until manually removed from that data folder/database. Do not use identifiable patient images unless you have the required permissions and safeguards.

## Disclaimer

T-Scanner is an AI-assisted research and educational prototype, not a medical diagnostic device. Its outputs do not replace radiologist interpretation, specialist review or clinical decision-making.
