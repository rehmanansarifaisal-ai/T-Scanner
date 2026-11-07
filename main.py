"""Root launcher for T-Scanner.

Run from the project directory with:
    python main.py

No virtual environment is created or activated by this launcher.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# T-Scanner was built and is maintained for Python 3.10.8.
# Accept later 3.10 patch releases, but do not silently run under 3.11/3.12/3.13.
if sys.version_info[:2] != (3, 10) or sys.version_info < (3, 10, 8):
    current = ".".join(str(part) for part in sys.version_info[:3])
    raise SystemExit(
        "T-Scanner requires Python 3.10.8 (or a later Python 3.10 patch release).\n"
        f"Current interpreter: Python {current}.\n"
        "On Windows, launch it with: py -3.10 main.py"
    )

PROJECT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_DIR / "backend"

if not BACKEND_DIR.is_dir():
    raise SystemExit(f"T-Scanner backend folder was not found: {BACKEND_DIR}")

# The backend modules live directly inside backend/ (not in an installed package).
sys.path.insert(0, str(BACKEND_DIR))

# Load backend/main.py under a distinct module name so it cannot accidentally
# resolve to this root launcher named main.py.
backend_spec = importlib.util.spec_from_file_location(
    "tscanner_backend_main", BACKEND_DIR / "main.py"
)
if backend_spec is None or backend_spec.loader is None:
    raise SystemExit("Could not load the T-Scanner backend entry point.")
backend_main = importlib.util.module_from_spec(backend_spec)
sys.modules[backend_spec.name] = backend_main
backend_spec.loader.exec_module(backend_main)


def run() -> None:
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    requested_port = int(os.getenv("PORT", "8000"))
    search_limit = int(os.getenv("PORT_SEARCH_LIMIT", "50"))
    port = backend_main.find_free_port(requested_port, host=host, search_limit=search_limit)
    print(f"T-Scanner: http://{host}:{port}", flush=True)
    uvicorn.run(backend_main.app, host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
