"""Entry point to launch the CCTV Layout AI FastAPI server.

Usage:
    python backend/main.py          # from repo root
    python main.py                  # from backend/ directory
"""
import sys
from pathlib import Path
import uvicorn

# Ensure repository root is on sys.path so `import backend...` works from any cwd
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    uvicorn.run("backend.api.app:app", host="0.0.0.0", port=8000, reload=True, app_dir=str(ROOT))
