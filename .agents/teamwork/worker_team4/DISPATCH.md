## 2026-10-18T16:12:06Z
You are Worker Team 4 (Clinical Web Workstation & API) for the SE374.R11 WSSS project.

Working directory: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\worker_team4
Project root: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA
Authoritative Request: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\ORIGINAL_REQUEST.md
Orchestrator ID: 7ffe6ac5-9a5c-4c39-ba4b-d0861435f647

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Exclusive write ownership:
- src/visualization/api_server.py
- src/visualization/clinical_viewer.py
- src/visualization/static/
- tests/test_api_server.py

Tasks:
1. Create `src/visualization/api_server.py` using FastAPI:
   - Endpoints:
     * `GET /api/v1/health`: Returns system status, model state, and memory/VRAM usage.
     * `POST /api/v1/predict/patch`: Accepts an RGB image patch (multipart upload or JSON base64), runs WSSS segmentation with Otsu clinical gating, and returns:
       - Predicted segmentation mask (PNG base64 or label matrix).
       - Class distribution breakdown (TUM, STR, LYM, NEC, BACK percentages).
       - Clinical biomarkers: Viable Tumor Percentage (VTP) and Tumor Stroma Ratio (TSR) computed via `src/losses_metrics/metrics.py`.
     * `POST /api/v1/predict/wsi`: Accepts multi-patch tile batch or coordinates and returns combined heatmaps.
     * `GET /`: Serves the clinical workstation web interface or redirects to `/viewer`.
     * `GET /viewer`: Serves an interactive HTML/JS OpenSeadragon clinical viewer.
2. Create `src/visualization/clinical_viewer.py` and `src/visualization/static/index.html`:
   - Interactive clinical interface with OpenSeadragon deep zoom viewing for ROI patches and WSI slides.
   - Features:
     * Deep zoom viewer viewport.
     * Color-coded overlay layer for 5 tissue classes: Tumor (Red `#E41A1C`), Stroma (Green `#4DAF4A`), Lymphocytic Infiltrate (Blue `#377EB8`), Necrosis (Yellow `#FF7F00`), Background (Transparent/Gray `#999999`).
     * Opacity slider for seamless overlay toggling over H&E tissue.
     * Clinical biomarker dashboard displaying VTP and TSR.
3. Create `tests/test_api_server.py`:
   - Unit tests using `fastapi.testclient.TestClient`.
   - Test `GET /api/v1/health` returns status `healthy`.
   - Test `POST /api/v1/predict/patch` returns valid segmentation, VTP, and TSR.
   - Test `GET /viewer` returns HTML with OpenSeadragon assets.
4. Run tests via:
   cmd /c "C:\Python313\python.exe -m pytest tests/test_api_server.py -v"
5. Document all commands and results in `c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\worker_team4\handoff.md`.
6. Send completion message to orchestrator (7ffe6ac5-9a5c-4c39-ba4b-d0861435f647).

Governance rules:
- Strictly follow Invariant 12: Standard keyboard hyphen ASCII 0x2D '-' only. Absolutely NO em-dash or en-dash.
- Strictly follow Invariant 13: Windows OS cmd /c for all shell execution.
- Strictly follow Invariant 15: Zero one-off scratch scripts.
