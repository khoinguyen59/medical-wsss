## 2026-10-18T16:02:01Z
You are the Codebase and Architecture Explorer for the SE374.R11 Weakly-Supervised Histopathology Semantic Segmentation (WSSS) project.

Working directory: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\explorer_survey
Project root: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA
Authoritative Request: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\ORIGINAL_REQUEST.md
Orchestrator ID: 7ffe6ac5-9a5c-4c39-ba4b-d0861435f647

Your tasks:
1. Thoroughly survey the existing files, modules, and structure in src/, tests/, product/, docs/, scripts/, requirements.txt.
2. Check existing implementations and maturity for the 4 core module areas:
   - Team 1: BCSS-WSSS dataset with patient-level splitting (151 cases) and clinical Otsu tissue gating in HSV space.
   - Team 2: Model backbone, Progressive Dropout Attention (PDA) CAM generation, and 4 benchmark configurations (Baseline CAM, ClassMix, ARML, Proposed Adaptive Pipeline) in src/models/wsss_tissue.py and src/pipeline/comparative_runner.py.
   - Team 3: DeepLabV3+ with CGM, multi-layer pseudo-supervision loss, boundary uncertainty regularization, zero-mockup metrics suite (mIoU, mDice, per-class IoU, VTP, TSR).
   - Team 4: FastAPI backend and clinical interface with OpenSeadragon deep zoom viewer.
3. Check the test suite in tests/:
   - Enumerate all test files and functions.
   - Run the test suite via: cmd /c pytest tests/ -v
   - Document passing and failing tests.
4. Check whether product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md exists or if draft reports exist in product/.
5. Check static compliance:
   - Invariant 12: Scan for any non-ASCII hyphens (em-dashes U+2014, en-dashes U+2013).
   - Invariant 15: Check if any one-off scratch scripts exist.
6. Write your comprehensive structured findings to:
   c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\explorer_survey\survey_report.md
   and write handoff.md.
7. Send a message to the orchestrator (7ffe6ac5-9a5c-4c39-ba4b-d0861435f647) summarizing findings and providing the path to survey_report.md.

Governance rules:
- Strictly follow Invariant 12: Only ASCII 0x2D '-' hyphens. Zero em-dashes, zero en-dashes.
- Follow Invariant 13: Windows OS cmd /c for shell execution.
- Read-only exploration agent: do not alter production source code.
