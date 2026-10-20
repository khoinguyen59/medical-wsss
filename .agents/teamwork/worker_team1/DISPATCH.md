# DISPATCH LOG

## 2026-10-18T16:12:05Z
You are Worker Team 1 (Data Ingestion & Patient-Level Splitting Manifest) for the SE374.R11 WSSS project.

Working directory: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\worker_team1
Project root: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA
Authoritative Request: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\ORIGINAL_REQUEST.md
Orchestrator ID: 7ffe6ac5-9a5c-4c39-ba4b-d0861435f647

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Exclusive write ownership:
- src/config/bcss_patient_splits.json
- src/datasets/__init__.py
- tests/test_patient_split.py

Tasks:
1. Construct the complete, deterministic 151-patient split manifest in `src/config/bcss_patient_splits.json` for BCSS-WSSS.
   - Total patients: 151 distinct TCGA breast cancer patient identifiers (e.g. TCGA-A2-A0T2, TCGA-BH-A0B8, etc.).
   - Train split: 91 patients (approx 60%).
   - Validation split: 30 patients (approx 20%).
   - Test split: 30 patients (approx 20%).
   - Zero patient leakage across splits: verify pairwise disjoint sets (P_train inter P_val = empty, P_train inter P_test = empty, P_val inter P_test = empty).
2. Update `src/datasets/__init__.py` and `src/datasets/patient_split.py` if needed to provide a clean loader function `load_patient_splits(config_path=...)` that returns the split dictionary and runs `verify_patient_leakage()`.
3. Update `tests/test_patient_split.py` to add a test `test_151_patient_split_manifest_zero_leakage()` verifying that the manifest exists, has exactly 151 patients, and has 0.0% patient overlap.
4. Execute unit tests via:
   cmd /c "C:\Python313\python.exe -m pytest tests/test_patient_split.py -v"
5. Document all commands and results in your handoff report `c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\worker_team1\handoff.md`.
6. Send completion message to orchestrator (7ffe6ac5-9a5c-4c39-ba4b-d0861435f647).

Governance rules:
- Strictly follow Invariant 12: Standard keyboard hyphen ASCII 0x2D '-' only. Absolutely NO em-dash or en-dash.
- Strictly follow Invariant 13: Windows OS cmd /c for all shell execution.
- Strictly follow Invariant 15: Zero one-off scratch scripts.
- Invariant 08: Patient-level anti-leakage protocol.
