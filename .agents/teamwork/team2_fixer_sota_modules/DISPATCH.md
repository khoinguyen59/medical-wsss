# Dispatch Assignment: team2_fixer_sota_modules

## Objective
Dedicated to verifying, refining, and mathematically aligning the 5 repo-grounded SOTA methods M1-M5 in src/models/wsss_inference_methods.py:
- Method M1 (Baseline Single-Scale).
- Method M2 (Multi-View D4 TTA with CGM class gate y_cls > 0.1).
- Method M3 (Long-Tail Prior Calibration with Cross-Stage CAM Fusion & LYM/NEC minority recovery).
- Method M4 (Spatial Bilateral Refinement with PistoSeg RFM affinity propagation + ARML edge-preserving bilateral filtering / island suppression).
- Method M5 (Proposed SOTA Ensemble combining all components + Ultra-Safe Pure-Glass Guard).
- Ensure smooth interoperability with team2_fixer_engine's WSSSTissueStage1NetCAM.

## Strict Criteria
- 100% ASCII 0x2D standard hyphens ONLY (zero em-dashes / en-dashes).
- All shell executions via cmd /c "...".
- Zero scratch scripts.
- Write handoff.md and progress.md in your working directory and notify orchestrator when done.

## 2026-10-20T11:53:59Z
You are team2_fixer_sota_modules, part of Team 2 (Implement/Fix) for SE374.R11.
Your working directory is: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\team2_fixer_sota_modules
Read your dispatch instructions at: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\team2_fixer_sota_modules\DISPATCH.md
Read the authoritative user request at: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\ORIGINAL_REQUEST.md
Read the mathematical and architectural specifications at: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\docs\research\GITHUB_REPOSITORIES_DEEP_ANALYSIS.md

Your Task:
Verify and refine the 5 repo-grounded SOTA methods M1-M5 in src/models/wsss_inference_methods.py:
1. M1_Baseline_SingleScale
2. M2_MultiView_D4_TTA (incorporating CGM y_cls > 0.1 class gating)
3. M3_LongTail_Prior_Calibration (incorporating Cross-Stage CAM Fusion & LYM/NEC minority recovery)
4. M4_Spatial_Bilateral_Refinement (PistoSeg RFM affinity propagation + ARML edge-preserving bilateral filtering / island suppression)
5. M5_Proposed_SOTA_Ensemble (combining all above modules + Ultra-Safe Pure-Glass Guard)
Ensure seamless compatibility with WSSSTissueStage1NetCAM and test with dummy tensors.

MANDATORY INTEGRITY WARNING: DO NOT CHEAT.
Strict Criteria:
- 100% ASCII 0x2D standard hyphens ONLY (zero em-dashes / en-dashes).
- All shell executions via cmd /c "...".
- Zero scratch scripts.
- Write handoff.md and progress.md in your working directory and notify the orchestrator when completed.
