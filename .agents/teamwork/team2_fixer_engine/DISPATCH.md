# Dispatch Assignment: team2_fixer_engine

## 2026-10-20T11:41:55Z

## Objective
Port exact ResNet38d + Net_CAM Stage-1 architecture (WSSSTissueStage1NetCAM) and implement 5 repo-grounded SOTA methods in src/models/wsss_inference_methods.py:
1. Port exact ResNet38d + Net_CAM Stage-1 architecture directly from repos/WSSS-Tissue/network/resnet38d.py and repos/WSSS-Tissue/network/resnet38_cls.py so stage1_checkpoint_trained_on_bcss.pth loads with 100% key match (strict=True).
2. Implement 5 genuine repo-grounded SOTA methods in MultiMethodWSSSEngine (compatible when model_stage1 is loaded and with graceful fallback when model_stage1=None):
   - M1_Baseline_SingleScale
   - M2_MultiView_D4_TTA (CGM y_cls > 0.1 + Multi-View Dihedral TTA)
   - M3_LongTail_Prior_Calibration (Cross-Stage CAM Fusion + LYM/NEC minority recovery)
   - M4_Spatial_Bilateral_Refinement (PistoSeg RFM affinity propagation + ARML edge-preserving bilateral / island suppression)
   - M5_Proposed_SOTA_Ensemble (Full proposed pipeline combining Stage-1 Net_CAM CGM + CAM Fusion + Multi-View TTA + Long-Tail Prior Calibration + RFM/Edge-Preserving Spatial Refinement + Ultra-Safe Pure-Glass Guard)
3. MANDATORY INTEGRITY WARNING:
   DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. An independent auditor will verify your work.
4. Strict Criteria:
   - 100% ASCII 0x2D hyphens only (zero em-dashes/en-dashes).
   - All shell executions via cmd /c "...".
   - Zero scratch scripts.
   - Run verification and report passing tests.
   - Write handoff.md and progress.md in your working directory and notify orchestrator when done.

## 2026-10-20T11:50:48Z
**Context**: Implementation of WSSSTissueStage1NetCAM and MultiMethodWSSSEngine
**Content**: Team 1 (Research) has finalized `docs/research/GITHUB_REPOSITORIES_DEEP_ANALYSIS.md` with the complete architectural, layer-naming, and mathematical specifications for `WSSSTissueStage1NetCAM` and Methods M1 through M5. You may inspect `docs/research/GITHUB_REPOSITORIES_DEEP_ANALYSIS.md` for exact details.
**Action**: Integrate these specifications into `src/models/wsss_inference_methods.py`.

