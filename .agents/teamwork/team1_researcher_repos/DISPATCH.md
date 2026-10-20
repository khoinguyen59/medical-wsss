# Dispatch Assignment: team1_researcher_repos

## Objective
Update docs/research/GITHUB_REPOSITORIES_DEEP_ANALYSIS.md with a comprehensive code-to-code comparative matrix across cloned repositories in repos/:
- WSSS-Tissue, LowMagWSS, ARML, PistoSeg, HisynSeg, PBIP, ConStruct, S2C, SAM2PATH, PFM_Segmentation, SEINE, WaWeHis.
- Analyze exact implementations, architectures, loss formulations, supervision level, dataset compatibility (BCSS, WSSS4LUAD, Camelyon16), and performance metrics.
- Formulate the exact architectural & mathematical specifications for improving our 68.71% mIoU baseline:
  - ResNet38d Stage-1 Net_CAM porting details
  - Method M1 (Single-Scale Baseline)
  - Method M2 (Multi-View D4 TTA with Class Gating Margin y_cls > 0.1)
  - Method M3 (Long-Tail Prior Calibration with Cross-Stage CAM Fusion & LYM/NEC minority recovery)
  - Method M4 (Spatial Bilateral Refinement with PistoSeg RFM affinity propagation + ARML edge-preserving bilateral filtering / island suppression)
  - Method M5 (Proposed SOTA Ensemble combining all components + Ultra-Safe Pure-Glass Guard)

## Strict Criteria
- 100% ASCII 0x2D standard hyphens ONLY (zero em-dashes / en-dashes).
- Deep code-to-code grounding citing real files in repos/*.
- Write handoff.md and progress.md in your working directory and notify orchestrator when done.


## 2026-10-20T11:41:55Z
You are team1_researcher_repos, specialized in code-to-code repository analysis for SE374.R11.
Your working directory is: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\team1_researcher_repos
Read your dispatch instructions at: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\team1_researcher_repos\DISPATCH.md
Read the authoritative user request at: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\ORIGINAL_REQUEST.md

Your Task:
Update docs/research/GITHUB_REPOSITORIES_DEEP_ANALYSIS.md with a comprehensive code-to-code comparative matrix across cloned repositories in repos/:
- WSSS-Tissue, LowMagWSS, ARML, PistoSeg, HisynSeg, PBIP, ConStruct, S2C, SAM2PATH, PFM_Segmentation, SEINE, WaWeHis.
- Analyze exact implementations, architectures, loss formulations, supervision level, dataset compatibility (BCSS, WSSS4LUAD, Camelyon16), and performance metrics.
- Formulate the exact architectural & mathematical specifications for improving our 68.71% mIoU baseline:
  - ResNet38d Stage-1 Net_CAM porting details
  - Method M1 (Single-Scale Baseline)
  - Method M2 (Multi-View D4 TTA with Class Gating Margin y_cls > 0.1)
  - Method M3 (Long-Tail Prior Calibration with Cross-Stage CAM Fusion & LYM/NEC minority recovery)
  - Method M4 (Spatial Bilateral Refinement with PistoSeg RFM affinity propagation + ARML edge-preserving bilateral filtering / island suppression)
  - Method M5 (Proposed SOTA Ensemble combining all components + Ultra-Safe Pure-Glass Guard)

Strict Criteria:
- 100% ASCII 0x2D standard hyphens ONLY (zero em-dashes / en-dashes).
- Deep code-to-code grounding citing real files in repos/*.
- Write handoff.md and progress.md in your working directory and notify the orchestrator when completed.
