# Dispatch Records

## 2026-10-20T11:38:14Z

You are the Project Orchestrator for SE374.R11 Weakly-Supervised Semantic Segmentation (WSSS) Histopathology pipeline project.
Your working directory is: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\orchestrator_3
Authoritative request is at: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\ORIGINAL_REQUEST.md (specifically see section ## 2026-10-20T11:36:26Z).

Mandate:
Organize and lead a full multi-agent team with a Leader dividing work across a group of worker agents and a dedicated group of recheck/auditor agents to independently verify and cross-check all work.

Tasks to accomplish:
1. R1: Full Restoration & Cross-Verification of Academic Paper & Research Dossiers (docs/):
   - Restore all 9 remaining 0-byte Markdown files in docs/papers/ (CLAM_arxiv_2004.09666.md, HoVerNet_arxiv_1812.06499.md, PathMamba_MICCAI2024.md, UNI_arxiv_2308.05814.md, Virchow_arxiv_2309.07778.md) and docs/references/ (deep-research-report.md, Digital_Pathology_WSI_Resources.md, Gigapixel_WSI_Segmentation_Survey.md, Nghien_Cuu_AI_WSI.md) from corresponding PDFs in docs/papers/*.pdf and workspace research sources.
   - Every file must have a complete, structured academic Markdown synthesis (> 2,500 bytes, >= 1,900 bytes minimum per criteria) with 100% ASCII 0x2D standard hyphens (zero em-dashes / en-dashes).
   - Update docs/research/GITHUB_REPOSITORIES_DEEP_ANALYSIS.md with a comprehensive code-to-code comparative matrix across cloned repositories (WSSS-Tissue, LowMagWSS, ARML, PistoSeg, HisynSeg, PBIP, ConStruct, S2C, SAM2PATH, PFM_Segmentation, SEINE, WaWeHis).

2. R2: Code-to-Code Repo-Grounded Multi-Method WSSS Engine (src/models/wsss_inference_methods.py):
   - Port exact ResNet38d + Net_CAM Stage-1 architecture (WSSSTissueStage1NetCAM) directly from repos/WSSS-Tissue/network/resnet38d.py and repos/WSSS-Tissue/network/resnet38_cls.py so stage1_checkpoint_trained_on_bcss.pth loads with 100% key match (strict=True).
   - Implement 5 genuine repo-grounded SOTA methods in MultiMethodWSSSEngine (compatible when model_stage1 is loaded and with graceful fallback when model_stage1=None):
     - M1_Baseline_SingleScale
     - M2_MultiView_D4_TTA (CGM y_cls > 0.1 + Multi-View Dihedral TTA)
     - M3_LongTail_Prior_Calibration (Cross-Stage CAM Fusion + LYM/NEC minority recovery)
     - M4_Spatial_Bilateral_Refinement (PistoSeg RFM affinity propagation + ARML edge-preserving bilateral / island suppression)
     - M5_Proposed_SOTA_Ensemble (Full proposed pipeline combining Stage-1 Net_CAM CGM + CAM Fusion + Multi-View TTA + Long-Tail Prior Calibration + RFM/Edge-Preserving Spatial Refinement + Ultra-Safe Pure-Glass Guard)

3. R3: Synchronized Google Colab GPU Notebook Suite (notebooks/ & scripts/generate_colab_variants.py):
   - Update scripts/generate_colab_variants.py and regenerate all 4 GPU-specific Colab notebooks (SE374_Colab_WSSS_T4_Sanity150.ipynb, SE374_Colab_WSSS_T4_Full4986.ipynb, SE374_Colab_WSSS_L4_Full4986.ipynb, SE374_Colab_WSSS_A100_Full4986.ipynb) via cmd /c "python scripts/generate_colab_variants.py".
   - Notebooks must verify and download both official BCSS checkpoints:
     - stage1_checkpoint_trained_on_bcss.pth (ID 118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc)
     - stage2_checkpoint_trained_on_bcss.pth (ID 1cXm29ko8UevbaeW6YreSg-vuNaQiN_je)
   - Notebooks must instantiate both model and model_stage1 (WSSSTissueStage1NetCAM), evaluate all 5 methods on real BCSS-WSSS data, export ablation tables and render comparison grids.

4. Rigorous Recheck & Governance Verification:
   - Zero 0-byte .md files anywhere under docs/.
   - 100% of unit tests pass: cmd /c "python -m pytest tests/test_colab_notebook.py tests/test_governance_invariants.py -v".
   - Zero non-ASCII hyphens (Invariant 12).
   - All shell commands use cmd /c "..." (Invariant 13).
   - Zero scratch scripts (Invariant 15).
   - Keep graph current via cmd /c "graphify update .".

## 2026-10-20T11:39:57Z

URGENT DIRECTIVE FROM USER - STRICT 3-TEAM ORGANIZATIONAL STRUCTURE REQUIRED:

You MUST organize and divide your subagents into 3 distinct, specialized teams (do NOT lump everything into generic workers):

1. Team 1 - Research Team (Team Research):
   - Dedicated strictly to reading the 30 academic papers in docs/papers/ (and docs/references/), cross-referencing the 24 cloned GitHub repositories in repos/ (WSSS-Tissue, LowMagWSS, ARML, PistoSeg, HisynSeg, PBIP, ConStruct, S2C, SAM2PATH, PFM_Segmentation, SEINE, WaWeHis), restoring the 9 remaining 0-byte Markdown dossiers in docs/papers/ and docs/references/, updating docs/research/GITHUB_REPOSITORIES_DEEP_ANALYSIS.md, and formulating the exact mathematical & architectural specifications to improve our 68.71% mIoU baseline.

2. Team 2 - Implementation / Fix Team (Team Implement/Fix):
   - Dedicated strictly to coding, implementing, and fixing based on Team Research's specifications:
     - Porting WSSSTissueStage1NetCAM (ResNet38d + Net_CAM from repos/WSSS-Tissue/network/resnet38d.py & resnet38_cls.py) and the 5 repo-grounded SOTA methods into src/models/wsss_inference_methods.py.
     - Updating scripts/generate_colab_variants.py and regenerating all 4 GPU Colab notebooks (SE374_Colab_WSSS_T4_Sanity150.ipynb, SE374_Colab_WSSS_T4_Full4986.ipynb, SE374_Colab_WSSS_L4_Full4986.ipynb, SE374_Colab_WSSS_A100_Full4986.ipynb) with both Stage-1 (118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc) and Stage-2 (1cXm29ko8UevbaeW6YreSg-vuNaQiN_je) checkpoints.
     - Updating unit tests in tests/test_colab_notebook.py.

3. Team 3 - Recheck / Audit Team (Team Recheck/Audit):
   - Independent verification and auditing team that NEVER writes the initial implementation, but strictly cross-checks Team 1 (Research) and Team 2 (Implement/Fix) against all acceptance criteria:
     - Verifying zero 0-byte .md files in docs/ and all docs/papers/*.md >= 1,900 bytes.
     - Verifying exact layer/parameter parity of WSSSTissueStage1NetCAM against repos/WSSS-Tissue.
     - Running cmd /c "python -m pytest tests/test_colab_notebook.py tests/test_governance_invariants.py -v".
     - Auditing 100% compliance with AGENTS.md Invariant 07 (Zero Mockup), Invariant 08 (Patient-Level Split), Invariant 12 (ASCII hyphen 0x2D only), Invariant 15 (Zero scratch scripts), and running graphify update ..
     - Sending any defects back to Team 2 (Team Implement/Fix) until 100% pass.

Ensure that all subagents spawned are explicitly named and grouped according to these 3 teams. Record this structure in your plan.md, BRIEFING.md, and progress.md.

## 2026-10-20T11:52:53Z

URGENT HEADCOUNT MANDATE FROM USER - EXPAND TEAM SIZES IMMEDIATELY:

The user explicitly clarified the required minimum headcount for each of the 3 teams under your leadership:
- Team 1 (Team Research): MUST have >= 2 agents (currently: team1_researcher_papers & team1_researcher_repos).
- Team 2 (Team Implement/Fix): MUST have >= 4 agents! You must immediately spawn at least 2 additional subagents so Team 2 has >= 4 agents:
  1. team2_fixer_engine (active)
  2. team2_fixer_colab (active)
  3. team2_fixer_sota_modules (SPAWN NOW: dedicated to verifying and refining the 5 repo-grounded SOTA methods M1-M5, RFM affinity module from PistoSeg, Cross-Stage CAM fusion from HisynSeg/PBIP, and Long-Tail Prior Calibration in src/models/)
  4. team2_fixer_tests_integration (SPAWN NOW: dedicated to writing and expanding comprehensive unit tests in tests/test_colab_notebook.py for WSSSTissueStage1NetCAM, dual-checkpoint loading, and M1-M5 numerical invariants, and fixing any integration issues)
- Team 3 (Team Recheck/Audit): MUST have >= 4 agents! You must immediately spawn at least 4 distinct auditor subagents for Team 3:
  1. team3_auditor_papers_docs: Audits all 30 paper .md files in docs/papers/, docs/references/, and docs/research/ (verifies zero 0-byte files, >= 1,900 bytes each, accuracy against PDFs, and zero non-ASCII dashes).
  2. team3_auditor_repo_parity: Audits code-to-code parity of src/models/wsss_inference_methods.py (WSSSTissueStage1NetCAM and M1-M5) line-by-line against repos/WSSS-Tissue, repos/LowMagWSS, repos/ARML, repos/PistoSeg, repos/HisynSeg, repos/PBIP, and repos/ConStruct.
  3. team3_auditor_colab_notebooks: Audits scripts/generate_colab_variants.py and all 4 generated .ipynb notebooks in notebooks/ (verifies dual checkpoint IDs 118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc and 1cXm29ko8UevbaeW6YreSg-vuNaQiN_je, hardware configs for T4/L4/A100, and JSON syntax validity).
  4. team3_auditor_governance_pytest: Runs full pytest suite (tests/test_colab_notebook.py, tests/test_governance_invariants.py), audits all 25 AGENTS.md invariants (Zero Mockup, Patient-Level Split, ASCII hyphen 0x2D only, Zero scratch scripts), and runs graphify update ..

Total subagents under you MUST be >= 10 agents (Team 1 >= 2, Team 2 >= 4, Team 3 >= 4). Spawn them immediately, update your plan.md and BRIEFING.md, and send back a report containing all their conversation IDs!
