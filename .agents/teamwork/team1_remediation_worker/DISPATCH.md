# Dispatch Assignment: team1_remediation_worker

## 2026-10-20T12:08:08Z

## Objective
Remediate the 4 specific defects identified by team3_auditor_papers_docs to achieve 100% dossier completeness and Invariant 24 compliance across docs/:

1. Author missing structured academic syntheses in docs/papers/:
   - docs/papers/PathMamba_arxiv_2401.04722.md (from docs/papers/PathMamba_arxiv_2401.04722.pdf: U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation, Ma et al.)
   - docs/papers/WSSS_Tissue_arxiv_2112.03053.md (from docs/papers/WSSS_Tissue_arxiv_2112.03053.pdf: Learn2Reg: comprehensive multi-task benchmark for medical image registration, Siebert et al.)
   Both files must be >= 1,900 bytes, structured academic Markdown, 100% ASCII 0x2D hyphens.

2. Re-download clean, complete PDFs for HoVerNet and ConStruct:
   - docs/papers/HoVerNet_arxiv_1812.06499.pdf (re-download from https://arxiv.org/pdf/1812.06499.pdf)
   - docs/papers/ConStruct_arxiv_2512.10316.pdf (re-download from https://arxiv.org/pdf/2512.10316.pdf)
   Ensure both PDFs have valid EOF markers and read cleanly in PyMuPDF/pypdf.

3. Fix Invariant 24 deliverable auditor violations:
   - In docs/papers/WSSS_Tissue_MLPS_arxiv_2110.08048.md: replace raw ASCII box diagram (+---+) with clean Mermaid flowchart or Markdown table.
   - In docs/papers/CONCH_arxiv_2307.12914.md: replace raw ASCII box diagram (+---+) with clean Mermaid flowchart or Markdown table.

4. Clean up stale duplicate file:
   - Remove or replace the stale 12 KB duplicate docs/GITHUB_REPOSITORIES_DEEP_ANALYSIS.md so it does not conflict with the canonical 42.8 KB specification in docs/research/GITHUB_REPOSITORIES_DEEP_ANALYSIS.md. (Prefer: remove docs/GITHUB_REPOSITORIES_DEEP_ANALYSIS.md or replace with a short pointer referencing docs/research/GITHUB_REPOSITORIES_DEEP_ANALYSIS.md).

5. Verification:
   - Run cmd /c "python src/utils/deliverable_auditor.py" to verify 100% PASSED with 0 warnings/defects.
   - Run cmd /c "python -m pytest tests/test_governance_invariants.py -v".
   - Run cmd /c "graphify update .".
   - Write handoff.md and progress.md in your working directory and notify orchestrator when done.
