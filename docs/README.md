# SE374.R11: Medical Image Processing - Documentation Index

> **Course Profile:** SE374.R11 - Phat trien phan mem xu ly anh y khoa (Medical Image Processing Software Development)  
> **Faculty:** Khoa Ky thuat Phan mem (Faculty of Software Engineering), University of Information Technology (UIT), VNU-HCM  
> **Topic Focus:** Weakly-Supervised Semantic Segmentation (WSSS) on Gigapixel Whole-Slide Images (WSI) for Breast Cancer Subtyping (BCSS Cohort)

---

## 1. Documentation Structure Overview

This directory houses all academic, technical, architectural, and governance documentation for the SE374.R11 capstone project. The repository follows a decoupled documentation architecture:

| Directory | Scope and Purpose | Key Artifacts |
|:---|:---|:---|
| `architecture/` | System architecture, multi-agent governance, and technical specifications | `AGENT_TEAM_SPEC.md`, `SYSTEM_TECHNICAL_SPECIFICATION.md`, `TEAMWORK_MULTI_AGENT_WORKFLOW.md`, `CO_CHE_HOAT_DONG_AI_AGENT.html` |
| `course/` | Course curriculum, topic allocations, and official project tracking registry | `SE374.R11 - Đồ án cuối kỳ.xlsx`, `topic.md`, `Khảo Sát WSSS Trên WSI.docx` |
| `design/` | Scientific figures, system diagrams, and pipeline dataflows in vector format | `fig1_standard_wsi_pipeline.svg`, `fig2_system_architecture.svg`, `fig3_pipeline_dataflow.svg` |
| `diary/` | Research progress logs, engineering audit trails, and execution checkpoints | `audit/` sub-logs, sprint retrospectives |
| `lectures/` | Official lecture notes, slides, and medical imaging fundamentals | `Lecture_01_Introduction_to_Medical_Image.md`, `Lecture_01_Introduction_to_Medical_Image.pdf` |
| `notes/` | Advisory meeting minutes, team notes, and instructor feedback records | Working notes and review logs |
| `papers/` | Core SOTA scientific papers in dual format (full-text PDF and structured Markdown) | `WSSS-Tissue`, `LowMagWSS`, `ARML`, `CGRL`, `CG_GLORE`, `CLAM`, `C2RM_Seg` |
| `references/` | Deep research dossiers, digital pathology surveys, and background analyses | `deep-research-report.md`, `Digital_Pathology_WSI_Resources.md`, `Gigapixel_WSI_Segmentation_Survey.md` |
| `reports/` | Milestone deliverables, project evaluation dossiers, and audit attestations | `PROJECT_EVALUATION_DOSSIER.md`, `audit/` reports |
| `requirements/` | Dynamic project specifications and decoupled requirements | `project_spec.md` (8 topics matrix, GPU profiles, dataset layouts) |
| `research/` | Research methodology, comparative benchmarks, patient-split protocols, and topic roadmaps | `COMPARATIVE_BENCHMARKING_PROTOCOL.md`, `DATASET_SPECIFICATION_AND_PATIENT_LEAKAGE_PROTOCOL.md`, `GITHUB_REPOSITORIES_DEEP_ANALYSIS.md` |
| `templates/` | Official templates for proposal, thesis, final report, and Springer LNCS | `01_Mau_Mini_Proposal_Thay_Nghiep_GOC.docx`, `02_Bao_Cao_Do_An_UIT_Sinh_Vien_Tai_Truc_Tiep.docx`, Springer LNCS templates |

---

## 2. Core Governance and Invariants

All technical and research work documented within this directory adheres to the governance invariants defined in `AGENTS.md`:

1. **Patient-Level Anti-Leakage Protocol:** All dataset partitions (train, validation, test) must be split strictly by Patient ID / Case ID to prevent data snooping across gigapixel WSI slices.
2. **Absolute Scientific Integrity (Zero Mockup):** 100% of reported quantitative metrics (mIoU, Dice, C-Index, Kaplan-Meier log-rank p-values) derive from genuine experiment logs and reproducible code execution.
3. **Typography Standard (Invariant 12):** Strictly standard keyboard hyphen '-' (ASCII 0x2D) only. Non-ASCII dashes (em-dash, en-dash, horizontal bar, minus sign) are prohibited across all markdown, code, and documentation files.
4. **Decoupled Architecture:** Clean separation between core model services (`src/`), automated verification suites (`tests/`), reproducibility notebooks (`notebooks/`), and structured documentation (`docs/`).

---

## 3. Recommended Reading Path

1. **Requirements & Scope:** Start with `docs/requirements/project_spec.md` for complete topic definitions and dataset configurations.
2. **Research Methodology:** Review `docs/research/DATASET_SPECIFICATION_AND_PATIENT_LEAKAGE_PROTOCOL.md` and `docs/research/COMPARATIVE_BENCHMARKING_PROTOCOL.md`.
3. **Architecture & Engineering:** Read `docs/architecture/SYSTEM_TECHNICAL_SPECIFICATION.md` and inspect vector diagrams in `docs/design/`.
4. **Evaluation & Verification:** Consult `docs/reports/PROJECT_EVALUATION_DOSSIER.md` for comprehensive benchmark comparisons against published baselines.
