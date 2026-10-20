# Operational Plan - 3-Team Architecture (Expanded Headcount)

## Team 1 - Research Team (Team Research) [Headcount: 2]
- `team1_researcher_papers` (Role: Academic Paper & Dossier Specialist) [Conv ID: 88a979d4-a6b6-4b10-a128-0327d7fa9d24]
  - Status: in-progress (restoring 9 0-byte markdown files)
- `team1_researcher_repos` (Role: Code-to-Code Repository Analyst) [Conv ID: 326d4788-025b-410c-92bd-d31b85f34e41]
  - Status: completed (updated docs/research/GITHUB_REPOSITORIES_DEEP_ANALYSIS.md with 42.8 KB specification)

## Team 2 - Implementation / Fix Team (Team Implement/Fix) [Headcount: 4]
- `team2_fixer_engine` (Role: Medical WSSS Engine & Architecture Engineer) [Conv ID: b1dcccc5-cbf8-4833-9a9d-372240c8f82a]
  - Status: in-progress (porting WSSSTissueStage1NetCAM and implementing 5 SOTA methods in src/models/wsss_inference_methods.py)
- `team2_fixer_colab` (Role: Colab GPU Pipeline & Notebook Suite Engineer) [Conv ID: 61fccc96-db50-43b6-9b57-d7ed1d2c508e]
  - Status: in-progress (updating scripts/generate_colab_variants.py and 4 Colab GPU notebooks)
- `team2_fixer_sota_modules` (Role: SOTA Modules Specialist) [Conv ID: 7018767d-d089-4969-940f-a97607c34bd2]
  - Status: in-progress (verifying and refining M1-M5, RFM affinity module, CAM fusion, and long-tail calibration)
- `team2_fixer_tests_integration` (Role: Tests & Integration Engineer) [Conv ID: 7b950c84-a8ba-4ea8-b9f3-7e93843c281b]
  - Status: in-progress (expanding unit tests in tests/test_colab_notebook.py for WSSSTissueStage1NetCAM and M1-M5 invariants)

## Team 3 - Recheck / Audit Team (Team Recheck/Audit) [Headcount: 4]
- `team3_auditor_papers_docs` (Role: Papers & Docs Auditor) [Conv ID: 08ae6269-17cf-44c3-b0ff-614ec74582c1]
  - Status: in-progress (auditing all 30 papers in docs/papers/, references, and research for 0 0-byte files, >=1900 bytes, and ASCII hyphens)
- `team3_auditor_repo_parity` (Role: Repo Parity Auditor) [Conv ID: c36ec28b-2948-4171-9f61-9c4cf70f1599]
  - Status: in-progress (auditing code parity of WSSSTissueStage1NetCAM and M1-M5 against repos/*)
- `team3_auditor_colab_notebooks` (Role: Colab Notebooks Auditor) [Conv ID: a7b82e37-0cf8-40c0-910b-fe9c4954a9c0]
  - Status: in-progress (auditing 4 Colab notebooks, dual checkpoints, JSON validity, and hardware configurations)
- `team3_auditor_governance_pytest` (Role: Governance & Pytest Auditor) [Conv ID: a14da241-6c0d-4525-a2b8-5c9a4a023f35]
  - Status: in-progress (running full pytest suite, auditing 25 AGENTS.md invariants, deliverable auditor, and updating graphify)
