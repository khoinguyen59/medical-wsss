# Dispatch Assignment: team3_auditor_governance_pytest

## 2026-10-20T11:54:00Z
You are team3_auditor_governance_pytest, an independent auditor in Team 3 (Recheck/Audit) for SE374.R11.
Your working directory is: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\team3_auditor_governance_pytest
Read your dispatch instructions at: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\team3_auditor_governance_pytest\DISPATCH.md
Read the authoritative user request at: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\ORIGINAL_REQUEST.md

Your Task:
Perform an independent, strict audit of test execution, AST knowledge graph, and all 25 AGENTS.md governance invariants:
1. Run full pytest regression suite via cmd /c "python -m pytest tests/test_colab_notebook.py tests/test_governance_invariants.py -v".
2. Audit compliance with AGENTS.md Invariant 07 (Zero Mockup), Invariant 08 (Patient-Level Split), Invariant 12 (ASCII hyphen 0x2D only), Invariant 13 (Windows shell cmd /c), Invariant 15 (Zero scratch scripts), Invariant 24 (Publication-grade monochrome visuals), and Invariant 25 (Automated deliverable auditor).
3. Run cmd /c "python src/utils/deliverable_auditor.py" to ensure 100% deliverable quality pass.
4. Run cmd /c "graphify update ." to ensure AST knowledge graph is up to date.
5. Produce a structured audit report in your handoff.md with a definitive verdict (APPROVE or REQUEST_CHANGES).

DO NOT MODIFY source files directly. Report any defects back to the orchestrator.
