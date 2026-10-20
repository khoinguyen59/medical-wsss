# Dispatch Assignment: team3_auditor_papers_docs

## Objective
Independent audit of all academic paper dossiers and research documents in docs/papers/, docs/references/, and docs/research/:
1. Scan for any 0-byte .md files anywhere under docs/ (FAIL if any exist).
2. Check that all 30 papers in docs/papers/*.md have >= 1,900 bytes and contain structured academic syntheses.
3. Check against source PDFs in docs/papers/*.pdf for accuracy of title, authors, architecture, and benchmarks.
4. Scan all docs/*.md for forbidden non-ASCII dashes (Invariant 12: fail on any em-dash, en-dash, horizontal bar, or unicode minus).
5. Produce a structured audit report in your handoff.md with verdict (APPROVE or REQUEST_CHANGES).

## 2026-10-20T11:53:59Z
You are team3_auditor_papers_docs, an independent auditor in Team 3 (Recheck/Audit) for SE374.R11.
Your working directory is: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\team3_auditor_papers_docs
Read your dispatch instructions at: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\team3_auditor_papers_docs\DISPATCH.md
Read the authoritative user request at: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\ORIGINAL_REQUEST.md

Your Task:
Perform an independent, strict audit of all academic paper dossiers and research documents in docs/papers/, docs/references/, and docs/research/:
1. Scan for any 0-byte .md files anywhere under docs/ (FAIL immediately if any exist).
2. Check that all 30 papers in docs/papers/*.md have >= 1,900 bytes and contain structured academic syntheses.
3. Check against source PDFs in docs/papers/*.pdf for accuracy of title, authors, architecture, and benchmarks.
4. Scan all docs/*.md for forbidden non-ASCII dashes (Invariant 12: fail on any em-dash, en-dash, horizontal bar, or unicode minus).
5. Produce a structured audit report in your handoff.md with a definitive verdict (APPROVE or REQUEST_CHANGES).

DO NOT MODIFY the target documents yourself. Report any defects back to the orchestrator.

