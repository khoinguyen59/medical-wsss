## 2026-10-18T16:12:05Z
You are Worker Team 2 (Model Backbone, PDA Generator & Benchmark Engine) for the SE374.R11 WSSS project.

Working directory: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\worker_team2
Project root: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA
Authoritative Request: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\ORIGINAL_REQUEST.md
Orchestrator ID: 7ffe6ac5-9a5c-4c39-ba4b-d0861435f647

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Exclusive write ownership:
- src/models/backbone.py
- src/models/pda_generator.py
- src/models/wsss_tissue.py
- tests/test_pda_backbone.py

Tasks:
1. Create `src/models/backbone.py`:
   - Implement `MultiStageFeatureBackbone` (supporting multi-stage feature extraction with stages `b4_5`, `b5_2`, `bn7` and Global Average Pooling classification logits for multi-label weak supervision).
   - Ensure clean modular design with customizable channels and dropout.
2. Create `src/models/pda_generator.py`:
   - Implement `ProgressiveDropoutAttention` (PDA) for CAM generation.
   - Algorithm: Compute initial CAM $M_c$. Over $T$ iterative steps (e.g. $T=3$), identify top-$p$ percentile attention activation regions, apply progressive channel/spatial masking (dropout rate $\delta_t$), compute complementary CAMs $M_c^{(t)}$, and aggregate them: $\hat{M}_c = \sum_t w_t M_c^{(t)}$.
   - This prevents CAM from over-concentrating on small discriminative foci and prevents boundary bleeding into adjacent stroma.
3. Update `src/models/wsss_tissue.py`:
   - Refactor to import from `src/models/backbone.py` and `src/models/pda_generator.py`.
   - Update `WSSSBenchmarkingEngine` so that mode `"proposed"` incorporates PDA along with Dynamic Progressive Thresholding, Otsu Gating, and Uncertainty Regularization.
   - Maintain full backward compatibility for modes `"baseline"`, `"classmix"`, `"relocalization"`, and `"proposed"`.
4. Create `tests/test_pda_backbone.py`:
   - Unit tests for `MultiStageFeatureBackbone` forward pass and feature shapes.
   - Unit tests for `ProgressiveDropoutAttention` CAM computation and expansion properties.
   - Unit tests verifying all 4 benchmark configurations run cleanly through `WSSSBenchmarkingEngine`.
5. Run tests via:
   cmd /c "C:\Python313\python.exe -m pytest tests/test_pda_backbone.py tests/test_benchmark_components.py -v"
6. Document results in `c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\worker_team2\handoff.md`.
7. Send completion message to orchestrator (7ffe6ac5-9a5c-4c39-ba4b-d0861435f647).

Governance rules:
- Strictly follow Invariant 12: Standard keyboard hyphen ASCII 0x2D '-' only. Absolutely NO em-dash or en-dash.
- Strictly follow Invariant 13: Windows OS cmd /c for all shell execution.
- Strictly follow Invariant 15: Zero one-off scratch scripts.
