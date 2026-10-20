## 2026-10-18T16:37:49Z

You are the Performance & Clinical Stress Fix Engineer for the SE374.R11 WSSS project.

Working directory: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\worker_fix_clinical
Project root: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA
Authoritative Request: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\ORIGINAL_REQUEST.md
Orchestrator ID: 7ffe6ac5-9a5c-4c39-ba4b-d0861435f647

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Exclusive write ownership:
- src/models/wsss_tissue.py
- src/models/deeplabv3plus.py

Defect Report from Checker Team C:
1. [Critical Memory Bottleneck] In `src/models/wsss_tissue.py` (lines 181-192):
   `self.relocalizer` is invoked AFTER bilinear interpolation to `(h_orig, w_orig)` instead of BEFORE.
   For 224x224, N=50,176 pixels, which creates a 50,176 x 50,176 attention affinity matrix allocating ~10 GB RAM and 47 seconds per patch!
   For 512x512, N=262,144 pixels, allocating 274 GB VRAM (guaranteed OOM).
   Fix Directive: Apply `self.relocalizer(cam, features)` at feature map token resolution (e.g. 7x7 or 14x14) BEFORE calling `F.interpolate` up to `(h_orig, w_orig)`.
   In mode "relocalization":
   `cam = self.compute_cam(features, weights)` (at feature resolution)
   `cam = self.relocalizer(cam, features)` (at feature resolution)
   `cam_up = F.interpolate(cam, size=(h_orig, w_orig), mode="bilinear", align_corners=False)`
   In mode "proposed":
   `cam = self.pda_generator(features, weights, ...)` (at feature resolution)
   `cam = self.relocalizer(cam, features)` (at feature resolution)
   `cam_up = F.interpolate(cam, size=(h_orig, w_orig), mode="bilinear", align_corners=False)`
2. [Minor Training Issue] In `src/models/deeplabv3plus.py` (`ASPPPooling`):
   `nn.BatchNorm2d(out_channels)` following `AdaptiveAvgPool2d((1, 1))` throws ValueError if batch size is 1 during `model.train()` because variance cannot be computed for a single 1x1 value.
   Fix Directive: In `ASPPPooling`, replace `nn.BatchNorm2d(out_channels)` with `nn.GroupNorm(num_groups=1, num_channels=out_channels)` (which computes layer-wise normalization across channels without requiring B > 1) or handle B=1 training gracefully.

Tasks:
1. Apply the relocalization fix in `src/models/wsss_tissue.py`.
2. Apply the ASPPPooling normalization fix in `src/models/deeplabv3plus.py`.
3. Run the test suite:
   cmd /c "C:\Python313\python.exe -m pytest tests/test_pda_backbone.py tests/test_benchmark_components.py tests/test_deeplabv3plus.py tests/test_adversarial_clinical_stress.py -v"
   Verify all tests pass with rapid execution (< 5 seconds)!
4. Run full test suite:
   cmd /c "C:\Python313\python.exe -m pytest tests/ -v"
5. Write your handoff report to:
   c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\worker_fix_clinical\handoff.md
6. Notify orchestrator (7ffe6ac5-9a5c-4c39-ba4b-d0861435f647).

Governance rules:
- Strictly follow Invariant 12: Standard keyboard hyphen ASCII 0x2D '-' only.
- Invariant 13: Windows OS cmd /c for shell execution.
