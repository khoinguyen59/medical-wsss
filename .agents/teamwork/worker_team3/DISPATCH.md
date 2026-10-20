## 2026-10-18T16:12:06Z
You are Worker Team 3 (Segmentation Network & Losses/Metrics) for the SE374.R11 WSSS project.

Working directory: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\worker_team3
Project root: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA
Authoritative Request: c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\ORIGINAL_REQUEST.md
Orchestrator ID: 7ffe6ac5-9a5c-4c39-ba4b-d0861435f647

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Exclusive write ownership:
- src/models/deeplabv3plus.py
- tests/test_deeplabv3plus.py

Tasks:
1. Create `src/models/deeplabv3plus.py`:
   - Implement `DeepLabV3PlusWithCGM` architecture:
     * Atrous Spatial Pyramid Pooling (ASPP) module with dilation rates [1, 6, 12, 18], batch normalization, and image pooling.
     * Decoder module fusing low-level feature maps and upsampled ASPP features.
     * Classification Gate Mechanism (CGM):
       Accepts image-level class presence probabilities or multi-hot labels $y \in \{0, 1\}^C$.
       For any class $c$ where $y_c = 0$, CGM gates the corresponding segmentation logits:
       $\hat{Z}_c = Z_c \cdot y_c + (1 - y_c) \cdot (-\infty \text{ or large negative penalty } -1e4)$.
       This guarantees zero false-positive segmentations for absent tissue types.
     * Forward method supporting training with pseudo-masks (CrossEntropyLoss, MLPS, UncertaintyRegularizationLoss) and inference with softmax / argmax segmentation masks.
2. Create `tests/test_deeplabv3plus.py`:
   - Test forward pass with input tensor `(B, 3, H, W)`.
   - Test ASPP module output shapes.
   - Test Classification Gate Mechanism (CGM): verify that when class $c$ is marked absent in $y$, the predicted segmentation mask contains zero pixels of class $c$.
   - Test end-to-end loss computation with `MultiLayerPseudoSupervisionLoss` and `UncertaintyRegularizationLoss`.
3. Run tests via:
   cmd /c "C:\Python313\python.exe -m pytest tests/test_deeplabv3plus.py -v"
4. Document all commands and results in `c:\Users\Nguyen Trong Khoi\Downloads\Y_KHOA\.agents\teamwork\worker_team3\handoff.md`.
5. Send completion message to orchestrator (7ffe6ac5-9a5c-4c39-ba4b-d0861435f647).

Governance rules:
- Strictly follow Invariant 12: Standard keyboard hyphen ASCII 0x2D '-' only. Absolutely NO em-dash or en-dash.
- Strictly follow Invariant 13: Windows OS cmd /c for all shell execution.
- Strictly follow Invariant 15: Zero one-off scratch scripts.
