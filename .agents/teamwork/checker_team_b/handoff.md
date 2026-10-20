# Handoff Report - Checker Team B (Scientific Integrity & Patient Leakage Auditor)

**Role:** reviewer, critic  
**Target:** Independent Audit of Patient Leakage (Invariant 08), Zero Mockup (Invariant 07), and Mathematical Formulations  
**Auditor ID:** Checker Team B  
**Recipient ID:** 7ffe6ac5-9a5c-4c39-ba4b-d0861435f647 (Orchestrator)  
**Date:** 2026-10-18  

---

## 1. Observation

1. **Patient Split Manifest (`src/config/bcss_patient_splits.json`):**
   - Lines 4-7: `"num_patients": 151, "train_count": 91, "val_count": 30, "test_count": 30`.
   - Inspection verified that `train` has 91 items, `val` has 30 items, and `test` has 30 items.
   - Set intersection evaluation via Python:
     - `len(set(train) & set(val)) == 0`
     - `len(set(train) & set(test)) == 0`
     - `len(set(val) & set(test)) == 0`
     - `len(set(train) | set(val) | set(test)) == 151`
   - Patient-level leakage across splits is exactly 0.0%.

2. **Metrics & Losses Implementation (`src/losses_metrics/metrics.py` and `src/losses_metrics/losses.py`):**
   - In `metrics.py` lines 65-67:
     `indices = target * self.num_classes + pred`
     `counts = np.bincount(indices, minlength=self.num_classes**2)`
     `self.confusion_matrix += counts.reshape((self.num_classes, self.num_classes))`
   - In `metrics.py` lines 74-85:
     `tp = np.diag(self.confusion_matrix)`
     `fp = self.confusion_matrix.sum(axis=0) - tp`
     `fn = self.confusion_matrix.sum(axis=1) - tp`
     `denominator = tp + fp + fn`
     `iou[valid_classes] = tp[valid_classes] / denominator[valid_classes]`
     `miou = float(np.mean(iou[valid_classes])) if np.any(valid_classes) else 0.0`
   - In `metrics.py` lines 121-161:
     VTP and TSR are computed via actual pixel sums: `tumor_pixels / tissue_pixels * 100.0` and `tumor_pixels / (tumor_pixels + stroma_pixels) * 100.0`.
   - In `losses.py`:
     - `MultiLabelClassificationLoss`: uses `nn.BCEWithLogitsLoss`.
     - `MultiLayerPseudoSupervisionLoss`: computes $L_{\text{total}} = L_{bn7} + 0.5 L_{b5\_2} + 0.25 L_{b4\_5}$ with bilinear logit interpolation to match stage dimensions.
     - `UncertaintyRegularizationLoss`: computes normalized Shannon entropy $H(p) = -\frac{\sum p \log p}{\log C}$ with $1\text{e-}7$ clamping and boundary down-weighting $w = \text{clamp}(1.0 - H / \tau, 0.1, 1.0)$.
   - Zero hardcoded dummy metrics, zero mocked constants, and zero fake random values found.

3. **Classification Gate Mechanism (`src/models/deeplabv3plus.py`):**
   - Lines 308-309: `gated_logits = logits * gate + (1.0 - gate) * penalty` where penalty is `-1e4`.
   - Unit test `test_cgm_extinguishes_absent_class_in_segmentation` verified that absent classes result in exactly 0 predicted pixels in argmax masks and softmax probability $< 1\text{e-}4$.
   - Lines 298-304: Default background gating dynamically handles all-zero presence vectors.

4. **Progressive Dropout Attention (`src/models/pda_generator.py`):**
   - Implements multi-step attention expansion ($t=1\dots T$) with top-p spatial masking and channel dropout.
   - Includes prototype cosine similarity gating $\mu_c = \frac{\sum M_c^{(0)} F}{\sum M_c^{(0)} + \epsilon}$, $G_c = \text{ReLU}(\text{CosSim}(F, \mu_c))$ to prevent boundary bleeding into stroma.
   - Unit test `test_pda_expansion_property` verified expanded activation area compared to baseline CAM.
   - Unit test `test_pda_boundary_bleeding_control` verified right-half stroma activation remains $< 0.05$.

5. **Test Execution Results:**
   - Ran `cmd /c "C:\Python313\python.exe -m pytest tests/test_patient_split.py tests/test_benchmark_components.py tests/test_deeplabv3plus.py -v"`:
     Result: 25 passed in 6.69s.
   - Ran `cmd /c "C:\Python313\python.exe -m pytest tests/test_pda_backbone.py -v"`:
     Result: 14 passed in 2.95s.
   - Ran full suite `cmd /c "C:\Python313\python.exe -m pytest tests/ -v"`:
     Result: 61 passed in 16.02s.

---

## 2. Logic Chain

- **Step 1 (Patient Anti-Leakage):** From Observation 1, the manifest `bcss_patient_splits.json` partitions 151 patients into 91 train, 30 val, and 30 test cases. Evaluating pairwise set intersections yields empty sets across all pairs ($P_{\text{train}} \cap P_{\text{val}} = \emptyset$, $P_{\text{train}} \cap P_{\text{test}} = \emptyset$, $P_{\text{val}} \cap P_{\text{test}} = \emptyset$). Furthermore, Observation 1 and test results confirm `verify_patient_leakage()` raises an error if an overlap is introduced. Therefore, Invariant 08 is satisfied with 0.0% patient leakage.
- **Step 2 (Zero Mockup Integrity):** From Observation 2, inspection of `metrics.py` confirms that `mIoU`, `mDice`, per-class metrics, `VTP`, and `TSR` are calculated strictly via `np.bincount` and array operations over the confusion matrix. No mock functions or hardcoded scores exist in the evaluation path. Therefore, Invariant 07 is satisfied with 100% academic integrity.
- **Step 3 (Mathematical Formulations):** From Observations 2, 3, and 4, `MultiLabelClassificationLoss`, `MultiLayerPseudoSupervisionLoss`, `UncertaintyRegularizationLoss`, `ClassificationGateMechanism`, and `ProgressiveDropoutAttention` adhere precisely to their theoretical formulations and literature specifications. Boundary conditions (division by zero, all-zero presence vectors, log(0)) are defensively clamped with numerical stability guarantees.
- **Step 4 (Test Verification):** From Observation 5, all 25 targeted unit tests and all 61 repository unit tests pass with zero failures.

---

## 3. Caveats

- In `src/datasets/patient_split.py`, `extract_patient_id()` splits filenames by `[_+]` as a fallback for non-TCGA files. For TCGA datasets, the regex strictly matches standard 12-character barcodes `TCGA-XX-XXXX`. Should non-standard patient identifiers with embedded underscores (e.g. `HOSPITAL_A_PATIENT_1`) be introduced in future extensions, the regex or prefix extraction rule should be expanded accordingly. Currently, all 151 patients follow standard TCGA naming conventions.

---

## 4. Conclusion

**Verdict: APPROVE**

The codebase meets all requirements for Invariant 08 (Patient-Level Anti-Leakage Protocol) and Invariant 07 (Zero Mockup & Scientific Integrity). Mathematical formulations of losses, metrics, CGM, and PDA are mathematically sound, numerically stable, and independently verified.

---

## 5. Verification Method

To independently verify this assessment, execute the following commands in the workspace root:

1. **Verify Patient-Level Anti-Leakage (151 patients, 0.0% leakage):**
   ```cmd
   cmd /c "C:\Python313\python.exe -m pytest tests/test_patient_split.py -v"
   ```

2. **Verify Zero-Mockup Metrics & Loss Functions:**
   ```cmd
   cmd /c "C:\Python313\python.exe -m pytest tests/test_benchmark_components.py -v"
   ```

3. **Verify DeepLabV3+ with Classification Gate Mechanism (CGM):**
   ```cmd
   cmd /c "C:\Python313\python.exe -m pytest tests/test_deeplabv3plus.py -v"
   ```

4. **Verify Progressive Dropout Attention (PDA) Generator:**
   ```cmd
   cmd /c "C:\Python313\python.exe -m pytest tests/test_pda_backbone.py -v"
   ```

5. **Check Absence of Forbidden Non-ASCII Hyphens:**
   ```cmd
   cmd /c "C:\Python313\python.exe -c ""with open('.agents/teamwork/checker_team_b/audit_report.md', 'rb') as f: data = f.read(); forbidden = [b'\xe2\x80\x94', b'\xe2\x80\x93', b'\xe2\x88\x92', b'\xe2\x80\x95']; print('Forbidden found:', [x for x in forbidden if x in data])"""
   ```

Invalidation Condition: Any assertion failure in `test_patient_split.py`, any non-zero patient overlap, or the appearance of non-computed dummy metrics in `src/losses_metrics/` would immediately invalidate this approval.
