# Handoff Report: Deep Investigation and Refactoring Architecture for Colab WSSS Pipeline

- **Author:** Explorer Colab Agent (`explorer_colab_1`)
- **Recipient:** Lead Orchestrator (`cba3207a-b739-4afa-9ca5-4a3829438a4b`)
- **Target File:** `notebooks/SE374_Colab_WSSS_Pipeline.ipynb`
- **Timestamp:** 2026-10-19T19:23:00Z
- **Governance Standards:** Invariant 01 (Human confirmation), Invariant 07 (Zero Mockup), Invariant 08 (Patient anti-leakage), Invariant 12 (100% ASCII 0x2D hyphens), Invariant 13 (Windows `cmd /c`), Invariant 15 (Zero scratch scripts).

---

## 1. Observation

### 1.1 Notebook Cell Structure & Execution Flow Inventory
The target notebook `notebooks/SE374_Colab_WSSS_Pipeline.ipynb` consists of 19 cells (10 Markdown cells, 9 Code cells), conforming to `nbformat: 4`, `nbformat_minor: 0`:

| Cell Index | Cell Type | Header / Purpose | Key Statements / Lines | Observed Status |
| :---: | :---: | :--- | :--- | :--- |
| **0** | Markdown | Notebook Title & Overview | Lines 4-24: SE374.R11 WSSS Pipeline overview, 3-step execution guide | Valid overview, but references missing full-pipeline details |
| **1** | Markdown | Section 1 Header | Lines 26-30: `### 1. Kiểm tra cấu hình GPU & Môi trường` | Header only |
| **2** | Code | GPU & Environment Diagnostics | Lines 33-47: `!nvidia-smi`, `import torch`, print CUDA/VRAM | Functional |
| **3** | Markdown | Section 2 Header | Lines 49-54: `### 2. Cài đặt các thư viện cần thiết` | Header only |
| **4** | Code | Dependency Installation | Lines 56-64: `!pip install -q gdown opencv-python-headless matplotlib seaborn torchvision` | Missing `scipy` |
| **5** | Markdown | Section 3 Header | Lines 66-70: `### 3. Tự động tải Dữ Liệu Thật BCSS-WSSS bằng gdown` | Header only |
| **6** | Code | Dataset Download & Unzip | Lines 72-96: `!gdown --id 1sCK8Cqsf4yCHzzIx98vitocES2L_tkjn -O /content/BCSS-WSSS.zip`, `!unzip` | Correct Google Drive ID; directory search fallback needed |
| **7** | Markdown | Section 4 Header | Lines 98-103: `### 4. Thiết lập Chế độ Chạy: Mẫu Bệnh Nhân Thật (T4 Free) hoặc Toàn Bộ Dataset (L4 Pro)` | Header only |
| **8** | Code | Test Sample Selection | Lines 105-136: `MAX_TEST_SAMPLES = 150`, `test_img_dir.glob('*.png')`, naive slice `[:150]` | Hardcoded slice without patient-level metadata |
| **9** | Markdown | Section 5 Header | Lines 138-143: `### 5. Tải Trọng Số Checkpoint Đã Huấn Luyện SOTA (DeepLabV3+ Stage 2)` | Header only |
| **10** | Code | Checkpoint Download | Lines 145-161: `!gdown --id 1cXm29ko8UevbaeW6YreSg-vuNaQiN_je -O .../stage2_checkpoint_trained_on_bcss.pth` | Downloads weight file, but NEVER loads it into any PyTorch model |
| **11** | Markdown | Section 6 Header | Lines 163-168: `### 6. Cổng Lọc Mô Học Otsu Gating & Thiết Lập Bộ Nhớ GPU` | Header only |
| **12** | Code | Otsu Gating & Device Setup | Lines 170-194: `OtsuClinicalGating` class with HSV saturation Otsu; `device = torch.device(...)` | Naive Otsu (lacks grayscale intensity gate, lacks morphological cleanup); DeepLab architecture absent |
| **13** | Markdown | Section 7 Header | Lines 196-201: `### 7. Thực Thi Đánh Giá Định Lượng Trên Dữ Liệu Bệnh Nhân Thật` | Header only |
| **14** | Code | Quantitative Evaluation | Lines 203-267: Naive HSV color slicing (Lines 235-241); confusion matrix accumulation; IoU printout | **CRITICAL DEFECT:** Does NOT call DeepLabV3+ model; uses crude color heuristic |
| **15** | Markdown | Section 8 Header | Lines 269-274: `### 8. Trực Quan Hóa Lâm Sàng: Soi Ảnh Bệnh Nhân Thật Dưới Kính Hiển Vi Số` | Header only |
| **16** | Code | Clinical Visualization | Lines 276-322: Re-duplicates naive HSV color slicing; renders 3-column matplotlib figure | Displays naive heuristic output rather than deep model segmentations |
| **17** | Markdown | Section 9 Header | Lines 324-331: `### 9. Đóng Gói Toàn Bộ Kết Quả & Tự Động Tải Về Máy Tính` | Header only |
| **18** | Code | Artifact Export & Auto-Download | Lines 333-361: Saves PNG, writes `COLAB_EVALUATION_REPORT.md`, zips, triggers `files.download()` | Missing `COLAB_EVALUATION_REPORT.json`; missing Dice metrics |

---

### 1.2 Verbatim Code of the Naive Color Heuristic
In `notebooks/SE374_Colab_WSSS_Pipeline.ipynb`, lines 234-243:

```python
    start_t = time.time()
    hsv = cv2.cvtColor(raw_np, cv2.COLOR_RGB2HSV)
    pred = np.full(gt_mask.shape, 4, dtype=np.uint8)
    tissue_mask = hsv[:, :, 1] > 30
    pred[tissue_mask] = 1 # Mo dem
    dark_nuclei = (hsv[:, :, 2] < 140) & tissue_mask
    pred[dark_nuclei] = 0 # Khoi u (nhan dam mau)
    pred_final = OtsuClinicalGating.filter_background(raw_np, pred)
    total_time += (time.time() - start_t)
```

And duplicated in lines 298-303 for visualization:
```python
    hsv = cv2.cvtColor(raw_np, cv2.COLOR_RGB2HSV)
    pred = np.full(gt.shape, 4, dtype=np.uint8)
    pred[hsv[:, :, 1] > 30] = 1
    pred[(hsv[:, :, 2] < 140) & (hsv[:, :, 1] > 30)] = 0
    pred_final = OtsuClinicalGating.filter_background(raw_np, pred)
```

### 1.3 Verbatim Empirical Result of the Naive Heuristic
From previous execution recorded in `experiments/colab_runs/content/SE374_Colab_Results/COLAB_EVALUATION_REPORT.md`:

```markdown
# SE374.R11 Colab Real Experiment Results

- So luong mang anh da danh gia: 150
- mIoU Toan Cuc : 9.78%
- Toc Do Suy Luan: 929.26 FPS

### Chi tiet tung phan lop mo hoc:
- Khối U (TUM): IoU = 16.17%
- Mô Đệm (STR): IoU = 20.35%
- Lympho (LYM): IoU = 0.00%
- Hoại Tử (NEC): IoU = 0.00%
- Kính Nền (BACK): IoU = 12.37%
```

Observations from this run:
- **Global mIoU:** 9.78% (unacceptable, fails clinical validity).
- **Lymphocyte (LYM - Class 2) IoU:** 0.00% (never predicted).
- **Necrosis (NEC - Class 3) IoU:** 0.00% (never predicted).
- **Inference Speed:** 929.26 FPS (unrealistic because it did zero GPU neural network tensor computations, only simple numpy array indexing).
- **Root cause:** The deep neural network `stage2_checkpoint_trained_on_bcss.pth` was downloaded in Cell 10, but completely ignored in Cell 14 and Cell 16!

---

### 1.4 Architecture Mapping from SOTA Repositories
Direct inspection of official repository files:
1. `repos/WSSS-Tissue/3_train_stage2.py` (lines 8, 25-29, 61-70, 139-145, 189-200):
   - Backbone: `resnet` -> `resnet.ResNet101(output_stride=16, BatchNorm=nn.BatchNorm2d)`
   - ASPP: `build_aspp('resnet', output_stride=16, BatchNorm=nn.BatchNorm2d)` with dilations `[1, 6, 12, 18]` and inplanes=2048.
   - Decoder: `build_decoder(num_classes=4, 'resnet', BatchNorm=nn.BatchNorm2d)` with low-level inplanes=256, projected to 48, fused to 304, refined via two 3x3 convs (256 channels), and final 1x1 conv `last_conv.8` to `num_classes`.
   - Checkpoint state dictionary key hierarchy:
     - `backbone.conv1.weight`, `backbone.bn1.*`
     - `backbone.layer1.*`, `backbone.layer2.*`, `backbone.layer3.*`, `backbone.layer4.*`
     - `aspp.aspp1.atrous_conv.weight`, `aspp.aspp1.bn.*`
     - `aspp.aspp2.*`, `aspp.aspp3.*`, `aspp.aspp4.*`
     - `aspp.global_avg_pool.1.weight`, `aspp.global_avg_pool.2.*`
     - `aspp.conv1.weight`, `aspp.bn1.*`
     - `decoder.conv1.weight`, `decoder.bn1.*`
     - `decoder.last_conv.0.weight`, `decoder.last_conv.1.*`
     - `decoder.last_conv.4.weight`, `decoder.last_conv.5.*`
     - `decoder.last_conv.8.weight`, `decoder.last_conv.8.bias`
2. `repos/LowMagWSS/SegMethod/WSSS-Tissue/visualize_predictions.py` (lines 186-212):
   - Model loading function `load_state_dict_robust(model, ckpt_path, device)` removes `module.` prefix.
   - Default classes: 4 tissue classes (`TUM`, `STR`, `LYM`, `NEC`). Background is mapped via Otsu gating or class 4 index.
3. Divergence in `src/models/deeplabv3plus.py`:
   - `src/models/deeplabv3plus.py` uses custom sub-modules (`self.backbone.stem`, `self.backbone.stage1`, `self.aspp.branch1`, `self.decoder.project_low`, `self.decoder.classifier`).
   - Consequently, loading `stage2_checkpoint_trained_on_bcss.pth` directly into `src/models/deeplabv3plus.py` causes key mismatch errors.
   - **Resolution:** The Colab notebook MUST implement the exact `DeepLab` architecture matching the checkpoint tensor keys directly inside Cell 12.

---

## 2. Logic Chain

1. **Premise 1:** The user request (2026-10-19T19:11:48Z) explicitly commands:
   - "Implement the exact PyTorch DeepLabV3+ architecture with Atrous Spatial Pyramid Pooling (ASPP) and ResNet backbone matching the tensor shapes in `stage2_checkpoint_trained_on_bcss.pth`."
   - "Eliminate the naive HSV color thresholding in inference; forward pass must execute directly through the deep neural network on GPU."
   - "Evaluate on 100% real TCGA-BRCA patient data with Otsu clinical gating, achieve clinical-grade mIoU, and deliver an end-to-end automated download pipeline."
2. **Premise 2:** In the existing notebook, Cell 14 and Cell 16 bypass model inference entirely and execute naive HSV color thresholding, resulting in 9.78% mIoU, 0.00% for LYM and NEC, and 929 FPS artificial latency.
3. **Premise 3:** To run on Google Colab without external repository dependency or custom module installation issues, the notebook must be self-contained: Cell 12 must define the exact `DeepLab` architecture (ResNet101 backbone + ASPP + Decoder) and the medical-grade `OtsuClinicalGating` class.
4. **Premise 4:** In `stage2_checkpoint_trained_on_bcss.pth`, the classification head (`decoder.last_conv.8.weight`) outputs logits for tissue classes (4 classes: Tumor, Stroma, Lymphocyte, Necrosis). At test time, argmax yields tissue class indices 0..3, and Otsu clinical gating identifies empty glass pixels in HSV space and forces them to Class 4 (Background). If the checkpoint contains 5 classes, Otsu gating acts as an overlay filter.
5. **Premise 5:** To satisfy Requirement R2 (Dual-Mode Execution):
   - The notebook should feature an explicit configuration flag: `SANITY_MODE = True` (or `MODE = "sanity"`).
   - In Sanity Mode, it samples 150 real test patches from real TCGA-BRCA patients (e.g. `TCGA-AQ-A04H`, `TCGA-C8-A1HJ`, `TCGA-AR-A0TQ`).
   - In Full Mode, it evaluates all 4,986 test patches.
6. **Premise 6:** To satisfy Requirement R3 (Quantitative Metrics & Confusion Matrix):
   - Metric calculation must compute the full 5x5 confusion matrix over valid pixels `(gt >= 0) & (gt < 5)` using vectorized `np.bincount`.
   - Calculate per-class IoU, global mIoU, per-class Dice, global mDice, and inference throughput (FPS and ms/patch).
7. **Premise 7:** To satisfy Requirement R4 (Visualization & Closed-Loop Packaging):
   - Renders 300 DPI side-by-side comparison: Patient H&E vs Ground Truth vs DeepLabV3+ AI prediction.
   - Saves both `COLAB_EVALUATION_REPORT.md` and `COLAB_EVALUATION_REPORT.json`.
   - Packs all outputs into `/content/SE374_Colab_Results.zip` and triggers automatic browser download via `google.colab.files.download()`.

---

## 3. Caveats

1. **Network Connectivity in Colab:**
   Downloading `BCSS-WSSS.zip` (~1.2 GB) and `stage2_checkpoint_trained_on_bcss.pth` (~240 MB) via `gdown` requires active internet access within Colab. Google Drive rate limits may occasionally require retry flags (`gdown --id ... --fuzzy`).
2. **Number of Output Channels in Checkpoint:**
   In Han et al. (MedIA 2022), Stage 2 was trained with `n_class = 4` (Tumor, Stroma, Lymphocyte, Necrosis) with background treated by exclusion. To ensure total robustness, the model instantiation in Cell 12 must dynamically inspect `ckpt['state_dict']['decoder.last_conv.8.weight'].shape[0]` (or `ckpt['decoder.last_conv.8.weight'].shape[0]`) before building the model, seamlessly supporting either 4 or 5 classes.
3. **Inference Latency on Colab T4 vs L4:**
   On a free T4 GPU (16 GB VRAM), processing 150 patches with DeepLabV3+ ResNet-101 (batch size 8 or 16) takes ~3-8 seconds (~20-40 FPS). Evaluating all 4,986 patches takes ~2-3 minutes. This fits comfortably within Colab compute budgets.

---

## 4. Conclusion & Cell-by-Cell Refactoring Recommendations

The notebook `notebooks/SE374_Colab_WSSS_Pipeline.ipynb` requires a comprehensive, cell-by-cell refactoring as follows:

### Cell-by-Cell Refactoring Plan:

- **Cell 0 [Markdown]:**
  - Update title to reflect full DeepLabV3+ deep neural network evaluation on real TCGA-BRCA data.
  - Document the two operational modes: T4 Free Sanity Mode (150 real patches, ~1-2 min) and L4 Pro Full Mode (4,986 real patches).
  - Clarify the Zero Mockup, real patient data, and Otsu clinical gating guarantees.
  - Strictly maintain ASCII 0x2D hyphens.

- **Cell 1 [Markdown] & Cell 2 [Code]: Environment & GPU Verification**
  - Inspect GPU model (`torch.cuda.get_device_name(0)`), VRAM size, CUDA version, PyTorch version.
  - Set deterministic seeds: `torch.manual_seed(42)`, `np.random.seed(42)`.

- **Cell 3 [Markdown] & Cell 4 [Code]: Dependencies**
  - Install required packages:
    `!pip install -q gdown opencv-python-headless matplotlib seaborn torchvision scipy`

- **Cell 5 [Markdown] & Cell 6 [Code]: Real Dataset Download (BCSS-WSSS)**
  - Automated download using Google Drive ID `1sCK8Cqsf4yCHzzIx98vitocES2L_tkjn`.
  - Extract to `/content/` and resolve dataset paths dynamically (`DATA_ROOT / 'test' / 'img'`).
  - Report total real test patches found (expected: 4,986).

- **Cell 7 [Markdown] & Cell 8 [Code]: Execution Mode Configuration (Sanity vs Full)**
  - Add explicit parameter:
    ```python
    SANITY_MODE = True  # Set to False to evaluate all 4,986 test patches
    MAX_SANITY_SAMPLES = 150
    ```
  - In Sanity Mode, stratify or select patches across diverse TCGA patients (e.g., `TCGA-AQ-A04H`, `TCGA-C8-A1HJ`, `TCGA-AR-A0TQ`).
  - Print patient IDs and sample counts.

- **Cell 9 [Markdown] & Cell 10 [Code]: SOTA Checkpoint Ingestion**
  - Automated download using Google Drive ID `1cXm29ko8UevbaeW6YreSg-vuNaQiN_je` to `/content/checkpoints/stage2_checkpoint_trained_on_bcss.pth`.
  - Validate file size and existence.

- **Cell 11 [Markdown] & Cell 12 [Code]: Exact DeepLabV3+ Architecture & Medical Otsu Gating**
  - Implement full PyTorch `DeepLab` matching `stage2_checkpoint_trained_on_bcss.pth`:
    - `Bottleneck` (expansion=4, dilation handling).
    - `ResNet` (ResNet101 with layers `[3, 4, 23, 3]`, output_stride=16, Multi-Grid dilations `[1, 2, 4]` on layer 4).
    - `ASPP` (inplanes=2048, dilations `[1, 6, 12, 18]`, global image average pooling, projection to 256).
    - `Decoder` (low_level_inplanes=256, 1x1 conv to 48, last_conv: 304 -> 256 -> 256 -> `num_classes`).
  - Implement robust state dict loading:
    - Automatically strip `module.` prefix.
    - Inspect `decoder.last_conv.8.weight.shape[0]` to set `num_classes` (4 or 5).
    - Load state dict and set `model.eval()`.
  - Implement medical-grade `OtsuClinicalGating`:
    - HSV saturation channel thresholding with $0.8 \times T_{otsu}$.
    - Grayscale intensity gating ($I_{gray} < 235$).
    - Morphological cleanup ($3 \times 3$ ellipse kernel OPEN and CLOSE).
    - Hard gating: force background/glass pixels to Class 4.

- **Cell 13 [Markdown] & Cell 14 [Code]: Real Deep Neural Network Evaluation**
  - **REPLACE NAIVE COLOR HEURISTIC WITH TRUE NEURAL NETWORK FORWARD PASS:**
    ```python
    with torch.no_grad():
        inputs = batch_tensors.to(device)
        logits = model(inputs)
        # If model outputs 4 classes:
        pred_tissue = logits.argmax(dim=1).cpu().numpy()
        # Apply Otsu clinical gating to set glass to Class 4
        pred_final = apply_otsu_gating(raw_images, pred_tissue)
    ```
  - Accumulate 5x5 confusion matrix via `np.bincount` across all valid pixels `(gt >= 0) & (gt < 5)`.
  - Compute per-class IoU, global mIoU, per-class Dice, global mDice, and FPS.
  - Print formatted ASCII tables for console output.

- **Cell 15 [Markdown] & Cell 16 [Code]: High-Fidelity Clinical Visualizations**
  - Select 3-4 diverse patient patches displaying distinct histopathological patterns (Tumor, Stroma, Lymphocyte, Necrosis, and Glass).
  - Perform true DeepLabV3+ inference + Otsu clinical gating.
  - Render publication-grade 300 DPI 3-column figure:
    - Column 1: Patient H&E Image (with Patient ID title)
    - Column 2: Pathologist Ground Truth (colorized)
    - Column 3: WSSS AI DeepLabV3+ Prediction (colorized)
  - Color palette: TUM (Red), STR (Green), LYM (Blue), NEC (Orange), BACK (Off-white).

- **Cell 17 [Markdown] & Cell 18 [Code]: Closed-Loop Packaging & Auto-Download**
  - Save `clinical_segmentation_comparison.png` at 300 DPI.
  - Export `COLAB_EVALUATION_REPORT.md` with full markdown formatting and confusion matrix table.
  - Export `COLAB_EVALUATION_REPORT.json` containing structured machine-readable metrics, per-class scores, confusion matrix, timestamps, and hardware details.
  - Compress `/content/SE374_Colab_Results.zip`.
  - Trigger automatic browser download: `files.download('/content/SE374_Colab_Results.zip')`.

---

## 5. Verification Method

To verify the refactored notebook prior to deployment and execution:

1. **Jupyter Notebook Structure Verification:**
   Ensure the notebook parses as valid JSON and conforms to `nbformat == 4`:
   ```bash
   cmd /c "python -c \"import json; nb = json.load(open('notebooks/SE374_Colab_WSSS_Pipeline.ipynb', encoding='utf-8')); assert nb['nbformat'] == 4; print('Valid nbformat 4:', len(nb['cells']), 'cells')\""
   ```

2. **Code Cell Syntax Verification:**
   Compile every code cell in the notebook to ensure zero syntax errors:
   ```bash
   cmd /c "python -c \"import json, ast; nb = json.load(open('notebooks/SE374_Colab_WSSS_Pipeline.ipynb', encoding='utf-8')); [ast.parse(''.join([l for l in c['source'] if not l.strip().startswith('!')])) for c in nb['cells'] if c['cell_type'] == 'code']; print('All code cells passed syntax compilation cleanly!')\""
   ```

3. **Hyphen Standard Invariant 12 Verification:**
   Ensure zero non-ASCII em-dashes (`\u2014`), en-dashes (`\u2013`), horizontal bars (`\u2015`), or minus signs (`\u2212`):
   ```bash
   cmd /c "git grep -P \"[\x{2013}\x{2014}\x{2015}\x{2212}]\" -- notebooks/SE374_Colab_WSSS_Pipeline.ipynb || echo 100% ASCII 0x2D hyphens confirmed"
   ```

4. **Zero Naive Heuristic Quarantine Verification:**
   Ensure the naive color thresholding code (`tissue_mask = hsv[:, :, 1] > 30` and `dark_nuclei = (hsv[:, :, 2] < 140)`) is completely absent:
   ```bash
   cmd /c "git grep -n \"dark_nuclei\" -- notebooks/SE374_Colab_WSSS_Pipeline.ipynb || echo Clean: Naive color heuristic eliminated"
   ```

5. **DeepLab Model Instantiation Verification:**
   Verify that the model class defined in Cell 12 instantiates cleanly on a dummy input tensor `(1, 3, 224, 224)` and returns shape `(1, 4, 224, 224)` or `(1, 5, 224, 224)`.
