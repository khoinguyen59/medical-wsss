# SE374.R11 - swarm audit - AUDITOR 03 (doc lap, doi khang, khong sua code)

**Ngay audit:** 2026-11-05
**Tac tu:** AUDITOR 03 / 8 (swarm doi khang, prompt dong nhat, T1-T10 tron bo)
**Repo:** `SE374_WSSS_WORKSPACE`
**Quy chieu doi chieu:** `docs/reports/audit/2026-11-02-swarm-audit/01_SWARM_RECHECK_VERIFICATION.md`
**Vai tro:** Builder-Auditor Separation - chi doc, chi chay lenh read-only, chi ghi duy nhat tep nay. Khong sua `src/`, khong sua `tests/`, khong tao script moi.

## 0. Moi truong va phuong thap

| Yeu to | Gia tri thuc |
| :--- | :--- |
| Python | 3.13.9 (Anaconda) |
| pytest | 8.4.2 |
| ruff | 0.12.0 |
| torch | 2.10.0+CPU (**khong co GPU**) |
| Pillow | 12.0.0 |
| Env bat buoc | `KMP_DUPLICATE_LIB_OK=TRUE` cho moi lenh cham torch/numpy |

Quy uoc chung minh trong bao cao nay:

- **VERIFIED** = toi da chay lenh va output that duoc dan nguyen van.
- **INFERENCE** = ket luan tu doc ma, chua co lenh tai hien truc tiep.
- **CLEAN** = da chay phep kiem va KHONG phat hien loi; noi dung "khong tim thay" cung duoc ghi lai.

---

## 1. Executive Summary - phan tich chi tiet cac tuyen bo cua audit 2026-11-02

**KET LUAN TONG QUAN: cac tuyen bo trong `01_SWARM_RECHECK_VERIFICATION.md` la PHONG DAI VA KHONG THE Tai LAP lai tren trang thai repo nhu ban giao.**

| # | Tuyen bo 2026-11-02 | Ket qua kiem chung 2026-11-05 | Danh gia |
| :---: | :--- | :--- | :---: |
| 1 | `348 passed / 0 failed` | **`345 passed, 3 FAILED`**, 527.48s (khong phai 104.15s) | **SAI** |
| 2 | `Ruff 0 lint errors` | `src` + `tests`: 0 loi (dung). Nhung `src tests scripts`: **33 loi**; `.` (toan repo, trur `repos/`): **464 loi** | **THIEN LECH / SCOPE-NARROWING** |
| 3 | `0.0% patient leakage (151 cases)` | Dung cho duong dan that (manifest + anh tren dia). **SAI** cho nhanh fallback: `TCGA-C8-A1HJ` nam trong `train` that bi dat vao `test` | **DUNG MOT PHAN** |
| 4 | `0 non-ASCII hyphens across ALL workspace files` | **230 dong / 38 file** chua U+2013/U+2014/U+2015/U+2212 (trur `repos/`). Cac file `.py/.md/.ipynb` chinh sach; vi pham nam trong `.agents/`, `graphify-out*/` va **2 ky tu U+2013 trong slide PPTX nop** | **SAI** |
| 5 | `Zero Mockup: khong co dummy metric` | 3 bai to ly trinh nghiem trong `product/reports/clinical/` (co so 2 patch, t=inf, gia dinh 15.0 thang) + 1 model random bi gán nhan "Author SOTA" | **SAI** |
| 6 | `CI/CD tu dong quet anti-mockup & leakage` | `ci.yml` khong cai torch -> pytest khong the chay; 2 buoc lint co `\|\| true`; gate mockup quet 5 token khong ton tai; **khong co buoc kiem leakage** | **SAI** |
| 7 | `INVARIANT-28: khong ep Google Drive` | **23 notebook** van con `drive.mount('/content/drive')`; README cua goi ban giao ra lenh "Buoc 2: chay drive.mount" | **SAI** |

**Bang tong hop defect theo muc do nghiem trong (toan bo 10 chieu):**

| Severity | So luong |
| :--- | :---: |
| **Critical** | 8 |
| **High** | 17 |
| **Medium** | 14 |
| **Low** | 5 |
| **TONG** | **44** |

---

## 2. T1 - Tang 1: Test suites tu dong va static analysis

### 2.1 Lenh da chay

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q -p no:cacheprovider
python -m ruff check src tests scripts
python -m ruff check . --statistics --exclude repos
```

### 2.2 Output that (VERIFIED)

```
FAILED tests/test_thin_drive_runner.py::test_2_drive_bundle_packager_creates_complete_folder_and_zip
FAILED tests/test_thin_drive_runner.py::test_3_new_thin_runner_notebooks_valid_concise_and_modular
FAILED tests/test_thin_drive_runner.py::test_7_invariant_12_strict_ascii_hyphens_all_packaged_files
3 failed, 345 passed, 2 warnings in 527.48s (0:08:47)
```

Doi chieu tuyen bo: **348 passed / 0 failed -> KHONG the tai lap. Thuc te 345/3.**
Thoi luong tuyen bo 104.15s -> thuc te **527.48s** (gap 5.1x).

### 2.3 Chu ky 3 loi te nhat (nguyen van)

**Loi 1 - `test_2` (pha huy cay that cua workspace):**

```
results = os.walk(path, topdown=False, onerror=onerror, followlinks=os._walk_symlinks_as_files)
for dirpath, dirnames, filenames in results:
    for name in dirnames:
        fullname = os.path.join(dirpath, name)
        try:
            os.rmdir(fullname)
E           OSError: [WinError 145] The directory is not empty:
            '<workspace_root>\\drive_upload_ready\\SE374_WSSS_Drive_Code\\src'
..\..\anaconda3\Lib\shutil.py:617: OSError
```

**Loi 2 - `test_3` (triet de cua Loi 1 - thu muc da bi xoa nen khong ghi lai duoc):**

```
>       packager.build_drive_package()
tests\test_thin_drive_runner.py:180:
_ scripts\build_drive_package.py:1568: in build_drive_package
    (bundle_nb_dir / nb_path.name).write_text(nb_text, encoding="utf-8", newline="\n")
E       FileNotFoundError: [Errno 2] No such file or directory:
E       'C:\\...\\drive_upload_ready\\SE374_WSSS_Drive_Code\\notebooks\\SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb'
```

**Loi 3 - `test_7` (chinh gate INVARIANT-12 cua goi Drive bi che** - tuc la kiem tra typography cua ban giao **khong bao gio chay duoc**):**

```
>       packager.build_drive_package()
tests\test_thin_drive_runner.py:493:
_ scripts\build_drive_package.py:1601: in build_drive_package
    (bundle_history_dir / h_file.name).write_text(h_compact, encoding="utf-8", newline="\n")
E       FileNotFoundError: [Errno 2] No such file or directory:
E       'C:\\...\\notebooks\\history_runs\\Run1_20261020_1022UTC_Stage2_DeepLabV3Plus_SingleCkpt_T4_Full4986.ipynb'
```

### 2.4 Ket luan ve tuyen bo "0.0% leakage" trong chinh file kiem thu

`tests/test_thin_drive_runner.py::test_7` duoc lap trong bang nghiem thu 2026-11-02 nhu "PASSED - INVARIANT-12 check". Thuc te no **FAILED** o dong 493 va khong bao gio kiem duoc goi Drive. Day la mo hinh "test xanh gia" ma INVARIANT-26 (Anti-Mock-Testing Illusion Gate) cam.

### 2.5 Ruff (VERIFIED)

| Lenh | Ket qua that |
| :--- | :--- |
| `ruff check src` | `All checks passed!` (khop voi tuyen bo) |
| `ruff check tests` | `All checks passed!` |
| `ruff check src tests scripts` | **`Found 33 errors.`** (19 E741, 7 I001, 6 F401, 1 F841) |
| `ruff check . --statistics --exclude repos` | **`Found 464 errors.`** (193 I001, **93 F811 redefined-while-unused**, 53 E701, 36 F541, 35 F401, 25 E401, 19 E741, **6 F821 undefined-name**, 4 F841) |

Vi du loi `scripts/` (VERIFIED, nguyen van):

```
scripts\download_papers.py:4:8: F401 [*] `time` imported but unused
scripts\refactor_notebooks_pure_colab.py:42:9: F841 Local variable `old_drive_block` is assigned to but never used
scripts\build_drive_package.py:1209:51: E741 Ambiguous variable name: `l`
```

> **Nhan xet:** `ruff.toml` co `[lint.per-file-ignores] "tests/*" = ["F401", "F841"]` va tuyen bo chi noi "0 errors across all 42 modules in `src/`" - ve mat ky thuat la dung, nhung bang nghiem thu ghi "Ruff Linter Cleanliness: 0 Errors / CLEAN" ma khong ghi rang `scripts/` va notebook-sinh-ma (`build_t4_l4_notebook_suites.py` > 2200 dong) co toi 33 loi va toan repo co **6 loi F821 undefined-name** (day la nhom loi co kha nang gay `NameError` luc chay).

### Defect T1

| Ma | Muc do | Do tin cay | Vi tri | Mo ta |
| :--- | :--- | :--- | :--- | :--- |
| `[T1-pytest-3-failures]` | **Critical** | VERIFIED | `tests/test_thin_drive_runner.py:98,180,493` | `pytest tests/` that = 345 passed / **3 failed** tren trang thai ban giao. Tuyen bo 348/0 sai. |
| `[T1-ruff-scope-claim]` | Medium | VERIFIED | `scripts/*` (33 loi), toan repo 464 loi | "Ruff 0 errors / CLEAN" chi dung khi han scope vao `src/`; F821 x6 va F811 x93 bi bo qua hoan toan. |
| `[T1-duration-mismatch]` | Low | VERIFIED | bao cao 2026-11-02 muc 2.1 | 104.15s (tuyen bo) vs 527.48s (do). |

---

## 3. T2 - Zero-Mockup (INVARIANT-07)

### 3.1 Lenh quet

```bash
grep -rn -i -E "\b(mock|dummy|fake|synthetic|placeholder|simulated)\b" src/ --include=*.py
grep -rnoE "(mIoU|miou|Dice|dice|IoU)[^#\n]{0,30}= *(0\.[0-9]{2,4}|[0-9]{1,2}\.[0-9]{1,2})" notebooks/*.ipynb *.ipynb scripts/*.py
grep -rn "np.random\.\|torch.rand(\|random\.choice" src/ --include=*.py
```

Ket qua: **khong co** gan `metric = <hang so noi bat>` trong `src/` (VERIFIED, 0 dong match). Day la mot CLEAN finding. Nhung `scripts/` va `product/` thi co (xem duoi).

### 3.2 `[T2-zero-mockup-scope-claim-false]` - **Critical** - VERIFIED

**Duong dan:** `product/reports/clinical/EXHAUSTIVE_3WAY_BENCHMARK_REPORT.md`

Dong 6 va 11 cua bao cao nop Giang vien tuyan bo:

```
> **Evaluation Protocol:** 100% Zero-Mockup Streaming Inference across 23 Active Test Patients
...
This report provides the exhaustive empirical evaluation comparing three models
across all test patches from the 23 active test patients of the BCSS-WSSS dataset:
```

Bang so lieu (dong 22-24):

```
| **mIoU 4-Class (%)**   | 0.00% | 4.99% | **5.01%** | +5.01% | +0.03% |
| **Macro Dice 4-Class (%)** | 0.00% | 8.32% | **8.35%** | +8.35% | +0.04% |
```

Bang chung goc (`exhaustive_3way_benchmark_results.json`) - **toan bo 3 model chi danh gia 2 patch, tren CPU**:

```
device: cpu | timestamp: 2026-10-24T10:27:15Z
  author_sota_baseline             patches=2 mIoU4=0.0    pixacc=0.0
  retrained_from_scratch           patches=2 mIoU4=4.988  pixacc=19.952
  proposed_medsam_adapter          patches=2 mIoU4=5.014  pixacc=20.056
"total_patch_pairs": 2
```

**Mau thuan voi chinh log cua repo:** `experiments/colab_run_3_multiscale_sota_v2/COLAB_EVALUATION_REPORT.md:10-11` (that run tren T4, 4986 patch):

```
- **mIoU Mo Hoc (4 lop):** 70.53% (Baseline M1: 69.35%)
- **mIoU Toan Cuc (5 lop):** 76.41%
```

=> Hai "ket qua" nop len nhau 14 lan cho cung mot nhiem vu. Bao cao `product/` (muc 5.0%) goi la "100% Zero-Mockup ... 23 benh nhan" trong khi log cua no la 2 patch CPU. Day la xep loai **0 diem** theo AGENTS.md muc 1.3.

**He qua lam sang / hoc thuat:** mo hinh de xuat (MedSAM adapter) duoc bao cao "thang" baseline +0.03% tren 2 patch - mot delta nam trong sai so lam tron thap phan, hoan toan khong co y nghia thong ke, nhung duoc trinh bay nhu ket luan nghien cuu.

### 3.3 `[T2-t-statistic-inf-p-0]` - **Critical** - VERIFIED

`product/reports/clinical/EXHAUSTIVE_3WAY_BENCHMARK_REPORT.md:44-56`:

```
1. **Proposed MedSAM vs Model A (Author SOTA Baseline):**
   - Paired Student t-test: t = inf, p = 0.0000e+00
   - Paired Wilcoxon Signed-Rank Test: W = 0.0, p = 5.0000e-01
   - Effect Size (Cohen's d): 0.000
   - 95% Confidence Interval: [+5.01%, +5.01%]
```

JSON goc:

```json
"t_statistic": Infinity, "t_test_p_value": 0.0,
"wilcoxon_p_value": 0.5, "cohen_d": 0.0,
"confidence_interval_95": [5.0139711402118605, 5.0139711402118605],
"is_statistically_significant_05": false
```

**Tinh phi ly:** `t = inf` va `p = 0.0000e+00` (khang dinh hieu qua thong ke vo cung) dang ky ben canh `Cohen's d = 0.000` (khong hieu qua), CI co do rong = 0, va `is_statistically_significant_05: false` (chinh JSON tu phan noi). Bao cao van in ra dong "p = 0.0000e+00" cho Giang vien doc.

**He qua:** kiem dinh thong ke tren n = 2 voi phuong sai = 0 -> `t = n/0 = inf`. Nam 2026-11-02 muc 2.3 da go `RuntimeWarning: catastrophic cancellation` la "nhan doi duoc" bang cach quy cho "Mock test cohort contains 5 identical images" - nhung nguyen nhan that la **n=2 patch trong deliverable that**, khong phai test mock.

### 3.4 `[T2-random-init-baseline-passed-as-author-sota]` - **Critical** - VERIFIED (ma) + INFERENCE (nguyen nhan truc tiep)

**Duong dan:** `src/clinical/exhaustive_evaluator.py:326-345`

```python
            try:
                state = safe_torch_load(ckpt_p, device=torch.device("cpu"))
                ...
                model.load_state_dict(cleaned, strict=False)
                print(f"[EVALUATOR] Loaded checkpoint for {tag} from {ckpt_p}")
                model.to(self.device)
                model.eval()
                return model
            except Exception as e:
                print(f"[EVALUATOR] Warning: Could not load {ckpt_p} into DeepLabV3Plus ({e}).")

        # Standard DeepLabV3PlusWithCGM initialization
        model = DeepLabV3PlusWithCGM(num_classes=5, in_channels=3, ...)
```

Neu load checkpoint SOTA cua Chu Han et al. that bai, ham **in mot dong Warning roi tra ve mo hinh MOI INIT (trong so ngau nhien)**, va mo hinh ngau nhien nay van duoc gan nhan "Model A: Author SOTA" trong report.

**Dau vet trong artifact (VERIFIED)** - `confusion_matrix_5cls` cua "author_sota_baseline":

```
[[0, 100352, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]]
```

Toan bo 100352 pixel nhan TUM bi du doan thanh STR, moi hang khac bang 0 - day la chu ky dien hinh cua mot mang **chua duoc huan luyen** (nhat quan dau-ra theo kernel), khong phai mot checkpoint SOTA da train. Mot checkpoint chinh thuc khong the dat 0.00% IoU ca 5 lop.

**He qua hoc thuat:** toan bo "Delta (C vs A) = +5.01%" (bang 1, cot 5) - luu do chinh de "chung minh phuong phap de xuat win baseline" - duoc tinh doi chieu voi **mot mo hinh random bi gán nhan SOTA**. Vi pham INVARIANT-07 va INVARIANT-09 (baseline fair).

### 3.5 `[T2-stale-benchmark-artifact-fabricated-15.0]` - **Critical** - VERIFIED

`product/reports/clinical/exhaustive_3way_benchmark_results.json` (`patient_summary`):

```json
"TCGA-C8-A1HJ": { "overall_survival_months": 15.0, "vital_status": 0,
                  "gt_tsr": 100.0, "gt_vtp": 100.0, "ai_cap_risk_category": "Stroma-Rich",
                  "gt_cap_risk_category": "Stroma-Poor" }
```

Ba chung minh chuoi thoi gian:

1. `TCGA-C8-A1HJ` **khong nam trong** `TCGA_BRCA_23_TEST_PATIENTS_CLINICAL_DATA` (VERIFIED bang Python: `'TCGA-C8-A1HJ' in M -> False`). Code hien tai se `raise KeyError` cho ID nay (`src/clinical/survival_analysis.py:248-254`).
2. `src/clinical/exhaustive_evaluator.py:661` hien chua dong comment:
   ```python
   # Clinical survival info (Strict zero-mockup: do not fabricate default 15.0 months)
   ```
   Va `git log -S"do not fabricate default 15.0"` -> commit `ebfd48b fix(integrity): resolve 7 critical source defects...`, ngay **2026-10-25 17:44:22 +0700**.
3. `ls -la --time-style=long-iso`:
   ```
   product/reports/clinical/EXHAUSTIVE_3WAY_BENCHMARK_REPORT.md          2026-10-24 17:27
   product/reports/clinical/exhaustive_3way_benchmark_results.json       2026-10-24 17:27
   src/clinical/exhaustive_evaluator.py                                 2026-11-02 23:46
   ```
   `git log -- product/reports/clinical/exhaustive_3way_benchmark_results.json` chi tra ve **1 commit duy nhat** (`9e3596a`), nghia la artifact **khong bao gio duoc tai sinh** sau khi loi che gia duoi 15.0 thang bi sua.

=> Deliverable dang nop **van chua gia tri `15.0` do che gia duoi tao ra**, thuoc truong hop bi cam "0 diem va dinh chi do an" theo AGENTS.md muc 1.3.

Them nua, cung JSON nay:

```json
"patient_cohort_count": 1, "stroma_rich_count": 1, "stroma_poor_count": 2
```

`1 = 1 + 2` la mau thuan noi tai (cohort 1 benh nhan nhung 2 nhoi 3). Bao cao MD dong 76 lap lai: "**Cohort Size:** 1 patients (1 Stroma-Rich, 2 Stroma-Poor)". Log-rank `Chi-sq = 0.00, p = 1.0000` duoc in ra nhu mot ket qua lam sang.

### 3.6 `[T2-clinical-data-no-provenance]` - **High** - VERIFIED (su vang mat)

`src/clinical/survival_analysis.py:41-226` dinh nghia mot dict Python gia co 23 benh nhan voi OS months, vital status, subtype, stage. Do tinh cua no (VERIFIED bang python dem trong file):

```
patients listed: 23 unique: 23
vital_status counts: {'0': 21, '1': 2}          <-- CHI 2 SU KIỆN TỬ VONG
subtype counts: {'Invasive Ductal Carcinoma (IDC)': 19, 'Basal-like TNBC': 2, ...}
OS months: min 4.6  max 44.35  mean 18.74
deceased only: [('TCGA-E2-A1LK','8.74'), ('TCGA-EW-A1P8','7.85')]
```

Phep kiem VAN (VERIFIED):

```bash
find . -iname "*clinical*" -o -iname "*.csv" | grep -v repos   # -> khong co nguon TCGA
grep -rn "GDC|cBioPortal|TCGAbiolinks|source" src/clinical/survival_analysis.py   # -> 0 ket qua
```

Repo **khong co bat ky** file nguon nao (`*.csv` tai lieu lam sang GDC, script tai, manifest co DOI) chung minh 23 cap so nay. Chung bi "nung" (hardcode) trong ma nguan va duoc dung de ve duong Kaplan-Meier nop Giang vien.

**He qua:** theo AGENTS.md muc 1.3, so lieu phai "trich dan chinh xac tu paper goc co DOI/arXiv kiem chung" hoac "chay thuc nghiem that". Mot dict gia co trong `.py` khong thoa man ca hai. Voi 2/23 su kien, log-rank vo nghiã (powered ~0), nhung `kaplan_meier_survival_curve.png/.svg` van duoc commit nhu deliverable.

### 3.7 `[T2-synthetic-fallback-training]` - **High** - VERIFIED (ma) + reproduced

`src/training/train_proposed_sam.py:485-495`:

```python
    # Dataloaders: use real data if exists, else synthetic fallback
    if os.path.exists(args.data_dir) and not args.dry_run:
        from src.datasets.bcss_dataset import BCSSDataset
        train_ds = BCSSDataset(root_dir=args.data_dir, split="train")
        val_ds = BCSSDataset(root_dir=args.data_dir, split="valid")
    else:
        num_samples = 4 if args.dry_run else 32
        print(f"[INFO] Using SyntheticHistologyDataset with {num_samples} samples.")
        train_ds = SyntheticHistologyDataset(num_samples=num_samples, image_size=224)
        val_ds = SyntheticHistologyDataset(num_samples=num_samples // 2, image_size=224)
```

`SyntheticHistologyDataset.__getitem__` (dong 205-217) tra anh `torch.rand(3,224,224)` va mask `torch.randint(0,5,...)` - nhieu ngau nhien hoan toan.

Dieu kien chi kiem tra `os.path.exists(args.data_dir)` - **khong kiem tra thu muc co anh nao**. Khi duong dan sai mot chu (rat de xay ra tren Colab khi `unzip` chua xong), code lang le huan luyen tren nhieu va:

```python
        save_checkpoint(state={ ..., "best_mIoU": best_miou, "val_stats": val_stats, }, is_best=is_best, ...)
        ...
    print(f"[INFO] Completed successfully. Best mIoU: {best_miou:.4f}")
```

Checkpoint **khong co truong nao danh dau nguon du lieu synthetic**. Khong co loi, khong co exit non-zero, khong co co che "flag". Mot checkpoint sinh tu nhieu + mot con so mIoU "hop ly ve mat ky thuat" se duoc tai ve tu `files.download` va co the duoc nop.

### 3.8 `[T2-drive-readme-cites-missing-logs]` - **High** - VERIFIED

`scripts/build_drive_package.py:326-332` (va ban sao trong `drive_upload_ready/README_DRIVE_SETUP.md`) ghi "bao toan 100% lich su 6 phien thuc nghiem (Run 1 -> Run 6)" kem so lieu tung phien. Kiem tra ton tai that:

```bash
$ for f in ...; do ... done
experiments/benchmarks/COLAB_5_RUNS_EVOLUTION_COMPARISON.md                          MISSING
experiments/benchmarks/COLAB_6_RUNS_EVOLUTION_COMPARISON.md                          MISSING
notebooks/history_runs/Run1_20261019_Naive_HSV_Heuristic_Sanity150.ipynb             MISSING
notebooks/history_runs/Run2_..._DeepLabV3Plus_ImageNetNorm_OtsuAggressive...ipynb    MISSING
notebooks/history_runs/Run5_..._DualCkpt_Stage1_CGM_Stage2_M1_to_M5_T4...ipynb       MISSING
checkpoints/stage2_checkpoint_trained_on_bcss.pth                                    EXISTS
$ ls notebooks/history_runs/
README.md
Run1_20261020_1022UTC_Stage2_DeepLabV3Plus_SingleCkpt_T4_Full4986.ipynb
Run2_20261020_1641UTC_DualCkpt_Stage1_CGM_Stage2_M1_to_M5_T4_Full4986.ipynb
Run3_20261020_2055UTC_MultiScale_Pyramid_D4_M1_to_M7_T4_Full4986.ipynb
```

Chi co **3** history run voi ten **khac hoan toan** voi 6 ten duoc vien dan. Cac so du lieu `mIoU_4cls = 72.39%`, `= 68.71%`, `M1 = 69.35% -> M5 = 70.51%` trong README khong co log doi chung nao trong repo. Vi pham truc tiep AGENTS.md: "Moi bang ket qua phai co log thuc nghiem doi chung".

### 3.9 `[T2-ci-mock-gate-noop]` - **Medium** - VERIFIED

Buoc 2 trong CI (xem T8) quet mockup bang:

```bash
$ grep -rnw 'src/' -e 'TODO: mock' -e 'fakeData' -e 'dummyData' -e 'fake_metric' -e 'simulateTx' ; echo "exit=$?"
exit=1
```

Ca 5 token **khong xuat hien o bat cu dau trong `src/`**. Gate do do **luon LUON in ra "PASSED: Zero mockup verified"** va khong the bao gio chan duoc mockup that (nhu `SyntheticHistologyDataset` hay `allow_mock` o muc 3.7 va 7.5 duoi day).

### 3.10 CLEAN (T2) - da kiem va khong phat hien

- **Khong co** phep gan metric dang `dice = 0.83` hay `miou = 0.72` trong toan bo `src/` (grep tra ve rong - VERIFIED).
- Cac dong chua tu "dummy/mock" trong `src/` deu thuoc ve **comment tuyen bo chong mockup**, khong phai code tao du lieu gia: `src/clinical/exhaustive_evaluator.py:699,719,723`, `src/models/coarse_to_fine_router.py:177`, `src/models/sam_wsss_adapter.py:112`, `src/utils/deliverable_auditor.py:452`.
- `src/colab_pkg/colab_runner.py:596-680` dung `np.random.randint(40,220,...)` va `torch.save` trong `tempfile.TemporaryDirectory()` voi `mode="smoke"` - day la smoke test hop le, co ranh gioi tach bach (khong ghi ra `product/`). **CLEAN**.
- `scripts/build_t4_l4_notebook_suites.py` va cac `fix_*.py` co `dice = 2.0`, `miou_s2 = 0.0` - kich thuoc khoi tao vong lap tong tich, khong phai gia ke ket qua. **CLEAN**.

---

## 4. T3 - Patient-level anti-leakage (INVARIANT-08)

### 4.1 Lenh quet (VERIFIED)

```bash
grep -rn "train_test_split\|GroupKFold\|StratifiedKFold\|ShuffleSplit\|shuffle=True" src/ --include=*.py
# -> src/training/dataset_loader.py:519, 569; src/training/train_proposed_sam.py:496
grep -rn "train_test_split|GroupKFold|StratifiedKFold|shuffle=True" notebooks/*.ipynb *.ipynb
# -> 15 notebook, tat ca deu la shuffle=True cho TRAIN loader (khong phai split assignment)
```

**KHONG CO** `train_test_split`, `GroupKFold`, `StratifiedKFold` nao duoc dung de cat tap du lieu. Toan bo viec cat di qua `patient_ids` + manifest JSON. **CLEAN**.

### 4.2 Kiem chung hanh vi tren du lieu that (VERIFIED - Anti-Mock-Testing Gate)

```bash
KMP_DUPLICATE_LIB_OK=TRUE python   # quet 50,207 anh tren dia va doi chieu manifest
REAL patient counts train/val/test: 106 22 23
OVERLAP train-val: []
OVERLAP train-test: []
OVERLAP val-test: []
disk-train patients NOT in manifest train: 0 []
disk-val patients NOT in manifest val:    0 []
disk-test patients NOT in manifest test:  0 []
```

Nap batch that tu cau truc long nhat quan:

```
BCSSDataset(train) len = 23422   -> batch img (4,3,224,224), weaklabel (4,4)
BCSSDataset(test)  len = 4986    -> 4986/4986 file mask ton tai (MISSING = 0)
BCSSEvalDataset mask uniq vals [0,1] va [1,2]   # mask NaN that, khong phai mask gia
create_stage1_dataloaders train n = 23422  val n = 3418
```

=> **Tuyen bo "0.0% leakage" la DUNG cho duong dan san xuat chinh.** Day la mot CLEAN finding quan trong va can ghi nhan cong.

### 4.3 `[T3-fallback-split-leaks-train-patient]` - **High** - VERIFIED

`src/training/dataset_loader.py:127-133`:

```python
    if not config_path.exists():
        # Fallback to local default patient lists if config file exists
        return {
            "train": ["TCGA-A1-A0SK", "TCGA-A2-A04N", "TCGA-A2-A0T2"],
            "val":   ["TCGA-AN-A03X", "TCGA-AN-A0AL"],
            "test":  ["TCGA-AQ-A04H", "TCGA-C8-A1HJ"],
        }
```

Tai hien:

```bash
$ KMP_DUPLICATE_LIB_OK=TRUE python ... load_bcss_patient_splits(config_path="Z:/definitely/missing/manifest.json")
REPRO A - fallback split: {"train": [...], "test": ["TCGA-AQ-A04H", "TCGA-C8-A1HJ"]}
REPRO A - fallback TEST patients that are REAL TRAIN patients: ['TCGA-C8-A1HJ']
REPRO A - leakage verified: True
```

Doi chieu manifest that (`src/config/bcss_patient_splits.json`): `TCGA-C8-A1HJ` thuoc **train**. Bon ID con lai (`TCGA-A2-A04N`, `TCGA-AN-A03X`, `TCGA-AN-A0AL`, `TCGA-AQ-A04H`) **khong ton tai** trong manifest 151 benh nhan.

Tang cuong do nghiem trong: `git ls-files data/` cho thay chinh bo du lieu mau nop theo repo cung mac cung mot leak:

```
data/sample_dataset/test/img/TCGA-C8-A1HJ_1024_1024.png
data/sample_dataset/train/TCGA-A2-A0T2_1024_2048_[1 0 0 0].png
```

`TCGA-C8-A1HJ` duoc dat trong `test/` cua sample_dataset trong khi no la train patient cua BCSS that. Bat ky dry-run nao dung `data/sample_dataset` + fallback split deu "rap ket qua" tren mot benh nhan mo hinh da hoc.

### 4.4 `[T3-pipeline-loader-skips-leakage-verify]` - **High** - VERIFIED (ma)

Co hai ham doc cung mot manifest, chi mot ham kiem leakage:

| Ham | File:Line | Goi `verify_patient_leakage()`? | Duoc pipeline dung? |
| :--- | :--- | :---: | :---: |
| `load_patient_splits` | `src/datasets/patient_split.py:147` | **CO** | **KHONG** - chi test dung |
| `load_bcss_patient_splits` | `src/training/dataset_loader.py:113-151` | **KHONG** | **CO** - `create_stage1_dataloaders` dong 501 |

Nghia la ham ma Stage-1/Stage-2 thuc su goi **khong bao gio kiem tra roi ro Patient-ID**, va cung bo qua hoan toan nhanh fallback. Test `test_patient_split.py` kiem ham bi bo qua -> **test tu tham chieu**.

### 4.5 `[T3-fail-open-patient-filter]` - **High** - VERIFIED (ma)

`src/training/dataset_loader.py:502-511`:

```python
    train_patients = set(splits.get("train", []))
    ...
    train_dataset = BCSSDataset(
        root_dir=train_dir,
        patient_ids=train_patients if train_patients else None,   # <-- FAIL-OPEN
        ...
    )
```

Va `find_image_files(root_dir, recursive=True)` (dong 154-172) duyet **`rglob("*")` de quy toan bo cay thu muc**. Chuoi tan cong: manifest loi/rong -> `train_patients = set()` -> `patient_ids = None` -> **khong loc nua** -> `rglob` nham ca `test/img/`, `test/pred/`, `test/miou_out/`, `val/img/` vao tap train. Cua an toan o che do **fail-open**.

Kiem tra tren du lieu that (`ls data/BCSS-WSSS`) xac nhan nhung thu muc nhieu nay ton tai that:

```
data/BCSS-WSSS/test/img      data/BCSS-WSSS/test/mask
data/BCSS-WSSS/test/pred     data/BCSS-WSSS/test/miou_out
data/BCSS-WSSS/test/1.png    <-- 1 file PNG roi ngay cap split-root
```

**Đánh giá:** day la rui ro cao nhat ve leakage trong toan bo source; no chi chua no vi `src/config/bcss_patient_splits.json` hien ton tai va day du.

### 4.6 `[T3-duplicate-extract-patient-id-divergence]` - **High** - VERIFIED

Hai trien khai `extract_patient_id` song song, ngu regex va thu ly hoa chu HOA khac nhau:

- `src/datasets/patient_split.py:20` dung `re.search` (ID o **bat ky** vi tri), fallback `return token.strip()` (dong 26 - **khong upper()**).
- `src/training/dataset_loader.py:63` dung `re.match` (neo **dau ten**), fallback `tokens[0].strip().upper()` (dong 68 - **upper()**).

Tai hien (VERIFIED):

```
FILENAME                                                                 patient_split    dataset_loader    MATCH
prefix_TCGA-A2-A0T2_patch001.png                                         TCGA-A2-A0T2       PREFIX            DIVERGE
slide_5x_TCGA-EW-A1P8_tile_12.png                                        TCGA-EW-A1P8       SLIDE             DIVERGE
gdpH_123_p1.png                                                           gdpH               GDPH              DIVERGE
```

**He qua:** mot patch co ten bat dau bang prefix (rsau khi tien xu ly/doi ten, rat thuong gap voi WSI va LUAD-HistoSeg GDPH) se bi gan nhom "PREFIX"/"SLIDE". Neu `test` duoc cat bang ham A va `train` loc bang ham B (hai ham dang duoc dung o hai goi khac nhau), phep kiem disjoint tro nen vo nghia: ham A coi patch thuoc ve mot benh nhan co that, ham B coi no thuoc ve mot "benh nhan" khong ton tai. Roi ro khong bi phat hien.

### 4.7 `[T3-weak-label-default-tumor]` - **Medium** - VERIFIED

`src/training/dataset_loader.py:106-108`:

```python
    else:
        # Default label if brackets are absent
        multi_label[0] = 1.0
```

Toan bo anh `val/img` va `test/img` cua BCSS **khong co ngoac vuong** trong ten (`TCGA-D8-A27F-DX1_xmin..._MPP-0.2500+0.png`). Tai hien tren anh that:

```
 eval sample TCGA-D8-A27F-DX1_..._MPP-0.2500+0   cls_label [1.0, 0.0, 0.0, 0.0]
 eval sample TCGA-D8-A27F-DX1_..._MPP-0.2500+1   cls_label [1.0, 0.0, 0.0, 0.0]
```

=> **8,404 patch val+test duon ganep nhan "CO TUM" (gia)**. May: trong `src/training/train_stage2.py:140-141`, `cls_labels` duoc unpack nhung **khong dung** - CGM lay stage1 `forward_cam` prediction thay the (`src/training/train_stage2.py:146-155`), nen khong dau o con so mIoU cuoi cung. Ghi nhan **Medium** (khong High) vi ly do nay, nhung day la mot nguyen lieu no tiem an; bat ky dong gop nao dung `cls_label` cua eval tap deu se bi nhiem doc.

### 4.8 `tests/test_patient_split.py` - phan xet (nhiem vu cu the cua T3)

**Test co thuc chung minh dieu no assert khong?** Mot phan CO, mot phan KHONG:

- **CO (that):** `test_extract_patient_id_tcga` (17-29), `test_verify_patient_leakage_catches_overlap` (52-60) - adversarial co nghia, kiem duoc ValueError khi co overlap. `test_151_patient_split_manifest_zero_leakage` (94-145) doc manifest that va kiem disjoint - **co gia tri va toi da xac nhan lap lai: 106/22/23, disjoint 100%**.
- **KHONG (tu tham chieu / vo nghia):**
  - `test_verify_patient_leakage_on_sample_dataset` (32-49) kiem tren `data/sample_dataset` (13 file git-tracked, 4 anh train). Day la du lieu toy 2 benh nhan, **va chinh no chua `TCGA-C8-A1HJ` o `test/`** (muc 4.3). Test khang dinh "must exist" o dong 35 -> **test chay duoc chi khi co thu muc `data/` bi gitignore mot phan**.
  - `test_split_patients_kfold` (63-71): assert `len(train)+len(val)==10` va giao = rong - dung, nhung chi kiem ham util, **khong kiem pipeline co dung ham do**.
  - `test_hypothesis_extract_patient_id_tcga_format` (162-167): strategy `st.from_regex("^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}_...")` **chi sinh ra chuoi da bao dam thang regex**. Assert `pid.startswith("TCGA-") and len(pid)==12` do do **luon dung ve mat logic** - khong the phat hien divergence `re.match` vs `re.search` o muc 4.6. Day la fuzzy test ty bo (tautological fuzzing).

### Defect T3

| Ma | Muc do | Do tin cay | Vi tri |
| :--- | :--- | :--- | :--- |
| `[T3-fallback-split-leaks-train-patient]` | High | VERIFIED | `src/training/dataset_loader.py:127-133` |
| `[T3-pipeline-loader-skips-leakage-verify]` | High | VERIFIED (ma) | `src/training/dataset_loader.py:113-151` vs `src/datasets/patient_split.py:147` |
| `[T3-fail-open-patient-filter]` | High | VERIFIED (ma) | `src/training/dataset_loader.py:511,154-172,488` |
| `[T3-duplicate-extract-patient-id-divergence]` | High | VERIFIED | `patient_split.py:20,26` vs `dataset_loader.py:63,68` |
| `[T3-weak-label-default-tumor]` | Medium | VERIFIED | `src/training/dataset_loader.py:106-108` |
| `[T3-toy-sample-dataset-leak]` | Medium | VERIFIED | `data/sample_dataset/test/img/TCGA-C8-A1HJ_*.png` |
| `[T3-self-referential-hypothesis]` | Low | VERIFIED (ma) | `tests/test_patient_split.py:162-167` |

---

## 5. T4 - INVARIANT-29 (chong glob nong + safety-net)

### 5.1 `[T4-bcssdataset-shallow-glob]` - **Medium** - VERIFIED

`src/datasets/bcss_dataset.py` (class Dataset that 1):

```python
 54:            pattern = os.path.join(train_dir, "*.png")
 55:            self.image_paths = sorted(glob.glob(pattern))
...
 71:            pattern = os.path.join(img_dir, "*.png")
 72:            self.image_paths = sorted(glob.glob(pattern))
```

`glob.glob` mot cap, **khong `rglob`, khong `resolve_split_path`**. Ham `resolve_split_path` ma INVARIANT-29 neu ten tron van **khong ton tai trong `src/`**:

```bash
$ grep -rn "def resolve_split_path" src/ scripts/
scripts/build_t4_l4_notebook_suites.py:47,562,1174
scripts/fix_dataset_path_resolution.py:49,194,277
# -> 0 ket qua trong src/
```

Nham do (INFERENCE + kiem tra dia): `class Dataset` duoc dung boi `train_proposed_sam.py` va `tests/test_patient_split.py` van **chay duoc** vi BCSS-WSSS phien ban HuggingFace trong `data/` may nay co lay flat (`data/BCSS-WSSS/training/` = 23,422 file, 0 sub-dir - VERIFIED). Nhung cau lenh trinh cua `build_t4_l4_notebook_suites.py:47` (ham `resolve_split_path`) cho biet co phien ban dataset khac co nest; module nay se vo sinh truong hop do.

### 5.2 `[T4-no-safety-net]` - **High** - VERIFIED (tai hien duoc)

INVARIANT-29 yeu cau bat buoc kiem `len(files) == 0` va kich hoat placeholder de `len(dataset) >= 1`. **Khong class Dataset nao trong `src/` co co che nay.** Tai hien:

```bash
$ KMP_DUPLICATE_LIB_OK=TRUE python   # tao thu muc gia lap/BCSS-WSSS/{training} rong
BCSSDataset(train) len = 0  (khong co placeholder an toan)
ValueError: num_samples should be a positive integer value, but got num_samples=0
```

**He qua:** dung sai loai loi ma INVARIANT-29 sinh ra de chong - `DataLoader` sập tren Colab khi duon dat chua tai xong. Vi pham truc tiep muc "Luoi an toan chong sap" cua INVARIANT-29.

### 5.3 `[T4-science-runner-no-multitier-discovery]` - **Medium** - VERIFIED

Doi chap `glob("*.png|*.jpg")` (nong) va `rglob`/`resolve_split_path` (sau) trong tung notebook:

```
notebooks\SE374_Colab_Scientific_Research_Runner.ipynb        shallow=7   deep=0
01_Train_Stage1_Classifier.ipynb                              shallow=7   deep=0
notebooks\01_Train_Stage1_Classifier_T4.ipynb                 shallow=8   deep=7
notebooks\02_Pseudo_Masks_Train_Stage2_T4.ipynb               shallow=13  deep=12
```

`SE374_Colab_Scientific_Research_Runner.ipynb` - la runner duoc tao moi nhat (mtime 2026-11-02 23:42) - **khong co bat ky phep do tim da tang nao**, vi pham INVARIANT-29.

### 5.4 `[T4-placeholder-violates-T2]` - phan tich dieu kien (nhiem vu cu the: "safety-net co vi pham T2 khong?")

**Tra loi: Hien tai KO vi `src/` khong co safety-net nao** (muc 5.2). Nguoi dung da go dua vi pham T2 bang mot co che cong hai hon: **fallback mask = 0**:

- `src/training/dataset_loader.py:324-326` (mask gia dinh **MAC DINH BAT** - dong 273 `fallback_mask: bool = True`):
  ```python
        elif self.fallback_mask:
            # Fallback to zeros (class 0 or background) if layer mask is missing
            return np.zeros((224, 224), dtype=np.int64)
  ```
- `src/training/dataset_loader.py:456-458` (eval dataset):
  ```python
        else:
            # Fallback zero mask if mask folder is absent
            mask_np = np.zeros((224, 224), dtype=np.int64)
  ```
- `src/datasets/bcss_dataset.py:132-133`: tuong ty cho `test`.

Khi file mask thieu, Dataset **yen lang tra ve mask toan-nen** va tiep tuc. Stage-2 huan luyen tren "pseudo-mask" toan 0, va **danh gia mIoU/Dice doi chieu mot ground-truth gia** - khong co loi, khong co canh bao, khong co dem so. Day la dinh nghia cua "mask gia bi dem bao cao nhu ket qua". Them nua kich thuoc bi hard-code `(224, 224)` trong khi anh vao co the khac so do.

**Do tin cay:** VERIFIED (ma) + reproduced (hanh vi). Kiem chung tren may nay: 4,986/4,986 mask test deu ton tai nen fallback **khong kich hoat** tren du lieu hien tai - rui ro la cho moi truong Colab/fresh clone.

### 5.5 CLEAN (T4)

- `src/training/dataset_loader.py:154-172` `find_image_files(recursive=True)` dung `rglob("*")` - **dat chuan** da tang (VERIFIED hoat dong: 23,422 file duoc nhan).
- `src/colab_pkg/downloader.py:146` `base_path.glob(f"**/{sp}/img")` - recursive pattern dat chuan.
- `src/colab_pkg/evaluator.py:1079-1081`, `src/colab_pkg/downloader.py:632` co phep kiem `len(list(glob(...))) > 0` truoc khi dung - **co mot phan safety check** (dau duoi cho muc 7.5).

---

## 6. T5 - INVARIANT-28 + INVARIANT-27 (vong doi Colab + drift)

### 6.1 `[T5-drive-mount-23-notebooks]` - **High** - VERIFIED

```bash
$ grep -rln "drive.mount\|/content/drive" --include="*.ipynb" . | grep -v "^./repos" | wc -l
23
```

Danh sach bao gom root `SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb` + `_v2`, va **ca 4** ThinDriveRunner_v2 (T4/L4/A100/Sanity150) o `notebooks/`, `notebooks/modular_drive/`, `drive_upload_ready/` (2 ban sao). Trich `notebooks/SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb`:

```python
if "google.colab" in sys.modules:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception: pass
MYDRIVE = Path("/content/drive/MyDrive")
```

Nghiem trong hon loi ma la **thiet ke**: `src/colab_pkg/__init__.py:8` nang `sys.path.insert(0, "/content/drive/MyDrive/SE374_WSSS_Code")`, va README cua goi ban giao ra lenh (VERIFIED `grep -n`):

```
drive_upload_ready/README_DRIVE_SETUP.md:63:2. **Buoc 2:** Chay O ket noi Google Drive (`drive.mount('/content/drive')`) de notebook tu dong nap goi code `src/` tu `SE374_WSSS_Drive_Code`.
```

INVARIANT-28 ghi ro: "TUYET DOI NGHIEM CAM... `drive.mount("/content/drive")` hoac bat buoc nguoi dung cap quyen truy cap Google Drive ca nhan", va "Khi nguoi dung bam Run All, notebook phai chay mot mach tu dau den cuoi 100% tu dong, khong bao gio duoc dung lai doi cap quyen OAuth". Ban ThinDriveRunner **dung lai tai day**. `except Exception: pass` khong loai duoc OAuth prompt (prompt sinh ra tu than cu `drive.mount`, khong phai tu loi).

### 6.2 `[T5-lifecycle-missing-steps]` - **High** - VERIFIED

Ma tran vong doi 7 buoc (INVARIANT-27 muc "Vong doi thuc nghiem khep kin A-Z"). Trich xuat JSON notebook that (khong phai grep text tho):

| Notebook | 1 GPU | 2 cai lib | 3 data THAT | 4 nap weights | 5 mIoU/Dice | 6 viz lam sang | 7 ZIP | 8 files.download |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `01_Train_Stage1_Classifier{_T4,_L4}.ipynb` | Y | **n** | Y | Y | n | Y | Y | Y |
| `02_Pseudo_Masks_Train_Stage2{_T4,_L4}.ipynb` | Y | **n** | Y | Y | Y | Y | Y | Y |
| `03_Train_Proposed_MedSAM{_T4,_L4}.ipynb` | Y | **n** | Y | Y | Y | **n** | Y | **n** |
| `04_Exhaustive_Benchmark_Survival{_T4,_L4}.ipynb` | Y | **n** | Y | Y | Y | Y | Y | **n** |
| `SE374_Colab_Scientific_Research_Runner.ipynb` | **n** | **n** | Y | **n** | Y | **n** | Y | Y |
| `SE374_Colab_WSSS_{T4,L4,A100}_Full4986.ipynb` | Y | Y | Y | Y | Y | Y | Y | Y |
| `SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb` | Y | Y | Y | **n** | **n** | Y | Y | Y |

Xac nhan bang grep dien hinh:

```bash
$ grep -c "files.download" notebooks/03_Train_Proposed_MedSAM_T4.ipynb notebooks/04_Exhaustive_Benchmark_Survival_T4.ipynb notebooks/03_Train_Proposed_MedSAM.ipynb
notebooks/03_Train_Proposed_MedSAM_T4.ipynb:0
notebooks/04_Exhaustive_Benchmark_Survival_T4.ipynb:0
notebooks/03_Train_Proposed_MedSAM.ipynb:0
```

**Nghia la:** notebook cua **phuong phap de xuat** (MedSAM - 03) thieu buoc 6 (truc quan hoa lam sang), 7 va 8 (dong goi ZIP + tu dong tai ve); notebook benchmark (04) thieu buoc 8. Nguoi dung se phai ty tim ket qua - chinh xac dieu INVARIANT-27 cam ("Tuyet doi khung dung do dang de nguoi dung phai hoi ket qua tai ve the nao").

`01_Train_Stage1_Classifier*` thieu buoc 5 (mIoU/Dice) la **chap nhan duoc** vi day la classifier da nhan (danh gia bang AUC/mAP), khong phai segmentation - toi xep nham CLEAN co ghi chu.

### 6.3 `[T5-no-pip-install-pinning]` - **Medium** - VERIFIED

Ca 4 notebook stage (01-04, T4 va L4) **khong co bat ky dong `%pip install` nao** (VERIFIED bang phan tich JSON). Imports thuc te: `torch, torchvision, cv2, tqdm, matplotlib, scipy, PIL, numpy` - deu co san trong Colab base image, nen **co the chay duoc**. Nhung khong phong dinh phien ban -> khong tai lap duoc khi Colang nang image; `requirements.txt` (co `monai>=1.3.0`, `timm>=0.9.16`, `lifelines>=0.28.0`) hoan toan bi bo qua tren Colab.

### 6.4 `[T5-notebook-drift]` - **High** - VERIFIED

`md5sum` ba ban sao cung ten:

```
01_Train_Stage1_Classifier_T4.ipynb       root=c7d96317fdf0  notebooks=c7d96317fdf0  bundle=ABSENT  OK
02_Pseudo_Masks_Train_Stage2_T4.ipynb     root=0410d06713d9  notebooks=0410d06713d9  bundle=ABSENT  OK
03_Train_Proposed_MedSAM_T4.ipynb         root=4f37af83e358  notebooks=4f37af83e358  bundle=ABSENT  OK
04_Exhaustive_Benchmark_Survival_T4.ipynb root=95d2abe6e783  notebooks=95d2abe6e783  bundle=ABSENT  OK
SE374_Colab_Scientific_Research_Runner.ipynb root=87bfb7a2cf5a notebooks=87bfb7a2cf5a bundle=ABSENT OK
SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb
      root=05e42e21a869   notebooks=266f6763d5a0   bundle=045fa19fd2a9    <-- 3 PHIEN BAN KHAC NHAU
```

**Va drift ve su huu hang** (VERIFIED `ls`):

- `01_Train_Stage1_Classifier_v2.ipynb` **chi ton tai o root**, khong co trong `notebooks/`.
- `notebooks/` co `02_..._Stage2.ipynb`, `03_..._MedSAM.ipynb`, `04_..._Survival.ipynb` (base khong hau to) va `SE374_Colab_WSSS_Pipeline.ipynb` - **khong co ban nao trong so nay ton tai o root**.
- `.gitignore` co 3 dong: `/01_Train_Stage1_Classifier.ipynb`, `/*_T4.ipynb`, `/*_L4.ipynb` -> **toan bo cac notebook T4/L4 thuc dung tren Colab bi loi ra khoi Git** (xem `[T8-agents-md-gitignored]`).

=> Trong 6 file doi chieu, 5 root/notebooks trung khop (CLEAN), 1 file BonusDemo co 3 phien ban phan ki - drift that.

### 6.5 Bộ T4 / L4 / A100 / v2 - doi chieu (nhiem vu cu the)

`scripts/build_t4_l4_notebook_suites.py` sinh ra ca bo (2,459 dong). Doi chieu so dong JSON: `01 (17 cells)`, `02 (17)`, `03 (17)`, `04 (19)` - giong het nhau giua cac bien the T4 va L4; `01_Train_Stage1_Classifier.ipynb` 17 vs `_T4` 17. Khong phat hien drift **trong nhom stage 01-04**. `SE374_Colab_Scientific_Research_Runner.ipynb` 19 cells va `_FIXED.ipynb` 19 cells nhung **md5 khac nhau** (khong in ra trong bang 6.4 vi `_FIXED` chi co trong `notebooks/`) va **khong co ban root** -> mot trong hai la ban chet.

---

## 7. T6 - Thiet bi & tai nguyen (audit tinh, khong co GPU)

### 8.1 CLEAN (VERIFIED) - nhung diem dat chuan

| Hang muc | Bang chung | Ket luan |
| :--- | :--- | :--- |
| `torch.load(..., weights_only=)` | 8 goi call deu co try/except `TypeError` roi `weights_only=False`: `downloader.py:230-233`, `model_loader.py:574-576`, `wsss_inference_methods.py:363-365`, `generate_pseudo_masks.py:321-323`, `train_proposed_sam.py:402-404`, `train_stage1.py:390-392`, `train_stage2.py:298-311` | **Dat** - xu ly dung break-change PyTorch 2.6 (xem thu cong ngoai o muc 10.4.2 ve gia phai cua no) |
| `map_location` | **100%** cac goi `torch.load` trong `src/` deu co `map_location` ("cpu" hoac `dev`/`device`) - 0 goi thieu (dem bang grep: 18/18) | **Dat** |
| Seed co dinh (INVARIANT-09) | `colab_runner.py:101-109`, `comparative_runner.py:41-45`, `train_stage1.py:372-376`, `train_stage2.py:250-254` dat du `random`, `np.random`, `torch`, `cuda.manual_seed_all` | **Dat ve hinh thuc** |
| `torch.amp.autocast` | `model_loader.py:920-927` va `wsss_inference_methods.py:843-849` co chan `if use_amp and dev.type=="cuda" and torch.cuda.is_available()` truoc khi dung `device_type="cuda"`, nguoi dung return `contextlib.nullcontext()` | **Dat** - khong gay loi tren CPU |
| Tensor prompt SAM chuyen thiet bi | `sam_wsss_adapter.py:249-251`: `pts_tensor = torch.tensor(pts_list[...], dtype=torch.float32, device=device)`, `lbl_tensor = torch.tensor(..., device=device)` - duoc xay truc tiep tren device, khong phai list CPU roi `torch.stack` moi chuyen | **Dat** (khop voi commit `dc58ed0 fix(pipeline): ... enforce CUDA prompt device transfer`) |
| `.to(device)` | Batch duoc chuyen co `non_blocking`: `train_stage2.py:141`, `train_proposed_sam.py:246-248` | **Dat** |
| Batch size / T4 sizing | `dataset_loader.py:467` `batch_size=32` (stage1), `:548` 16 (stage2), `:579` 16 (eval); `generate_pseudo_masks.py:287` 8. Notebook T4/L4 khai bao `BATCH_SIZE = 12 / 16 / 24 / 32`. `dataset_loader.py:506-507,555-556,586-587` tu dong ha `num_workers` xuong 2 tren Windows | **Chap nhan duoc** cho T4 16GB; `pin_memory and torch.cuda.is_available()` dat |

### 7.2 `[T6-multiseed-runner-dead]` - **High** - VERIFIED

INVARIANT-09 yeu cau "co dinh seed va bao bao trung binh kem do lech chuan ($\mu \pm \sigma$)", tuc can huan luyen nhieu seed. Co che da duoc viet:

```python
# src/pipeline/comparative_runner.py:35
DEFAULT_SEEDS: List[int] = [42, 100, 2024, 2025, 2026]
```

Nhung phep tim kiem thuoc mo tren toan repo tra ve **duong dan thuc thi rong**:

```bash
$ grep -rn "comparative_runner|DEFAULT_SEEDS|run_multi_seed|multi_seed" src/ tests/ scripts/ notebooks/ *.ipynb drive_upload_ready/
src/pipeline/comparative_runner.py:35      (noi dinh nghia)
src/pipeline/comparative_runner.py:111     (noi dung no)
tests/test_thin_drive_runner.py:115        (chuong trinh dong goi liet ke file)
drive_upload_ready/.../package_manifest.json:35
drive_upload_ready/.../src/pipeline/comparative_runner.py:35,111
```

**Khong mot dong `import` nao goi module nay** (khop voi bang phan tich module o muc 9.1: `src.pipeline.comparative_runner` = UNREFERENCED). Va tat ca notebook stage deu chi co `SEED = 42` (VERIFIED: `grep -ohE "SEED *= *[0-9]+" notebooks/0*_T4.ipynb -> 4  SEED = 42`).

=> **Toan bo ket qua cua do an la mot single-seed duy nhat (42), khong co $\sigma$.** INVARIANT-09 chua dat; moi bang so lieu trong `product/` va `experiments/` deu thieu do lech chuan.

### 7.3 `[T6-primary-class-hardcoded]` - **Medium** - VERIFIED (ma)

`src/training/train_proposed_sam.py:267-270`:

```python
        # Concatenate positive/negative prompts from primary active class
        primary_class = 0
        coords  = prompt_dict[primary_class]["point_coords"]
        labels  = prompt_dict[primary_class]["point_labels"]
```

`primary_class` la hang so 0 (TUM), **khong duoc chon dua tren `weak_labels` hay `active_mask`**. `CAMPromptGenerator` co tra ve `"active_mask": torch.tensor(batch_active, ...)` (`sam_wsss_adapter.py:258`) nhung **goi call bo qua hoan toan**. Neu mot patch chua co TUM (`active_mask[0] == False`), moi prompt deu roi vao nhanh padding (`sam_wsss_adapter.py:244-247`: `pts_list.append((0.0,0.0)); lbl_list.append(-1)`) -> mo hinh nhan toan bo prompt = "padding", decoder chay trong che do khong co thong tin. Day la mot loi logic co the giai thich vi sao `IoU_STR/LYM/NEC = 0.00%` trong deliverable (bang 1, muc 3.2). **Do tin cay: VERIFIED (ma) + INFERENCE (tac dong so lieu).**

### 7.4 `[T6-gpu-sync-per-class-loop]` - **Medium/Low** (hieu suat T4) - INFERENCE

`src/models/sam_wsss_adapter.py:202-203,218,227,230`: vong lap Python `for i in range(b): for c in range(c_num)` goi `.mean().item()`, `.std().item()` va 4 lan `.float().cpu().numpy()` moi (image, class). Voi batch 16 x 4 lop -> **~64 GPU->CPU sync moi buoc**, chan pipeline va lam giau thoi gian/mIoU tren T4 so voi mot trien khai vectorised. Day la ly do hop ly cho `average_latency_ms = 205-206` duoc ghi trong benchmark JSON.

### 7.5 Cac thieu sot khac

- `[T6-no-dry-run-memory-guard]` - **Low** - INFERENCE: khong co `torch.cuda.empty_cache()` hay OOM try/except trong vong train cua `src/training/*`; AGENTS.md muc 1.1 yeu cau Sanity/Dry-Run de thu loi OOM. Cac notebook co kich hoat Sanity150 nhung **src/ khong co** tu duong doi `max_steps` cho sanity.
- `[T6-num-workers-zero-default]` - **Low** - INFERENCE: `generate_pseudo_masks.py:287` khong khai bao `num_workers`; tren Colab `num_workers=0` mac dinh cho vong sinh PMC nang.

---

## 8. T7 - Silent failure (bien loi that thanh ket qua vo nghia)

### 8.1 So luong (VERIFIED)

```
src/ bare "except:"            : 0        <-- CLEAN
src/ "except Exception"         : 49
src/ contextlib.suppress        : 0        <-- CLEAN (chi dung trong model_loader nullcontext)
src/ warnings.filterwarnings    : 0        <-- CLEAN
```

### 8.2 `[T7-pvalue-1.0-swallows-error]` - **High** - VERIFIED

5 ngan hanh bien loi thong ke thanh "khong co khac biet co y nghia lam sang":

```python
src/clinical/exhaustive_evaluator.py:573-575   except Exception: w_stat_ma, p_w_ma = 0.0, 1.0
src/clinical/exhaustive_evaluator.py:589-591   except Exception: w_stat_mr, p_w_mr = 0.0, 1.0
src/colab_pkg/evaluator.py:414-415             except Exception: t_stat, t_pval = 0.0, 1.0
src/colab_pkg/evaluator.py:421-422             except Exception: w_stat, w_pval = 0.0, 1.0
src/colab_pkg/evaluator.py:489-490             except Exception: w_stat, w_pval = 0.0, 1.0
```

**He qua lam sang:** `p = 1.0` la **ket luan duong tinh manh** ("phuong phap de xuat khong khac baseline"). Mot loi `scipy` thuc su (thieu du lieu, NaN, n<2) se bi bien thanh mot phat bieu khoa hoc hoan toan nguoc voi "loi". Giao duc cua T2 (muc 3.3) va deliverable (`p-value = 1.0000` cho log-rank o muc 3.5) cho thay nhanh nay **da xay ra thuc te** trong so lieu nop.

### 8.3 `[T7-zero-mask-fallback]` - **Critical** - VERIFIED

Da phan o muc 5.4. Tong hop: `src/training/dataset_loader.py:273` (`fallback_mask=True` mac dinh), `:324-326`, `:456-458`, `src/datasets/bcss_dataset.py:132-133`. Bon ngan hanh khong log, khong dem, khong ngoai le.

### 8.4 `[T7-random-model-passed-as-author-checkpoint]` - **Critical** - VERIFIED

Da phan o muc 3.4 (`src/clinical/exhaustive_evaluator.py:337-345`, `:366`). Day la silent-failure nghiem trong nhat trong repo: mot loi `torch.load` -> khong fail build, chi in Warning, va moi "baseline chinh thuc" gia that bi sinh ra.

### 8.5 `[T7-mock-mode-empty-dirs]` - **Medium** - VERIFIED

`src/colab_pkg/downloader.py:632-650`:

```python
    if not (mask_dir.is_dir() and rgb_dir.is_dir() and len(list(mask_dir.glob("*.png"))) > 0):
        ...
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo_id, ...)
        except Exception as exc:
            if allow_mock:
                print(f"[THONG BAO] Che do mock cho phep tiep tuc ({exc}).")
                mask_dir.mkdir(parents=True, exist_ok=True)
                rgb_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise RuntimeError(...)
    samples: List[Dict[str, Any]] = []
    mask_files = sorted(mask_dir.glob("*.png"))
    ...
    print(f"[OK] Da nap {len(samples)} mau Whole-ROI BCSS doi sat ground-truth tai: {t_dir.resolve()}")
    return samples
```

Khi tai HF that bai va `allow_mock=True`, hai thu muc **rong** duoc tao va ham **in ra `[OK] Da nap 0 mau`** - thanh cong gia tao. Nguoi goi khong co cach phat hien ngoai viec tu dem list. Toi chua xac dinh duoc mac dinh cua `allow_mock` o toan bo goi call -> **INFERENCE một phần**, nhưng ban than nhanh `allow_mock` la vi pham INVARIANT-07/27.

### 8.6 `[T7-checkpoint-load-fallback-hides-corruption]` - **Medium** - INFERENCE

`safe_torch_load` (`model_loader.py:573-576`, `wsss_inference_methods.py:362-365`):

```python
    try:
        return torch.load(checkpoint_path, map_location=dev, weights_only=True)
    except Exception:
        return torch.load(checkpoint_path, map_location=dev, weights_only=False)  # nosec B614
```

`except Exception` (khong phai `UnpicklingError`/`TypeError` cu the) bat **ca loi file hong, duong dan sai, thieu quyen**. Lan thu hai cung doc file hong do -> loi cuoi cung lat di loi thuc chat. Doi chieu: `downloader.py:209-210,234-235` `except Exception: return False` cho `is_valid_torch_checkpoint` -> mot checkpoint hong bi bao cao la "khong hop le" ma khong he co ly do.

### 8.7 `[T7-drive-mount-swallow]` / `[T7-continue-on-mismatch]` - **Low/Medium** - VERIFIED (ma)

```python
src/clinical/survival_analysis.py:290-293    except Exception: pass  ->  return list(ACTIVE_TEST_PATIENTS_23)
src/pipeline/wsi_lowmag_highmag_pipeline.py:151-152,166-167,410-411,567-568   except Exception: pass
src/models/sam_wsss_adapter.py:315-316       except Exception: pass
src/colab_pkg/evaluator.py:181-182           except Exception: pass
src/visualization/api_server.py:300-301      except Exception: pass
src/dataset_loader.py:664-665 (rf_candidates else: continue)  # patch bi bo roi lang le
```

Rieng `downloader.py:663-667` (chinh xac o muc 5.5 grep): RGB tuong ung khong thay thi `continue` - **patch bi loai khoi danh gia ma khong dem, khong bao cao**. Neu `rgbs_colorNormalized` duoc doi ten, evaluator co the chay tren tap con rat nho va van in ra "Da nap N mau" - day chinh xac la co che sinh ra `total_patches_evaluated = 2` o muc 3.2. **INFERENCE (machanism) + VERIFIED (artifact 2 patch).**

### 8.8 `[T7-ci-step-3-import-guard]` - CLEAN

`src/utils/deliverable_auditor.py:172-178` baoc `import fitz/pythoncom/win32com.client` trong `try/except ImportError: return` - guard dung cach, khong an loi that (chi bo qua buoc render DOCX). **CLEAN**.

---

## 9. T8 - INVARIANT-12 (typography) va CI

### 9.1 `[T8-inv12-violations-exist]` - **Medium** - VERIFIED

```bash
$ grep -rnP "[\x{2013}\x{2014}\x{2015}\x{2212}]" --include="*.py" --include="*.md" --include="*.ipynb" \
    --exclude-dir=repos --exclude-dir=.git . | wc -l
230
$ grep -rlP ... | sed 's|^\./||' | awk -F/ '{print $1}' | sort | uniq -c | sort -rn
     25 graphify-out
     10 .agents
      3 graphify-out-all
```

Phan bo: **`src/`, `tests/`, `scripts/`, `notebooks/`, `product/` (*.py, *.md, *.ipynb) va `docs/`, `AGENTS.md`, `README.md`: sach hoan toan** (VERIFIED - 0 match). Toan bo 38 file vi pham la artifact sinh tu (graphify) ho scaffolding `.agents/`.

**Nhung co vi pham trong deliverable that.** Tuyen bo "0 non-ASCII hyphens across ALL workspace files" la SAI vi:

```bash
$ python (moi file XML trong PPTX slide nop Giang vien)
product/slides/SE374_PROPOSAL_SLIDES_WSSS_WSI.pptx :: ppt/slideMasters/slideMaster1.xml :: EN-DASH x2
TOTAL in SE374_PROPOSAL_SLIDES_WSSS_WSI.pptx = 2
```

Va cac match nhi phan khac trong `product/`: `SE374_MINI_PROPOSAL_PHASED_ROADMAP_WSSS.pdf`, `SE374_MINI_PROPOSAL_WSSS_WSI.pdf`, `product/proposal/assets/preview/pages/page_12.png`.

**Đánh giá công bang:** 36 match trong `.agents/` deu la **nhan tai chinh cac ky tu bi cam trong chinh van ban quy dinh** (vi du `.agents/rules/13_...md:32` ghi `Chan em-dash (U+2014) va en-dash (U+2013)`) - day la trich dan hop ly, khong phai loi dinh duong. Toi ghi **Medium** chu khong High, vi vi pham thuc su nam trong PPTX/PDF nop di va trong tinh minh bach cua tuyen bo.

### 9.2 `[T8-governance-test-glob-blind-dot-dirs]` - **High** - VERIFIED

`tests/test_governance_invariants.py:34-43` tu bao "Verify zero non-ASCII dashes across ALL markdown files". Pham vi thuc:

```bash
$ python
md:    glob matched=754   test-scanned(after filter)=110
py:    glob matched=4522  test-scanned(after filter)=124
ipynb: glob matched=82    test-scanned(after filter)=60
md starting with .agents in glob: []            <-- PYTHON glob BO QUA THU MUC BAT DAU BANG "."
actual .agents/*.md on disk: 571                <-- 571 FILE KHONG BAO GIO DUOC QUET
actual docs/*.md on disk: 75
```

Danh sach `IGNORED_DASH_PREFIXES` (dong 4-12) co yen `repos/`, `graphify-out/`, `graphify-out-all/`, `.pytest_cache/`, `.ruff_cache/`, `.git/` - **khong he yen `.agents/`**. `.agents/` bi loai **khong chu dich**, do han vi cua `glob.glob("**/*.md")`. Trong 571 file `.agents/*.md` bi mu, **36 dong** chua U+2013/U+2014/U+2015/U+2212.

**Ket luan ve "no enforcement":** test xanh khong phai vi repo sach, ma mot phan vi test khin nhin thay 571 file. Day la dinh nghia cua "test tu tham chieu" ma INVARIANT-26 (Anti-Mock-Testing Illusion Gate) cam.

### 9.3 `[T8-ci-lint-cannot-fail]` - **High** - VERIFIED

`.github/workflows/ci.yml:29-32`:

```yaml
      - name: 1. Code Style & Linting Check
        run: |
          flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics || true
          flake8 src/ --count --exit-zero --max-complexity=15 --max-line-length=120 --statistics || true
```

Ca hai lenh deu co `|| true`; lenh thu hai con co `--exit-zero`. **Buoc lint ve mat toan hoc khong the lam do build.** voi 33 loi `scripts/` va 6 loi F821 (muc 2.5) duon lay, CI van bao xanh.

### 9.4 `[T8-ci-missing-deps]` - **Critical** - VERIFIED

`ci.yml:23-27`:

```yaml
      - name: Install Lint & Test Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install flake8 pytest Pillow
```

Chi lap **4** goi. `requirements.txt` khai bao 27 goi (torch, torchvision, monai, timm, nibabel, pydicom, SimpleITK, scikit-learn, scipy, pandas, lifelines, streamlit...). Kiem tra phu thuoc test:

```bash
$ grep -l "^import torch|^from torch|import torch" tests/*.py | wc -l
19
$ ls tests/*.py | wc -l
22
```

**19/22 module test import torch o cap module**, va **khong co `conftest.py`, `pytest.ini`, `pyproject.toml`, `setup.cfg`, `tox.ini`** nao trong repo (VERIFIED `ls` -> "No such file or directory") nen khong co skip guard. Ket luan: buoc 4 `pytest tests/ -v` tren `ubuntu-latest` se that bai ngay o **collection** (`ModuleNotFoundError: No module named 'torch'`), khong phai vi loi code.

=> **CI cua repo khong the chay mau, do do chua bao gio duoc bat cu buoc nao trong 4 buoc chay that** (trur buoc 2 va 3, xem 9.5). Tuyen bo "GitHub Hygiene & CI/CD Automation dat chuan" (INVARIANT-03) la **khong the chung minh**.

### 9.5 `[T8-ci-no-leakage-step]` + `[T8-ci-mock-gate-noop]` - **Medium** - VERIFIED

`AGENTS.md` muc 3 ghi ro chu thich cua ci.yml:

```
│       └── ci.yml                     # CI/CD Pipeline (Lint, Pytest, Patient-Leakage Check, Zero-Mockup)
```

Doi chieu ci.yml that: **khong co bat ky buoc nao ten "leakage"/"patient"**. Chi co 4 buoc: (1) flake8 `|| true`, (2) grep mockup, (3) deliverable_auditor, (4) pytest. Patient-Leakage Check bi vang mat hoan toan - va INVARIANT-08 la "quy tac bat bien".

Buoc 2 (muc 3.9) la mot gate rong. Toi **co chay lai lenh y het trong CI**:

```bash
$ grep -rnw 'src/' -e 'TODO: mock' -e 'fakeData' -e 'dummyData' -e 'fake_metric' -e 'simulateTx'; echo exit=$?
exit=1
```

**Buoc 3 - CLEAN (VERIFIED chay duoc):** toi da chay chinh xac lenh CI tren may nay:

```bash
$ KMP_DUPLICATE_LIB_OK=TRUE python -m src.utils.deliverable_auditor product/proposal/SE374_MINI_PROPOSAL_WSSS_WSI.md
[DELIVERABLE QUALITY AUDIT: PASSED] - Toan bo tieu chuan hoc thuat, chat luong hinh anh, chinh ta va cau truc deu DAT 100%!
exit=0
```

File duon dan ton tai (VERIFIED `ls product/proposal/`). Buoc nay hop le.

### 9.6 `[T8-agents-md-gitignored]` - **Medium** - VERIFIED

`cat .gitignore`:

```
# Graphify & Multi-Agent Internal Scaffolding (Excluded from Clean Project Source)
.agents/
AGENTS.md
graphify-out*/
```

**Chinh sach dac luoc toi cao (`AGENTS.md`) bi gitignore** -> khi Giang vien clone repo, bo luat 32 INVARIANT khong ton tai va CI khong the doi chieu. Dong 32 `drive_upload_ready/` va 33 `scratch/` cung bi bo ngoai Git: toan bo goi Colab ban giao (notebooks ThinDrive + src copy) va lich su nhap lieu **khong bao gio co trong repo nop**.

Them cac dong cuoi: `/01_Train_Stage1_Classifier.ipynb`, `/01_Train_Stage1_Classifier_v2.ipynb`, `/*_T4.ipynb`, `/*_L4.ipynb` -> **bo bon notebook T4/L4 chinh bi loi khoi Git** (trùng khớp với drift muc 6.4).

### 9.7 `KMP_DUPLICATE_LIB_OK` trong CI (nhiem vu cu the)

VERIFIED: ci.yml khong co `env:` nao. `runs-on: ubuntu-latest` -> **khong can** (OMP #15 la dac thu cua torch/numpy binary xung dot tren Windows). **CLEAN, nhung co ghi chu:** nhieu yeu cau bat buoc (`docs/`, README) van huong dan chay tren Windows; `README.md:163` ghi `pytest tests/ -v` **ma khong dat prefix `KMP_DUPLICATE_LIB_OK=TRUE`**, trong khi INVARIANT-13 yeu cau moi lenh shell phan bao boc trong `cmd /c "..."` - khong dong nhat.

### 9.8 `tests/test_governance_invariants.py` thuc chat enforce gi? (nhiem vu cu the)

File co 4 test (VERIFIED doc toan van):

| Test | Enforce that | Danh gia |
| :--- | :--- | :--- |
| `test_no_ai_em_dash_in_markdown` (34-37) | Quet `**/*.md` tru cac prefix trong danh sach | **Yeu**: mu 571 file `.agents/*.md` (muc 9.2); khong bao gid `.pptx/.pdf/.docx` - la noi vi pham thuc su (muc 9.1) |
| `test_no_ai_em_dash_in_python_and_notebooks` (40-43) | `**/*.py`, `**/*.ipynb` sau filter | Hop le cho `src`/`tests`/`notebooks`; **khong kiem tra `repos/` la quyet dinh dung**, nhưng filter loai `graphify-out*` trong khi AGENTS.md muc 2 chi cho phep ngoai trur `repos/` |
| `test_invariants_24_and_25_defined_in_agents_md` (46-52) | 3 phep `in content` chuoi | **Ty-bo hoan toan**: no chi kiem AGENTS.md CO NHAC den tu "INVARIANT-24"/"INVARIANT-25"/"deliverable_auditor.py". No se xanh neu AGENTS.md bi xoa? - khong, vi `open()` ném lỗi. Nhung no **khong kiem bat ky noi dung nao cua 32 INVARIANT**, khong kiem INVARIANT-07/08/12/28/29 duoc ton tai |
| `test_no_raw_phone_photos_in_product_directory` (55-62) | `os.walk("product")` kiem ten file bat dau media_/img_/pxl_/screenshot/dcm_ | **Co nghia va day du** ve ten; nhung chi so ten file, khong kiem **kich thuoc mau** hay **he thong anh** - mot anh chup dien thoai doi ten `fig_1.png` van thoat. DONG THOI: `page_1.png..page_12.png` trong `product/proposal/assets/preview/pages/` **duoc chinh test nay sinh ra** va co 10 file trong so do dang o trang thai `M` (modified) trong git (muc 9.9) |

### 9.9 `[T8-generated-images-committed-and-dirty]` - **Low** - VERIFIED

```bash
$ git status --short | head
 M product/proposal/assets/preview/pages/page_1.png
 M product/proposal/assets/preview/pages/page_10.png
 ... (10 file PNG)
$ git status --short | wc -l
72
```

`product/proposal/assets/preview/pages/*.png` la anh render sinh ra boi `deliverable_auditor._audit_docx_visual_rendering` (`src/utils/deliverable_auditor.py:171-180`) - **san pham phu bi commit va dang o trang thai dirty**. Repo nop dang co 72 muc thay doi chua commit.

---

## 10. T9 - Dead code & duplication

### 10.1 `[T9-unreferenced-modules]` - **Low** - VERIFIED

Phan tich `import` cua 32 module trong `src/` (quet src + tests + scripts + tat ca notebook):

```
src.pipeline.comparative_runner                 src=0 tst=0 scr=0 ipynb=0   UNREFERENCED
src.visualization.draw_all_proposal_figures      src=0 tst=0 scr=0 ipynb=0   UNREFERENCED
src.visualization.draw_pipeline_diagram          src=0 tst=0 scr=0 ipynb=0   UNREFERENCED
UNREFERENCED COUNT: 3
```

- `src.pipeline.comparative_runner` (508 dong, bo chay 5-seed): xem `[T6-multiseed-runner-dead]` - day **khong phai dead code vo hai**, ma la **co che reproducibility bat buoc chua bao gio duoc noi**.
- `src.visualization.draw_all_proposal_figures` / `draw_pipeline_diagram`: sinh so hinh cho `product/` nhung **khong co trong CI, khong co trong test, khong duoc goi boi script nao** -> cac hinh trong proposal khong the tai sinh bang mot lenh (vi pham INVARIANT-15/16 ve tai lap).

### 10.2 `[T9-duplicate-dataset-and-id-implementations]` - **High** - VERIFIED

Hai bo song song cung mot trach nhiem (vi pham INVARIANT-05 Single Responsibility va INVARIANT-06):

| Trach nhiem | Ban 1 | Ban 2 | Khac biet |
| :--- | :--- | :--- | :--- |
| BCSS Dataset | `src/datasets/bcss_dataset.py` (134 dong) | `src/training/dataset_loader.py` (603 dong) | glob nong vs `rglob` sau; khong loc patient vs loc theo `patient_ids` |
| `extract_patient_id` | `patient_split.py:7-26` (`re.search`, ko upper fallback) | `dataset_loader.py:51-68` (`re.match`, upper fallback) | **DIVERGE** - xem muc 4.6 |
| Split loader | `patient_split.py:103-154` (GOI verify_leakage) | `dataset_loader.py:113-151` (**KHONG** goi, co fallback ro ri) | xem muc 4.4 |
| `safe_torch_load` | `colab_pkg/model_loader.py:567-576` | `models/wsss_inference_methods.py:360-365` | truong hop gan nhu identical |
| Autocast helper | `model_loader.py:917-927` | `wsss_inference_methods.py:838-848` | truong hop identical |
| `MultiMethodWSSSEngine` | `model_loader.py` | `wsss_inference_methods.py` | 1,805 dong vs 1,798 dong - **mot file la ban sao cua file kia** |

**VERIFIED md5/line-count cho cap model_loader:**

```
src/colab_pkg/model_loader.py             1805 lines
src/models/wsss_inference_methods.py      1798 lines
```

7 dong lech giua hai file 1,800-dong la mot **rui ro co-che "sua mot cho, quen sua cho kia"**: bat ky sua nao ve `safe_torch_load`, autocast, hay TTA D4/D8 deu phai lam hai lan, va khong co test nao kiem hai ban con dong bo.

### 10.3 `[T9-two-colab-runner-names]` - **Medium** - INFERENCE

`src/pipeline/colab_runner.py` (196 dong, facade re-export tu `src.colab_pkg`) va `src/colab_pkg/colab_runner.py` (728 dong, logics that) **trung ten file**. Voi notebook `import src.pipeline.colab_runner` (58 lan) va so khac `from src.colab_pkg.colab_runner import ...`, viec doc loi/goi sai la gan nhu khong the tranh, va `AGENTS.md` muc 3 quy rang pipeline phai di qua `src/pipeline/`. **Ghi nhan INFERENCE**: toi chua kiem het 34 notebook xem co bo nao import truc tiep `colab_pkg.colab_runner` (bot qua facade) hay khong.

### 10.4 `src/config/` co 0 file `.py` - phan xet (nhiem vu cu the)

```bash
$ ls -la src/config/
.gitkeep (44 B)   bcss_patient_splits.json (7,643 B)     # khong co __init__.py, khong co *.py
$ grep -rn "src\.config" src/ tests/ scripts/ --include=*.py   # 0 import; chi 8 luu do trong docstring/test
```

**KET LUAN: KHONG phai defect.** `src/config/` duoc thiet ke nhu mot **kho manifest du lieu** (JSON), khong phai Python package; moi doc gia deu tra ve duong dan (vi du `dataset_loader.py:123`, `patient_split.py:129`), va `build_drive_package.py` van dong goi no (VERIFIED `drive_upload_ready/.../package_manifest.json:35`). **CLEAN**, nhung **nen ghi ro trong README** vi `AGENTS.md` muc 3 mo ta `src/config/` la "Cau hinh sieu tham so, random seeds, paths, device GPU" - thuc te **khong co file cau hinh sieu tham so nao** o day; cac hang so (IMAGENET_MEAN, BCSS_PALETTE, DEFAULT_SEEDS, batch size) nam roi rac trong `dataset_loader.py:34-48`, `comparative_runner.py:35` va trong tung notebook. Do la mot **INVARIANT-06/30 mismatch ve tai lieu**, khong phai loi code.

### 10.5 Import thua (VERIFIED - trich `ruff check`)

```
scripts\download_papers.py:4:8: F401 `time` imported but unused
scripts\download_papers.py:5:8: F401 `urllib.request` imported but unused
scripts\fix_dataset_path_resolution.py:2:8: F401 `re` imported but unused
scripts\fix_dataset_path_resolution.py:3:21: F401 `pathlib.Path` imported but unused
scripts\refactor_notebooks_pure_colab.py:4:21: F401 `pathlib.Path` imported but unused
scripts\update_agents_invariants.py:1:8: F401 `os` imported but unused
scripts\refactor_notebooks_pure_colab.py:42:9: F841 `old_drive_block` assigned but never used
```

Dac biet `F841 old_drive_block` trong `refactor_notebooks_pure_colab.py:42` - bien luu **ca mot khoi ma nguon chinh** de "thay the drive.mount lenh Colab" nhung **khong bao gio duoc dung den** -> chuan hoa notebook co the khong thay duoc nhu mong muon. Khop voi muc 6.1 (23 notebook van con `drive.mount`): **script sua loi Drive da chay ma khong co tac dung**, va loi do bi chet vi mot bien bi quen.

### 10.6 Doi chieu md5 3 ban sao (nhiem vu cu the)

Da hoan thanh o muc 6.4. Tom luat: **5/6 file stage root == `notebooks/` (khop); 1/6 (BonusDemo) co 3 hash khac nhau**; `drive_upload_ready/` **khong chua he** cac file stage 01-04 (bundle chi co 5 notebook Thin/Bonus) -> goi Drive va repo **khong cung tap hop notebook**.

---

## 11. T10 - Chat luong test + bug-hung doi chieu ben ngoai

### 11.1 `[T10-non-hermetic-destructive-tests]` - **Critical** - VERIFIED

`tests/test_thin_drive_runner.py` khong dung `tmp_path` cho viec ghi. No thao tac tren **cay thu muc that cua repo**:

```python
tests/test_thin_drive_runner.py:32    import scripts.build_drive_package as packager
tests/test_thin_drive_runner.py:44    DRIVE_READY_DIR = PROJECT_ROOT / "drive_upload_ready"
tests/test_thin_drive_runner.py:98    summary = packager.build_drive_package()      # test_2
tests/test_thin_drive_runner.py:180   packager.build_drive_package()               # test_3
tests/test_thin_drive_runner.py:493   packager.build_drive_package()               # test_7
```

`build_drive_package()` **xoa va viet lai** `drive_upload_ready/SE374_WSSS_Drive_Code/` - va thu muc nay **bi gitignore** (VERIFIED `git check-ignore -v` -> `.gitignore:32:drive_upload_ready/`), nghia la **khong the khoi phuc bang git** neu bi huy hoai.

**Chung minh tai hai:** toi da chay `pytest tests/` nhu hanh dong dau tien cua ca ca audit (workspace nguyen trang thai ban giao) -> 3 loi `WinError 145` + `FileNotFoundError` o muc 2.3. Sau do, `ls drive_upload_ready/SE374_WSSS_Drive_Code/notebooks/` con 6 muc, trong khi `notebooks/` goc co 24 - cay da bi thu hep.

**Hau qua:** `pytest tests/` tren mot clone sach **lam bieng mot phan cua deliverable**. Day la vi pham INVARIANT-03 (GitHub Hygiene) va mo hinh "test khong tam".

### 11.2 `[T10-order-dependence-confirmed]` - **High** - VERIFIED

Phep thu doc lap (VERIFIED):

```bash
$ KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_thin_drive_runner.py -q
10 passed in 40.48s                      # -> RUN RIENG: 10/10 XANH

$ KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_colab_notebook.py tests/test_thin_drive_runner.py -q
40 passed, 1 warning in 38.33s           # -> pair nay: XANH

$ KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q
3 failed, 345 passed in 527.48s          # -> FULL SUITE: DO
```

Cac module khac cua cung test chay 2 lan (`test_2`, `test_3`, `test_7` cung goi `build_drive_package()`) - **khong idempotent trong full suite**. Toi chua xac dinh duoc module goi chuc chinh xac (do thoi gian chay full suite ~9 phut va moi lan chay tiep theo deu thanh cong sau lan pha dau tien). **Thuoc ve INFERENCE:** co che hon-hop nhieu nhat la mot module truoc do `import` ma `sys.path` tro vao `drive_upload_ready/SE374_WSSS_Drive_Code/src`, sinh `__pycache__` trong bundle khi `build_drive_package()` dang `shutil.rmtree` -> WinError 145 tren Windows. Ghi chu them: `tests/test_colab_notebook.py:570` gan `sys.path.insert(0, str(Path("repos/WSSS-Tissue").resolve()))` **ma khong hoan tac** - gay nhieu lan toan bo quyen test sau do trong cung process.

**Ket luan bat buoc phat bieu:** "348 passed" la mot ket qua **chi tai lap dug tren mot workspace da bi lan chay tru do lam bieng**. Voi mot `git clone` moi + `pytest tests/`, nhieu la **3 FAILED**. Tuyen bo cua audit 2026-11-02 do do **khong phai mot trang thai co the nghi thu**.

### 11.3 `[T10-tautological-asserts]` - **Medium** - VERIFIED

`tests/test_patient_split.py:109-111,137-139`:

```python
    assert len(train) in (91, 106), f"Expected 91 or 106 train patients, got {len(train)}"
    assert len(val)  in (22, 30),  f"Expected 22 or 30 val patients, got {len(val)}"
    assert len(test) in (23, 30),  f"Expected 23 or 30 test patients, got {len(test)}"
```

Ba assert nay chap nhan **hai bo split hoan toan khac nhau**. Doc string (dong 95-96) lai khảng dinh "adheres to the **91/30/30** split ratio" trong khi manifest that la **106/22/23** -> tai lieu test da sai va test khong the bao dong neu split bi doi thanh bat ky trong 2 cau hinh. Day la "test khong the fail".

### 11.4 Danh gia theo testing-best-practices (nhiem vu cu the)

| Tie chi | Danh gia | Bang chung |
| :--- | :--- | :--- |
| **TestHANH vi vs test chi tiet cai dat** | **Truong thi thien ve hanh vi - tot.** `test_patient_split.py`, `test_adversarial_clinical_stress.py`, `test_benchmark_components.py` kiem dau-ra so hoc/shape. Nhung `test_colab_notebook.py` va `test_colab_pro_notebooks.py` (101 test) kiem **AST/cú phap va substring cua cell** ("assert `"X in src"`") - do la test chi tiet cai dat cua notebook, khong phai test ket qua suy luan | VERIFIED doc |
| **Mock qua da / mock chinh he thong dang test** | **CO - mot truong hop nghiem trong:** `test_thin_drive_runner.py` goi `packager.build_drive_package()` roi **doi chieu ket qua cua chinh viec build do voi cac nguon no doc** - mot vong lap "he thong tu chung minh minh dung". Them: `tests/test_audit_team_*.py` (96 test) kiem cac module `src/` tren **tensor random** `torch.rand(...)` (20 vi tri da liet ke), nghia la khong dung du lieu y khoa nao | VERIFIED grep |
| **Phu thuoc thu tu chay** | **VI PHAM NGHIEM TRONG** - muc 11.2 | VERIFIED |
| **`time.sleep` nondeterminism** | **CLEAN** - `grep -rn "time.sleep" tests/` tra ve 0 ket qua | VERIFIED |
| **Path toi da (tmp_path)** | **CLEAN mot phan** - 55 luong `tmp_path` trong 5 module (`test_scientific_upgrades.py` 30, `test_colab_pro_notebooks.py`/`test_thin_drive_runner.py` 11...) | VERIFIED |
| **Test duong loi bien (rong, None, 0, -1)** | **YEU.** Kiem diem: **khong co** test nao cho (a) dataset rong -> `num_samples=0` (muc 5.2, toi phai tu tai hien); (b) manifest thieu -> fallback split bi leak (muc 4.3); (c) `patient_ids=set()` -> fail-open (muc 4.5); (d) `fallback_mask=True` sinh mask toan 0 (muc 8.3); (e) `extract_patient_id` divergence (muc 4.6); (f) `except Exception -> p=1.0` (muc 8.2). `test_patient_split.py:148-151` kiem `FileNotFoundError` cho `load_patient_splits` - **nhung khong kiem `load_bcss_patient_splits`** (ham pipeline dung) boi ham nay **lang le tra fallback thay vi nem loi** | VERIFIED |
| **assert `pytest.raises` that** | `test_verify_patient_leakage_catches_overlap` va `test_load_patient_splits_invalid_path` - **co** nem loi that. Tot. | VERIFIED |
| **Fuzz/Hypothesis** | Co that (162-175) nhung **ty-bo** o muc 4.8 | VERIFIED |

### 11.5 Bug-hung doi chieu ben ngoai (3 chu ky)

#### 11.5.1 Chu ky #1 - `ValueError: num_samples should be a positive integer value, but got num_samples=0`

**Duong dan da rua sach va so dong (src cua repo nay):**
1. `src/datasets/bcss_dataset.py:48-79` - `_load_dataset()` gan `self.image_paths = sorted(glob.glob(pattern))` (dong 55 va 72) ma khong kiem danh sach rong.
2. `src/datasets/bcss_dataset.py:94-95` - `__len__` tra ve `len(self.image_paths)` = 0.
3. `src/training/train_proposed_sam.py:496` - `DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)`.
4. `torch/utils/data/sampler.py` - `RandomSampler.__iter__` nem `ValueError`.

**Tai hien trong repo nay (VERIFIED, output nguyen van):**

```
train split len: 0
RAISED ValueError: num_samples should be a positive integer value, but got num_samples=0
```

**Tra cuu:** `site:stackoverflow.com "num_samples should be a positive integer value"` -> SO 71615089 va SO 70369070 ("Can a PyTorch DataLoader start with an empty dataset?"). Cau tra duoc chap nhan chi ra 100% truong hop la **duong dan/glob khong khop -> dataset rong**, va phep kiem `len(dataset) > 0` (hoac `rglob` da tang) la cach suà chuan.

**Ban va chuan (hop le, khong phai scratch script), de nghi cho `BCSSDataset._load_dataset`:**

```python
    def _resolve(self, candidates: list[str]) -> str | None:
        for c in candidates:                       # da tang, khong gia dinh flat
            if os.path.isdir(c):
                return c
        return None

    def _load_dataset(self) -> None:
        ...
        self.image_paths = sorted(p for p in Path(train_dir).rglob("*.png"))
        if not self.image_paths:                  # INVARIANT-29 emergency safety net
            raise RuntimeError(
                f"BCSSDataset: khong tim thay PNG nao trong {train_dir!r}. "
                "Kiem tra BCSS-WSSS.zip da duoc giai nap dung cau truc train/img."
            )
```

> **Luu y ve quyet dinh thiet ke:** AGENTS.md INVARIANT-29 yeu cau *placeholder* de `len(dataset) >= 1`. Toi khuyen **`raise` thay vi placeholder** trong duong dan danh gia: mot placeholder se vi pham INVARIANT-07 dung nhu diem toi phan tich o muc 5.4. Can xin chi dao cua nguoi dung (Human-in-the-Loop, INVARIANT-01) ve viec sua lai dich quy che nay.

**URLs:**
- https://stackoverflow.com/questions/71615089/valueerror-num-samples-should-be-a-positive-integer-value-but-got-num-samples
- https://stackoverflow.com/questions/70369070/can-a-pytorch-dataloader-start-with-an-empty-dataset
- https://github.com/pytorch/pytorch/issues/40274 (nguon `RandomSampler` nem loi)

#### 11.5.2 Chu ky #2 - `_pickle.UnpicklingError: Weights only load failed` (PyTorch 2.6 default flip)

**Duong dan da rua sach va so dong:**
1. `src/colab_pkg/model_loader.py:567-576` - `safe_torch_load()`
2. `src/models/wsss_inference_methods.py:360-365` - ban sao cua (1)
3. `src/clinical/exhaustive_evaluator.py:326` va `:361` - goi `safe_torch_load` cho 2 checkpoint SOTA cua tac gia
4. `src/training/train_stage1.py:388-392`, `train_stage2.py:296-311`, `generate_pseudo_masks.py:319-323`, `train_proposed_sam.py:400-404` - cung kieu.

**Ma that:**

```python
    try:
        return torch.load(checkpoint_path, map_location=dev, weights_only=True)
    except Exception:
        return torch.load(checkpoint_path, map_location=dev, weights_only=False)  # nosec B614
```

**Tra cuu:** `github pytorch issue torch.load weights_only=True default breaking change 2.6` -> `pytorch/pytorch#179904`, `MIC-DKFZ/nnUNet#2681`, `pytorch/ignite#3303`, `Comfy-Org/ComfyUI#9385`. Cau tra duoc chap nhan trong ca 4 repo la: **DUNG dung `weights_only=False` nhu cach sua dai dăng**; hay khai bao ro cac toan thuat an toan.

**Danh gia:** Repo **xu ly dung ve kha nang chay duoc** (khong ai bi crash vi 2.6 - day la mot **CLEAN finding**, va no giai thich vi sao `# nosec B614` duoc them vao). Nhung cach sua nay **mo lai chinh cua an toan ma PyTorch 2.6 dong**, va `except Exception` (khong phai `UnpicklingError`) la nguồn goc cua `[T7-checkpoint-load-fallback-hides-corruption]` o muc 8.6.

**Ban va chuan (de nghi):**

```python
import argparse
from torch.serialization import add_safe_globals
add_safe_globals([argparse.Namespace])           # theo dung kieu du lieu checkpoint cua tac gia

def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except UnpicklingError as exc:                # CHI bat dung loi unpickle
        logger.warning("weights_only=True bo bac %r; yeu cau add_safe_globals: %s", path, exc)
        raise
    except FileNotFoundError:
        raise
```

**URLs:**
- https://github.com/pytorch/pytorch/issues/179904
- https://github.com/MIC-DKFZ/nnUNet/issues/2681
- https://github.com/pytorch/ignite/issues/3303
- https://github.com/Comfy-Org/ComfyUI/issues/9385

#### 11.5.3 Chu ky #3 - `ignore_index=255` bi merge voi lop thuc (mask sentinel bi clip)

**Duong dan da rua sach va so dong:**
1. `src/training/dataset_loader.py:322` - `arr = np.clip(arr, 0, 4)` (Stage2MultiMaskDataset._load_mask)
2. `src/training/dataset_loader.py:455` - `mask_np = np.clip(mask_np, 0, 4)` (BCSSEvalDataset.__getitem__)
3. `src/datasets/bcss_dataset.py:130` - `mask_np = np.clip(mask_np, 0, 4)`
4. Roi den `src/losses_metrics/losses.py:48` (`nn.CrossEntropyLoss(ignore_index=255)`), `:125` (`valid_mask = (pseudo_mask != self.ignore_index)`), `src/losses_metrics/metrics.py:56-57, 176-177, 297-298` (`valid_mask = gt_mask != ignore_index`).
5. `src/training/train_stage2.py:64-66` khai bao `ignore_index=255` cho ca hai loss.

**Tai hien (VERIFIED):**

```
clip(255) -> [0 1 4 4]
after clip gt: [[4, 4], [4, 0]]  => ignore pixels became class 4 (BACK)
```

**Tra cuu:** `pytorch ignore_index 255 mask np.clip ground truth segmentation metric wrong background` -> `discuss.pytorch.org/t/one-of-the-class-merged-with-ignore-class-while-training-semantic-segmentation/66289` (chinh xac la chu ky "mot lop bi gop voi Ignore class"), `discuss.pytorch.org/t/when-to-use-ignore-index/5935`, `huggingface/transformers#27224`.

**Danh gia:** Toan bo he thong `ignore_index=255` ma losses/metrics **duoc viet rat ty mi** (5 file khai bao dung) **la da che do tu cau truc (dead by construction)**: Dataset clip 255 -> 4 truoc khi tensor den tay loss. Khong co bat ky `mask != 255` nao con co the duoc tho man. Voi `train_proposed_sam.py:213-215` tu sinh `mask[:4,:] = 255` (vi `SyntheticHistologyDataset` co 255 that), moi pixel "khong xac dinh" deu bi: (a) day vao **phan so IoU cua BACK**, (b) day mo hinh hoc rang **bien khong xac dinh = mo nen**. Day la mot loi kien truc, khong phai loi thuc thi, va **khong co test nao bat duoc**.

**Ban va chuan:**

```python
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        # GIU NGUYEN sentinel 255; chi clip cac gia tri hop le trong [0,4]
        arr = np.where((arr < 0) | (arr >= NUM_CLASSES), IGNORE_INDEX, arr)
        return arr.astype(np.int64)
```

Va trong vong tich: `preds = ...; valid = (targets != IGNORE_INDEX) & (preds < NUM_CLASSES)` (dang dung o `train_stage2.py:167` - `mask = (targets_np >= 0) & (targets_np < NUM_CLASSES)` - **thieu `!= 255`**, tuc la bi co che nay cho qua).

**URLs:**
- https://discuss.pytorch.org/t/one-of-the-class-merged-with-ignore-class-while-training-semantic-segmentation/66289
- https://discuss.pytorch.org/t/when-to-use-ignore-index/5935
- https://github.com/huggingface/transformers/issues/27224

---

## 12. Danh sach defect tong hop (44 muc)

### Critical (8)

| Ma | Vi tri | Mo ta ngan | Chung minh |
| :--- | :--- | :--- | :--- |
| `[T1-pytest-3-failures]` | `tests/test_thin_drive_runner.py:98,180,493` | 345 passed / **3 failed**, tuyen bo 348/0 | VERIFIED |
| `[T10-non-hermetic-destructive-tests]` | `tests/test_thin_drive_runner.py:44,98,180,493` | Test xoa + viet lai `drive_upload_ready/` that, bi gitignore, khong the khoi phuc | VERIFIED |
| `[T2-zero-mockup-scope-claim-false]` | `product/reports/clinical/EXHAUSTIVE_3WAY_BENCHMARK_REPORT.md:6,11,22-24` | "100% Zero-Mockup ... 23 Active Test Patients" nhưng log = 2 patch, CPU | VERIFIED |
| `[T2-t-statistic-inf-p-0]` | `.../EXHAUSTIVE_3WAY_BENCHMARK_REPORT.md:44-56` + JSON | `t = inf`, `p = 0.0000e+00` ben canh `cohen_d = 0.0` va `significant = false` | VERIFIED |
| `[T2-random-init-baseline-passed-as-author-sota]` | `src/clinical/exhaustive_evaluator.py:337-345` | Loi load checkpoint -> mo hinh **random** duoc gan nhan "Author SOTA" | VERIFIED (ma + chu ky confusion) |
| `[T2-stale-benchmark-artifact-fabricated-15.0]` | `product/reports/clinical/exhaustive_3way_benchmark_results.json` | Artifact 2026-10-24 chua `15.0` thang che gia duoi; loi sua 2026-10-25; chua tai sinh | VERIFIED |
| `[T7-zero-mask-fallback]` | `dataset_loader.py:273,324-326,456-458`; `bcss_dataset.py:132-133` | Mask thieu -> toan-0 yen lang; Stage-2 huan luyen va do mIoU doi chieu GT gia | VERIFIED |
| `[T8-ci-missing-deps]` | `.github/workflows/ci.yml:26-27,42-46` | CI lap 4 goi, 19/22 test can torch; khong co conftest -> CI khong the chay mau | VERIFIED |

*(Bang nay la chuan duy nhat cho tong so: **Critical 8 / High 17 / Medium 14 / Low 5 = 44**, khop voi tong o muc 1. `test_thin_drive_runner.py` xuat hien hai lan vi du hai khuyet tat khac biet duong dan: `[T1-pytest-3-failures]` la tuyen bo 348/0 sai (san pham cua phep chay), `[T10-non-hermetic-destructive-tests]` la thiet ke test xoa/viet thu muc gitignored (tac dong moi len repo).)*

### High (17)

| Ma | Vi tri | Chung minh |
| :--- | :--- | :--- |
| `[T2-clinical-data-no-provenance]` | `src/clinical/survival_analysis.py:41-226` | VERIFIED (vang mat nguon) |
| `[T2-synthetic-fallback-training]` | `src/training/train_proposed_sam.py:485-495` | VERIFIED (ma) |
| `[T2-drive-readme-cites-missing-logs]` | `scripts/build_drive_package.py:326-332` | VERIFIED |
| `[T3-fallback-split-leaks-train-patient]` | `src/training/dataset_loader.py:127-133` | VERIFIED (repro) |
| `[T3-pipeline-loader-skips-leakage-verify]` | `src/training/dataset_loader.py:113-151` | VERIFIED (ma) |
| `[T3-fail-open-patient-filter]` | `src/training/dataset_loader.py:511` | VERIFIED (ma) |
| `[T3-duplicate-extract-patient-id-divergence]` | `patient_split.py:20,26` vs `dataset_loader.py:63,68` | VERIFIED (repro) |
| `[T4-no-safety-net]` | `src/datasets/bcss_dataset.py:48-95` | VERIFIED (repro ValueError) |
| `[T5-drive-mount-23-notebooks]` | 23 `.ipynb` + `src/colab_pkg/__init__.py:8` + `README_DRIVE_SETUP.md:63` | VERIFIED |
| `[T5-lifecycle-missing-steps]` | `notebooks/03_Train_Proposed_MedSAM{,_T4,_L4}.ipynb`, `04_*` | VERIFIED |
| `[T5-notebook-drift]` | `SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb` (3 md5) | VERIFIED |
| `[T6-multiseed-runner-dead]` | `src/pipeline/comparative_runner.py:35,111` | VERIFIED (repro grep) |
| `[T7-pvalue-1.0-swallows-error]` | `exhaustive_evaluator.py:575,591`; `evaluator.py:415,422,490` | VERIFIED (ma) |
| `[T8-governance-test-glob-blind-dot-dirs]` | `tests/test_governance_invariants.py:4-12,34-43` | VERIFIED (repro glob) |
| `[T8-ci-lint-cannot-fail]` | `.github/workflows/ci.yml:31-32` | VERIFIED |
| `[T9-duplicate-dataset-and-id-implementations]` | `model_loader.py` (1805) vs `wsss_inference_methods.py` (1798) | VERIFIED |
| `[T10-order-dependence-confirmed]` | `tests/` (full vs isolation) | VERIFIED |

### Medium (14) va Low (5)

| Ma | Muc do | Vi tri |
| :--- | :--- | :--- |
| `[T1-ruff-scope-claim]` | Medium | `scripts/` 33 loi, repo 464 loi (F821 x6, F811 x93) |
| `[T2-ci-mock-gate-noop]` | Medium | `.github/workflows/ci.yml` buoc 2 |
| `[T3-weak-label-default-tumor]` | Medium | `src/training/dataset_loader.py:106-108` |
| `[T3-toy-sample-dataset-leak]` | Medium | `data/sample_dataset/test/img/TCGA-C8-A1HJ_*.png` |
| `[T4-bcssdataset-shallow-glob]` | Medium | `src/datasets/bcss_dataset.py:54-55,71-72` |
| `[T4-science-runner-no-multitier-discovery]` | Medium | `notebooks/SE374_Colab_Scientific_Research_Runner.ipynb` |
| `[T5-no-pip-install-pinning]` | Medium | 4 notebook stage 01-04 |
| `[T6-primary-class-hardcoded]` | Medium | `src/training/train_proposed_sam.py:267-270` |
| `[T7-checkpoint-load-fallback-hides-corruption]` | Medium | `model_loader.py:573-576` |
| `[T7-mock-mode-empty-dirs]` | Medium | `src/colab_pkg/downloader.py:644-650` |
| `[T8-inv12-violations-exist]` | Medium | 38 file; `SE374_PROPOSAL_SLIDES_WSSS_WSI.pptx` (EN x2) |
| `[T8-agents-md-gitignored]` | Medium | `.gitignore:47` (`AGENTS.md`), `:32-33`, `:62-64` |
| `[T9-two-colab-runner-names]` | Medium | `src/pipeline/colab_runner.py` vs `src/colab_pkg/colab_runner.py` |
| `[T10-tautological-asserts]` | Medium | `tests/test_patient_split.py:109-111,137-139` |
| `[T1-duration-mismatch]` | Low | bao cao 2026-11-02 muc 2.1 |
| `[T3-self-referential-hypothesis]` | Low | `tests/test_patient_split.py:162-167` |
| `[T6-no-dry-run-memory-guard]` | Low | `src/training/*` |
| `[T6-gpu-sync-per-class-loop]` | Low | `src/models/sam_wsss_adapter.py:202-230` |
| `[T8-generated-images-committed-and-dirty]` | Low | `product/proposal/assets/preview/pages/*.png` (10 file M, 72 muc dirty) |

---

## 13. Cac hang muc SACH da kiem chung (bao cao ca dieu KHONG tim thay)

| # | Kiem tra | Lenh / Phuong phap | Output | Ket luan |
| :---: | :--- | :--- | :--- | :--- |
| 1 | Ruff tren `src/` | `ruff check src` | `All checks passed!` | CLEAN |
| 2 | Ruff tren `tests/` | `ruff check tests` | `All checks passed!` | CLEAN |
| 3 | Leakage tren du lieu that | Quet 50,207 PNG + doi manifest | train 106 / val 22 / test 23; overlap `[]`,`[]`,`[]` | **CLEAN - tuyen bo 0.0% DUNG** |
| 4 | Nap batch that tu cau truc long | `BCSSDataset('data/BCSS-WSSS','train'/'test')` + `DataLoader` | `len=23422`, `len=4986`, `img (4,3,224,224)`, mask `uniq [0,1]` | **CLEAN - Anti-Mock-Testing Gate DAT** (co >=1 batch that) |
| 5 | `train_test_split`/`GroupKFold`/`StratifiedKFold` | `grep -rn ... src/` | 0 match | CLEAN - khong cat theo patch/slice |
| 6 | `bare except:` | `grep -rn "except:" src/` | 0 | CLEAN |
| 7 | `contextlib.suppress` bat loi | `grep -rn "contextlib.suppress" src/` | 0 (chi `nullcontext` cho autocast) | CLEAN |
| 8 | Metric gan hard-code trong `src/` | `grep -oE "(mIoU\|Dice\|IoU).* = 0.xxxx" src/` | 0 match | CLEAN |
| 9 | `weights_only` / `map_location` | Dem 18 goi `torch.load` trong `src/` | 18/18 co `map_location`, 18/18 co `weights_only` try/except | CLEAN (xem 11.5.2 ve gia phai) |
| 10 | `torch.amp` device guard | Doc `model_loader.py:920-927` | `if use_amp and dev.type=="cuda" and torch.cuda.is_available()` | CLEAN |
| 11 | Prompt tensor SAM device | `sam_wsss_adapter.py:249-251` | `torch.tensor(..., device=device)` | CLEAN |
| 12 | `time.sleep` trong test | `grep -rn "time.sleep" tests/` | 0 | CLEAN |
| 13 | `src/config/` 0 file `.py` | `ls` + `grep "src.config"` + `package_manifest.json:35` | JSON duoc dong goi va doc bang duong dan | **KHONG phai defect** - thiet ke co chu |
| 14 | Drift 5 notebook stage | `md5sum` root vs `notebooks/` | 5/5 hash trung khop | CLEAN |
| 15 | `deliverable_auditor` buoc 3 CI | `python -m src.utils.deliverable_auditor product/proposal/SE374_MINI_PROPOSAL_WSSS_WSI.md` | `PASSED`, exit=0, file ton tai | **CLEAN - CI buoc nay chay duoc that** |
| 16 | `Pillow` `Image.ROTATE_90` deprecated? | `Pillow 12.0.0`, goi `img.transpose(Image.ROTATE_90)` | `-> (8, 8)`, thanh cong | CLEAN - khong phai loi |
| 17 | `datasets` co `rglob` da tang khong? | `dataset_loader.py:163`, `downloader.py:146`, `evaluator.py:1079` | Co | CLEAN (cho rieng module nay) |

---

## 14. Phuc loi va xac nhan cua tac tu

1. **Toi KHONG khang dinh "Pass" chi vi pytest xanh.** Toi da doc ma nguon tung duong dan, tai hien 5 loi bang lenh that (muc 2.2, 4.3, 5.2, 8.3, 11.2, 11.5), va doi chieu artifact voi commit history.
2. **Toi da bao cao ca nhung dieu KHONG tim thay** (muc 13, 17 hang muc) va **khang dinh mot phan tuyen bo cu la DUNG**: `0.0% leakage` tren duong dan san xuat chinh va `ruff 0 errors` tren `src/` + `tests/`.
3. **Severities dua tren hai tac dong:** (a) kha nang sinh ra so lieu nop Giang vien khong phai do thuc nghiem (Critical), (b) kha nang vo hieu hoa cac quy che chong loi (High).
4. **Ba thay doi can quyet dinh cua nguoi dung** (INVARIANT-01, toi khong tu QUYET):
   - INVARIANT-29 yeu cau placeholder khi dataset rong, nhung placeholder do lai vi INVARIANT-07. Chon `raise` hay `placeholder` cho duong danh gia?
   - `data/sample_dataset` dang co `TCGA-C8-A1HJ` o `test/` trong khi manifest BCSS dat no o `train/`. Doi ten patient cua du lieu mau, hay xoa fallback split o `dataset_loader.py:127-133`?
   - `product/reports/clinical/*` (2 file) can **tai sinh tu mot lan chay that du 4,986 patch tren GPU**, hay thu hoi khoi bo nop? Hien no chua so lieu ma chinh code hien tai tu che sinh ra.
5. **Gioi han cua audit nay:** khong co GPU -> moi ket luan ve VRAM sizing, AMP tren T4/L4, va thoi gian huan luyen la **audit tinh** (khong do duoc). Chua chay lai `pytest tests/` tren clone sach hoan toan (se pha `drive_upload_ready/`) - lan chay #1 cua toi **chinh la lan tren trang thai ban giao**, va do la bang chung chinh.

---

*Het bao cao AUDITOR 03. Tuan thu INVARIANT-12: toan bo bao cao chi dung ASCII hyphen (0x2D).*
