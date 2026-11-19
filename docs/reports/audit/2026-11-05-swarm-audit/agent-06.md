# AUDIT 06 - Bao Cao Kiem Dinh Doc Lap (Swarm 8 Tac Tu)

**Tac tu:** AUDITOR 06
**Ngay audit:** 2026-11-05
**Repo:** `SE374_WSSS_WORKSPACE`
**Pham vi:** Toan bo 10 chieu kiem dinh T1-T10 tren khap he thong (khong phan ra task)
**Moi truong:** Windows + Git Bash, Python 3.13.9, pytest 8.4.2, ruff 0.12.0, torch 2.10.0+CPU, khong GPU
**Che do:** Chi doc + chay lenh READ-ONLY + ghi duy nhat tep bao cao nay (Builder-Auditor Separation)

---

## 0. Ket Luan Executive Ve Cac Tuyen Bo Cua Audit 2026-11-02

 tep `docs/reports/audit/2026-11-02-swarm-audit/01_SWARM_RECHECK_VERIFICATION.md` tuyen bo 4 dieu. Ket qua kiem lai:

| Tuyen bo cu | Ket luan | Bang chung |
| :--- | :--- | :--- |
| `348 passed / 0 failed` | **SAI** | `2 failed, 346 passed, 2 warnings in 441.75s`. Tong so test = 348 (dung), trang thai pass (sai). |
| `ruff 0 errors` | **SAI (ngoai pham vi)** | `ruff check src tests scripts` = **33 errors**. `ruff check . --exclude repos` = **464 errors**. Chi `src` va `tests` moi 0. |
| `0.0% patient leakage` | **DUNG nhung phuong phap cu la yeu** | Test chi doi chieu file JSON manifest, khong bao gio cham du lieu that. Toi da kiem bang du lieu that (xem T3) -> ket luan DUNG, nhung bang chung cu KHONG chung minh duoc dieu do. |
| `0 non-ASCII hyphen` | **SAI** | **230 dong / 40 file** chua U+2013/U+2014/U+2015/U+2212 trong `.agents/` va `graphify-out*/`. |

**Nhan toan hoc quan trong nhat:** Cac khiếm khuyet nghiem trong nhat nam NGOAI tam mata cua bo test 348 test, cu the o `product/reports/clinical/EXHAUSTIVE_3WAY_BENCHMARK_REPORT.md` (san pham nop Giang vien) - bo test khong bao gio doc tep nay.

---

## 1. T1 - Tang 1: Test Suites Tu Dong [VERIFIED]

### 1.1 Lenh va output that

```
$ KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8 python -m pytest tests/ -q -p no:cacheprovider
2 failed, 346 passed, 2 warnings in 441.75s (0:07:21)
```

### 1.2 Nguyen van chu ky 2 loi (te nhat, theo muc do nghiem trong)

**Loi 1 - `test_2_drive_bundle_packager_creates_complete_folder_and_zip`:**

```
tests\test_thin_drive_runner.py:98: in test_2_drive_bundle_packager_creates_complete_folder_and_zip
    summary = packager.build_drive_package()
scripts\build_drive_package.py:1668: in build_drive_package
    zip_outer_path.unlink()
E   PermissionError: [WinError 32] The process cannot access the file because it is being
    used by another process:
    '<workspace_root>\\drive_upload_ready\\SE374_WSSS_Drive_Code.zip'
```

**Loi 2 - `test_3_new_thin_runner_notebooks_valid_concise_and_modular`:**

```
tests\test_thin_drive_runner.py:180: in test_3_new_thin_runner_notebooks_valid_concise_and_modular
    packager.build_drive_package()
scripts\build_drive_package.py:1545: in build_drive_package
    shutil.rmtree(bundle_dir)
E   PermissionError: [WinError 32] The process cannot access the file because it is being
    used by another process:
    '<workspace_root>\\drive_upload_ready\\SE374_WSSS_Drive_Code\\SE374_WSSS_Drive_Code.zip'
```

### 1.3 Chung minh FLAKINESS (khong phai loi moi)

Toi da chay lap lai rieng tep test nay 4 lan cung mot moi truong:

```
Run 1 (full suite):    2 failed, 346 passed
Run 2 (standalone):    10 passed in 91.75s
Run 3 (standalone):    1 FAILED   (PermissionError nhu tren)
Run 4 (standalone):    10 passed in 54.37s
Run 5 (standalone):    10 passed in 45.44s
Run 6 (standalone):    10 passed in 34.27s
```

-> **Ty le fail quan sat duoc: 2/6 lan (~33%).** day la race condition tren file handle Windows (antivirus/Windows Defender/Search Indexer giu mo `.zip vừa được ghi), KHONG phai do code thay doi.

### 1.4 Dinh nghia loi

`[T1-flaky-drive-bundle-tests]` | **Critical** | Do tin cay: Cao (VERIFIED, tai lap 2 lan) | `tests/test_thin_drive_runner.py:96`, `tests/test_thin_drive_runner.py:180`, thu muc `drive_upload_ready/` qua `scripts/build_drive_package.py:1545` va `:1668`

He qua hoc thuat: bo test la "chot chan nghiem thu" (Anti-Mock-Testing Gate). Mot chot chan ngau nhien fail thi 33% lan chay khong the dung de bao lanh bat ky ket luan "0 failed" nao. Tuyen bo 2026-11-02 la mot **luu do mau (lucky run)** duoc ghi nhu mot dinh luat.

### 1.5 Ruff

| Lenh | Output that |
| :--- | :--- |
| `python -m ruff check src` | `All checks passed!` |
| `python -m ruff check tests` | `All checks passed!` |
| `python -m ruff check scripts` | **Found 33 errors** (19 E741, 7 I001, 6 F401, 1 F841) |
| `python -m ruff check src tests scripts` | **Found 33 errors** |
| `python -m ruff check . --statistics --exclude repos` | **Found 464 errors** |

Phan bo 464 error toan repo:

```
193  I001   unsorted-imports
 93  F811   redefined-while-unused
 53  E701   multiple-statements-on-one-line-colon
 36  F541   f-string-missing-placeholders
 35  F401   unused-import
 25  E401   multiple-imports-on-one-line
 19  E741   ambiguous-variable-name
  6  F821   undefined-name
  4  F841   unused-variable
```

`[T1-ruff-scope-overclaim]` | **Medium** | Do tin cay: Cao (VERIFIED) | `scripts/` (33 loi), `notebooks/` + `graphify-out/` + `.agents/` (431 loi con lai)

Tuyen bo "0 lint errors" chi dung trong pham vi `src/`. Doi nguoi doc bao cao cu, cum tu "Ruff 0 errors" tao cam tuong ca repo sach. Day la loi trinh bay lam phong dai ket qua.

**6 loi F821 (`undefined-name`) - PHAN TINH LA FALSE POSITIVE cua ruff notebook, KHONG phai loi chay:**

```
notebooks\03_Train_Proposed_MedSAM.ipynb:cell 15:14:52: F821 Undefined name `DRY_RUN`
notebooks\03_Train_Proposed_MedSAM_L4.ipynb:cell 15:14:52: F821 Undefined name `DRY_RUN`
notebooks\03_Train_Proposed_MedSAM_T4.ipynb:cell 15:14:52: F821 Undefined name `DRY_RUN`
notebooks\04_Exhaustive_Benchmark_Survival.ipynb:cell 11:32:55: F821 Undefined name `DRY_RUN`
notebooks\04_Exhaustive_Benchmark_Survival_L4.ipynb:cell 11:32:55: F821 Undefined name `DRY_RUN`
notebooks\04_Exhaustive_Benchmark_Survival_T4.ipynb:cell 11:32:55: F821 Undefined name `DRY_RUN`
```

Toi da mo notebook doi chieu (INFERENCE, co bang dang doc):
- `03_Train_Proposed_MedSAM_T4.ipynb` cell 14 dong 14: `DRY_RUN = False if 'DRY_RUN' not in globals() else DRY_RUN` -> duoc bao ve bang `globals()`.
- `04_Exhaustive_Benchmark_Survival_T4.ipynb` cell 10 dong 32: `MAX_TEST_PATCHES = 150 if ('DRY_RUN' in globals() and DRY_RUN) else None` -> short-circuit, an toan.

Ruff pham vi theo cell nen khong nhin thay binding tu cell truoc. **Khong co NameError luc chay.** Ghi nhan o day de cac tac tu khac khong double-count la Critical.

### 1.6 Warnings (2, khop voi bao cao cu - CLEAN)

```
tests/test_clinical_service.py::test_exhaustive_benchmark_evaluator_dry_run_execution
  scipy/stats/_axis_nan_policy.py:423: RuntimeWarning: Precision loss occurred in moment
  calculation due to catastrophic cancellation. This occurs when the data are nearly identical.
  Results may be unreliable.

tests/test_colab_notebook.py::test_stage1_netcam_architecture_and_forward_cam
  repos/WSSS-Tissue/tool/infer_utils.py:6: DeprecationWarning: scipy.misc is deprecated
```

**Bo sung phan tich ma bao cao cu thieu:** warning #1 khong chi la "noise cua mock cohort". No la **hanh vi that cua `src/clinical/exhaustive_evaluator.py:571 stats.ttest_rel(scores_m, scores_a)`** tren cohorts qua nho. Cung co che nay da sinh ra `t_statistic = inf` trong bao cao benchmark nop Giang vien (xem T2). Do la mot trieu chung, khong phai canh bao vo nghia.

---

## 2. T2 - Zero-Mockup [INVARIANT-07]

### 2.1 [T2-benchmark-report-2-patches] Bao cao lam nop Giang vien duoc do tren DUNG 2 PATCH

| Severity | **Critical** | Do tin cay | **Cao - VERIFIED** |
| :--- | :--- | :--- | :--- |

`file:line` - `product/reports/clinical/EXHAUSTIVE_3WAY_BENCHMARK_REPORT.md:1-40` va bang chung goc `product/reports/clinical/exhaustive_3way_benchmark_results.json`

Lenh tai hien:

```
$ python -c "import json;d=json.load(open('product/reports/clinical/exhaustive_3way_benchmark_results.json',encoding='utf-8'));..."
/benchmark_models/author_sota_baseline/mIoU_4cls            = 0.0
/benchmark_models/author_sota_baseline/mDice_4cls           = 0.0
/benchmark_models/author_sota_baseline/Pixel_Accuracy_4cls  = 0.0
/benchmark_models/author_sota_baseline/FwIoU_4cls           = 0.0
/benchmark_models/author_sota_baseline/IoU_TUM..IoU_BACK    = 0.0 (5/5 lop)
/benchmark_models/author_sota_baseline/Dice_TUM..Dice_NEC   = 0.0 (4/4 lop)
/benchmark_models/author_sota_baseline/total_patches_evaluated      = 2
/benchmark_models/retrained_from_scratch/total_patches_evaluated    = 2
/benchmark_models/proposed_medsam_adapter/total_patches_evaluated   = 2
/statistical_significance/proposed_vs_author/t_statistic            = inf
/statistical_significance/proposed_vs_author/t_test_p_value         = 0.0
/statistical_significance/proposed_vs_author/wilcoxon_w_statistic   = 0.0
/statistical_significance/proposed_vs_author/wilcoxon_p_value       = 0.5
/statistical_significance/proposed_vs_author/cohen_d                = 0.0
/statistical_significance/proposed_vs_author/is_statistically_significant_05 = False
/clinical_survival_analysis/patient_cohort_count                    = 1
```

Trong khi header bao cao ghi ro:

> **Evaluation Protocol:** 100% Zero-Mockup Streaming Inference across 23 Active Test Patients

Va nhan du lieu that tren dia: `data/BCSS-WSSS/test/img` = **4,986 patch**, 23 benh nhan (toi da do truc tiep - xem T3).

Bon mau thuan khong the dong thoi la "ket qua that":

1. **`Pixel_Accuracy_4cls = 0.00%`** cho Model A. Mot mang conv khong the dat 0% pixel accuracy tren 5 lop (ngau nhien cung ~20-25%). Day la chu ky cua **mo hinh khong duoc nap trong so** hoac sai lech truc tiep canh tranh (xem T7).
2. **`total_patches_evaluated = 2`** -> bao cao "Exhaustive" va "23 Active Test Patients" la **khong dung le**. 2 patch thuoc 1-2 benh nhan khong the dai dien 4,986 patch.
3. **`t = inf` + `cohen_d = 0.000` + `p = 0.0` + `Wilcoxon p = 0.5`** tu mau chong che toan hoc: `cohen_d = mean_diff/std_diff` voi `std_diff > 0` thi `t = d * sqrt(n)` khong the bang `inf`. Bao cao in ra "95% Confidence Interval: [+5.01%, +5.01%]" - CI rong bang 0, dung la nhan `ci = [x, x]` khi `k_patients < 2` trong `src/colab_pkg/evaluator.py:190-191`.
4. **`is_statistically_significant_05 = False`** trong JSON, nhung bao cao Markdown lai trinh bay cot "Delta (C vs A) = +5.01%" nhu mot **ket lua cai thien**. Bao cao khong in ra `False` nay.

He qua lam sang/hoc thuat: day la bang ket qua cot loi cua do an (so sanh 3 mo hinh: Author SOTA vs Retrained vs Proposed MedSAM). Toan bo ket luan "Proposed vuot troi" da duoc xay tren mot cot baseline 0.00% va mau n = 2. Vi pham truc tiep INVARIANT-07 ("100% so lieu... phai chay thuc nghiem that") va INVARIANT-09 (bao co mu +/- sigma tren cohort that).

### 2.2 [T2-five-seed-table-no-log] Bang ablation 5 seed trong bao cao ky thuat thieu log doi chung

| Severity | **Critical** | Do tin cay | **Cao - VERIFIED** |
| :--- | :--- | :--- | :--- |

`product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md:190-197`

Bao cao ghi: *"Duoc ghi nhan that tu tensor weights trong tep `experiments/benchmarks/benchmark_results.json`"*.

```
$ ls experiments/benchmarks/
COLAB_3_FULL_RUNS_EVOLUTION_COMPARISON.md
README.md

$ find . -name "benchmark_results.json" -not -path "./repos/*"
(khong tra ve ket qua nao)
```

`src/pipeline/comparative_runner.py:164` chinh xac viet toi duong dan do (`os.path.join(output_dir, "benchmark_results.json")`, `output_dir` mac dinh `"experiments/benchmarks"`). => File do **chua tung duoc sinh ra**, hoac da bi xoa.

Ba dau hieu phi thuong trong bang:

| Cot | Gia tri | Van de |
| :--- | :--- | :--- |
| Stroma IoU | `0.00 +/- 0.00` o CA 4 cau hinh | Do lech chuan bang 0 tuyet doi -> khong co 5 gia tri khac nhau |
| Cau hinh 1 vs Cau hinh 2 | `6.32+/-4.96`, `8.80+/-6.38`, `27.93+/-22.43` **giong het den tung chu so** | Cau hinh 2 them ClassMix (tang cuong du lieu) ma khong lech mot chu so nao -> 2 dong la copy-paste, khong phai 2 run doc lap |
| VRAM dinh | `<50 MB` cho "ResNet-38 PDA + CGM" | Bat kha thi ve vat ly: `tests`/`src/colab_pkg/colab_runner.py:620` assertions xac nhan `m1` co **105,087,296** tham so. Rieng weights fp32 = ~400 MB. |

He qua: INVARIANT-07 yeu cau "moi bang ket qua phai co log thuc nghiem doi chung". Bang trung tam cua bao cao khong co log. Doc lap voi kha nang "co y che tao": toi ghi nhan bao cao **co** thong bao "Sanity Run"/"chua hoi tu sau" (dong 201-203) - tuc co y minh bach. Nhung minbach do khong the cuu mot bang co `std = 0.00` va VRAM vo ly.

### 2.3 [T2-verified-GOOD] Du lieu ket qua Colab that TON TAI va hop ly

De tranh ket luan oan "toan bo la gia", toi da doc log Colab that:

```
$ python -c "import json;print(json.dumps(json.load(open('experiments/colab_run_1_stage2_baseline/COLAB_EVALUATION_REPORT.json'))['metrics'],indent=1))"
"device": "cuda", "gpu_name": "Tesla T4"
"total_test_patches_in_dataset": 4986,  "evaluated_patches": 4986
"mIoU": 68.7061504800327,  "mDice": 81.15419784394602,  "Accuracy": 84.94484433787149
"per_class_iou": {TUM 79.12, STR 73.50, LYM 56.97, NEC 65.23, BACK 70.26}
"confusion_matrix_4cls": [[86410352, 11803803, 550567, 835236], ...]
```

Nhan xet: `mIoU 68.71%` **khop chinh xac** con so duoc trich dan trong cac notebook (`notebooks/SE374_Colab_WSSS_*_Full4986.ipynb` mdcell9). Ma tran nhan dat hang trieu pixel, hop ly ve kich thuoc. **Do la so lieu that.** Van de la: san pham nop (`product/reports/clinical/`) lai la ban 2-patch vo nghia, trong khi ban 4,986-patch that nam trong `experiments/` va khong duoc dua vao bao cao chinh. Day la **rui ro nop sai san pham**, khong phai che tao.

### 2.4 [T2-fake-gt-mask] Ground-truth gia khi mask thieu

| Severity | **Critical** | Do tin cay | **Cao - VERIFIED** (doc ma + tai lap hanh vi) |
| :--- | :--- | :--- | :--- |

`src/clinical/exhaustive_evaluator.py:431-439`

```python
def _load_ground_truth_mask(self, mask_path: Optional[Path], shape_hw: Tuple[int, int]) -> np.ndarray:
    """Loads ground truth mask or creates default tissue mask if missing."""
    if mask_path and mask_path.is_file():
        ...
        return mask_arr
    return np.full(shape_hw, fill_value=4, dtype=np.int64)
```

Ket qua duoc nap thang vao so do: `src/clinical/exhaustive_evaluator.py:486-494`

```python
self.eval_author.update(pred_author, gt_mask, latency_sec=lat_author, patient_id=pid)
self.eval_retrained.update(pred_retrained, gt_mask, ...)
self.eval_medsam.update(pred_medsam, gt_mask, ...)
```

Nghia la: mot patch **khong co nhan** duoc tinh Dice/mIoU chong lai mot mask "toan bo BACK" do code tu ve. Doc solution ghi "creates default tissue mask" - thuc chat la **che tao chu thich ho ly hoc (annotation fabrication)** - vi pham nghiem trong nhat trong anh y khoa. He qua lam sang: Dice lop TUM bi keo ve 0 mot im lang, trong khi so patch "duoc danh gia" van tang len, khong co truong nao dem so patch thieu GT.

### 2.5 [T2-zero-supervision-fallback] Pseudo-mask toan-zero khi `train_PM` rong - TAI LAP HANH VI

| Severity | **Critical** | Do tin cay | **Cao - VERIFIED (chay that)** |
| :--- | :--- | :--- | :--- |

`src/training/dataset_loader.py:323-325`

```python
elif self.fallback_mask:
    # Fallback to zeros (class 0 or background) if layer mask is missing
    return np.zeros((224, 224), dtype=np.int64)
else:
    raise FileNotFoundError(f"Pseudo-mask not found at: {mask_path}")
```

`fallback_mask: bool = True` la **mac dinh** (`src/training/dataset_loader.py:272`). Chu thich sai: voi bang BCSS, lop `0` la **TUM (Khoi u)** khong phai background (`src/datasets/bcss_dataset.py:17-22`: `0: TUM, 1: STR, 2: LYM, 3: NEC, 4: BACK`). `np.zeros` = "moi pixel la khoi u".

Trang thai dia that:

```
$ for d in train_PM/PM_bn7 train_PM/PM_b5_2 train_PM/PM_b4_5; do echo -n "$d : "; ls "$d" | wc -l; done
train_PM/PM_bn7 : 0
train_PM/PM_b5_2 : 0
train_PM/PM_b4_5 : 0
```

Lenh tai lap hanh vi (read-only, khong ghi de):

```
$ KMP_DUPLICATE_LIB_OK=TRUE python - <<'PY'
... BCSSStage2MultiMaskDataset(img_dir="data/BCSS-WSSS/training", pm_root_dir="train_PM",
                               patient_ids=sp["train"], max_samples=3)
... b = next(iter(DataLoader(st2, batch_size=1)))
PY
STAGE2 with EMPTY train_PM -> label uniq: [0]  shape (1, 224, 224)
```

-> **DUOC CHUNG MINH HANH VI**: neu chay `train_stage2.py` hom nay tren may nay, DeepLabV3+ se huan luyen voi toan bo nhan gian lan la "khoi u", in ra mot con so mIoU, va khong in ra bat ky canh bao nao. Kiem tra them: khong co `print`/`warnings.warn` trong nhan `elif self.fallback_mask`.

Cung mot kieu: `src/datasets/bcss_dataset.py:132-133`

```python
else:
    mask_tensor = torch.zeros((img_np.shape[0], img_np.shape[1]), dtype=torch.int64)
```

`src/training/dataset_loader.py:456-457` (`BCSSEvalDataset`):

```python
else:
    # Fallback zero mask if mask folder is absent
    mask_np = np.zeros((224, 224), dtype=np.int64)
```

Nhan xet them: cung 3 noi hard-code kich thuoc `(224, 224)` - neu patch that khong phai 224x224 se sinh shape mismatch hoac gian lan sai ti le ma khong bao gio duoc canh bao.

### 2.6 [T2-invented-weak-label] Nhan yeu bi "che" khi ten file thieu duoi `[]`

| Severity | **High** | Do tin cay | **Cao - VERIFIED** |
| :--- | :--- | :--- | :--- |

`src/training/dataset_loader.py:105-108`

```python
    else:
        # Default label if brackets are absent
        multi_label[0] = 1.0
```

WSSS chi co nhan cap patch (multi-hot 4 lop). Khi ten file khong co `[1 0 0 1]`, code gan `multi_label[0] = 1.0` = **khoi u duong tinh**. Day la bịa mot nhan giam sat va dua vao huan luyen Stage-1 classifier. Cung ham `parse_patch_filename` duoc goi trong `__getitem__` cua `BCSSStage1Dataset` (`:229`) va `BCSSStage2MultiMaskDataset` (`:333`), nen mot file anh ten lahet qui uoc im lang tro thanh mau "TUM duong tinh". Cach xu ly dung: nem `ValueError` hoac loai mau va ghi log.

Cung kieu do: `src/datasets/bcss_dataset.py:82-92` tra ve `np.zeros(4)` (tat ca lop am tinh) khi ten file khong khop regex - cung la mot gia dinh nhan, huong nguoc lai, va cung im lang.

### 2.7 [T2-hardcoded-cohort-fallback] Cohort benh nhan hard-code khi manifest thieu

| Severity | **High** | Do tin cay | **Cao - VERIFIED** |
| :--- | :--- | :--- | :--- |

`src/training/dataset_loader.py:126-133`

```python
    if not config_path.exists():
        # Fallback to local default patient lists if config file does not exist
        return {
            "train": ["TCGA-A1-A0SK", "TCGA-A2-A04N", "TCGA-A2-A0T2"],
            "val": ["TCGA-AN-A03X", "TCGA-AN-A0AL"],
            "test": ["TCGA-AQ-A04H", "TCGA-C8-A1HJ"],
        }
```

Manifest 151 benh nhan le duoc thay the bang **7 benh nhan tinh gion**, khong log, khong exception. Luong do lac tren con **bo qua hoan toan `verify_patient_leakage()`** (ham nay chi duoc goi trong `src/datasets/patient_split.py:147`, khong phai day). Neu `bcss_patient_splits.json` bi loi duong dan tren Colab (rat de xay ra vi `Path(__file__).resolve().parent.parent.parent` phu thuoc cach clone repo), toan bo thi nghiem "151 benh nhan" se chay tren 7 benh nhan va bao cao so do.

### 2.8 [T2-synthetic-training-fallback] Huan luyen tren du lieu Synthetic khi `data_dir` thieu

| Severity | **High** | Do tin cay | **Cao - VERIFIED** |
| :--- | :--- | :--- | :--- |

`src/training/train_proposed_sam.py:485-494`

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

`SyntheticHistologyDataset` (`src/training/train_proposed_sam.py:196-218`) sinh `torch.rand` anh va `torch.randint(0,5,...)` mask **gau nhien**. Dieu kien `else` bao gom `not os.path.exists(args.data_dir)` khi `args.dry_run == False`. Nghia la: mot lenh `train_proposed_sam.py` voi duong dan bi sai (ty po, chiu anh huong cua `/content/`) se **im lang huan luyen tren nhieu gau nhien va in ra `Val mIoU_fg=...`** (dong 530-534) - mot con so hoan toan vo nghia ve lam sang.

Co mot dong `[INFO] Using SyntheticHistologyDataset` in ra, day la diem cong. Nhung no khong phai la chan ngat (gate): chuong trinh khong `raise`, khong ghi co che "SYNTHETIC" vao checkpoint JSON.

### 2.9 [T2-checkpoint-surrogate] Mo hinh trong so ngau nhien van duoc dua vao bang benchmark

| Severity | **Critical** | Do tin cay | **Cao - VERIFIED (doc + giai thich chu ky 0.00%)** |
| :--- | :--- | :--- | :--- |

`src/clinical/exhaustive_evaluator.py:318-347`

```python
    def _build_deeplabv3plus(self, checkpoint_path, tag):
        ckpt_p = Path(checkpoint_path)
        if ckpt_p.is_file():
            try:
                ...
                model.load_state_dict(cleaned, strict=False)
                print(f"[EVALUATOR] Loaded checkpoint for {tag} from {ckpt_p}")
                model.to(self.device); model.eval(); return model
            except Exception as e:
                print(f"[EVALUATOR] Warning: Could not load {ckpt_p} into DeepLabV3Plus ({e}).")

        # Standard DeepLabV3PlusWithCGM initialization
        model = DeepLabV3PlusWithCGM(num_classes=5, in_channels=3)
        model.to(self.device); model.eval(); return model
```

Ba loi chong chat:
1. `except Exception` bat **moi** loi (thieu file, sai kieu, khong duong tensor) roi **rơi xuong** nhanh mo hinh **khoi tao trong so ngau nhien**.
2. Fast-path dung `strict=False`, ma `strict=False` **im lang bo qua missing_keys / unexpected_keys** -> trong so khong khớp van nguyen gia tri khoi tao.
3. **Khong co bat ky co che nao** ghi lai trang thai "checkpoint nay co that duoc nap hay khong". Toi da kiem:

```
$ grep -rn "is_surrogate" --include=*.py src/
src/models/sam_wsss_adapter.py:745  self.is_surrogate = True
src/models/sam_wsss_adapter.py:755  self.is_surrogate = False
src/models/sam_wsss_adapter.py:761  self.is_surrogate = True
src/models/sam_wsss_adapter.py:765  self.is_surrogate = True

$ grep -rn "untrained|randomly_init|checkpoint_loaded|fallback_used|surrogate" src/clinical/exhaustive_evaluator.py
(chi thay 2 dong docstring/print, KHONG co truong nao trong ket qua JSON)
```

-> `is_surrogate` duoc **dat ma khong bao gio duoc doc** (dead flag). Bang ket qua JSON khong luu trang thai nap trong so.

**Day la loi giai thich chinh xac nhat cho `Pixel_Accuracy_4cls = 0.00%` cua Model A** o muc 2.1. Model "Author SOTA" trong bang nop Giang vien rat co the la mot mang resnet-duoc-khoi-tao-ngau-nhien, khong phai checkpoint chinh thuc cua Chu Han et al.

### 2.10 Cac nghi van DA LOAI (khong phai loi) - ghi lai theo yeu cau

| Nghi van | Ket luan sau khi mo doc ngu canh |
| :--- | :--- |
| `src/colab_pkg/colab_runner.py:656` `np.random.randint(40,220,(32,32,3))` | **AN TOAN.** Nam trong `run_package_self_test()` (`:525`), duoc goi bo co qua CLI flag `--self-test` (`src/pipeline/colab_runner.py:160,181`). Ghi `mode_tag="SelfTest_Sanity"` va dung `tempfile.TemporaryDirectory()`. Khong len ket qua vao bao cao. |
| `src/colab_pkg/evaluator.py:221` va `:492` `np.random.RandomState(seed)` | **AN TOAN.** Bootstrap resampling tren patient confusion matrices - phuong phap thong ke chuan, co dinh seed = 42. |
| `src/models/coarse_to_fine_router.py:373` `rng_t = np.random.RandomState(seed + t_idx*97)` | **AN TOAN.** Sinh nhieu test-time augmentation co dinh. |
| `src/clinical/exhaustive_evaluator.py:661,699,719,723` | **TICH CUC.** Chuoi `if` thieu du lieu survival ghi `"status": "Insufficient stratified cohorts for logrank test; no dummy patients injected."` thay vi che `15.0 months` - dung INVARIANT-07. |
| `src/colab_pkg/downloader.py:119-153` `_is_mock_path`, `ignored_repo_dirs` | **TICH CUC.** Tu choi `sample_dataset` va `repos/` khi tim du lieu. |
| `src/utils/deliverable_auditor.py:452-462` `dummy_markers` | **AN TOAN.** Day la danh sach scan, khong phai marker thuc. |
| `src/visualization/static/index.html:499` `createSyntheticHnEDataURL` | **Low.** Placeholder demo cho viewer, gan nhan ro "synthetic sample... for instant clinical testing", khong sinh metric. |
| `src/training/train_proposed_sam.py:254-259` `surrogate_cams = torch.zeros(...)` | **AN TOAN trong ngan canh.** Sinh CAM dai dien tu nhan yeu khi CAM khong duoc tinh truoc - co so thuoc cua pipeline WSSS (ClassMix-style), khong phai bịa metric. |
| Cac `.md` trong `product/proposal/` chua tu "gia dinh" | **AN TOAN.** Tyeng "mat na gia dinh (pseudo-mask)" la thuat ngu WSSS chuan, khong phai so lieu gia. |
| `notebooks/SE374_Colab_WSSS_*.ipynb` "mIoU 68.71%" | **AN TOAN + CO CHUNG MINH.** La trich dan baseline tac gia, va **khop chinh xac** `experiments/colab_run_1_stage2_baseline/COLAB_EVALUATION_REPORT.json` mIoU 68.7061. |

---

## 3. T3 - Patient-Level Anti-Leakage [INVARIANT-08]

### 3.1 PHAN TINH TICH CUC: kiem chung tren du lieu THAT (Anti-Mock-Testing Gate)

Bao cao cu chi doi chieu file JSON manifest. Toi da do truc tiep 31,826 anh tren dia:

```
$ KMP_DUPLICATE_LIB_OK=TRUE python -X utf8 - <<'PY'
> tu src.datasets.patient_split import extract_patient_id
> quy don vi vao extract_patient_id cua tung ten file trong data/BCSS-WSSS/{training,val/img,test/img}
PY
training: 23422 patches, 106 unique patient IDs
val/img :  3418 patches,  22 unique patient IDs
test/img : 4986 patches,  23 unique patient IDs
train^val overlap : 0 []
train^test overlap: 0 []
val^test overlap  : 0 []
--- manifest vs filesystem ---
manifest train 106 val 22 test 23
fs-test patients NOT in manifest test:              0 []
manifest test NOT present in fs test dir:           0 []
fs-training patients that ARE in manifest test:     0 []
fs-training patients that ARE in manifest val:      0 []
fs-val patients in manifest test:                   0
```

**KET LUAN: 0.0% ro ri tren du lieu BCSS-WSSS that. Tuyen bo cu DUNG ve ket qua** (106+22+23 = 151 benh nhan, disjoint hoan toan, khớp 100% voi manifest). Day la bang chung HANH VI, not test-green.

Chung minh duong nap that cung chay duoc:

```
$ python -X utf8 - <<'PY'  (BCSSDataset + DataLoader batch_size=2)
BCSSDataset test len: 4986
REAL BATCH OK: (2, 3, 224, 224) (2, 224, 224) mask classes: [1, 2]
Stage1 filtered by manifest patients: 8
Eval batch: (2, 3, 224, 224) (2, 224, 224) GT uniq: [1, 1, 2]
```

### 3.2 [T3-manifest-not-enforced-in-primary-loader] `BCSSDataset` khong he consult manifest

| Severity | **Medium** | Do tin cay | **Cao - VERIFIED** |
| :--- | :--- | :--- | :--- |

`src/datasets/bcss_dataset.py:48-79` (ca ham `_load_dataset`), va noi dung: `src/training/train_proposed_sam.py:488-489`, `src/pipeline/comparative_runner.py:124`

```python
    def _load_dataset(self) -> None:
        is_train_split = self.split in ["train", "training"]
        if is_train_split:
            train_dir = os.path.join(self.root_dir, "training")
            ...
            pattern = os.path.join(train_dir, "*.png")
            self.image_paths = sorted(glob.glob(pattern))       # <-- KHONG co patient filter
        else:
            img_dir = os.path.join(self.root_dir, split_name, "img")
            ...
            self.image_paths = sorted(glob.glob(pattern))       # <-- KHONG co patient filter
```

Ham `__init__` cua `BCSSDataset` (`:29-46`) **khong co tham so `patient_ids`** va khong goi `load_patient_splits()`. Split hoan toan duoc dinh nghoi boi **cấu trúc thư mục**, không bởi manifest.

Lenh chung minh:

```
$ grep -rn "load_patient_splits|bcss_patient_splits|verify_patient_leakage" --include=*.py src/ | grep -v patient_split.py
src/visualization/api_server.py:25   from src.datasets.patient_split import load_patient_splits
src/visualization/api_server.py:416  return load_patient_splits()
src/training/dataset_loader.py:501   splits = load_bcss_patient_splits(splits_config)
src/training/train_stage2.py:286     splits = load_bcss_patient_splits()
src/clinical/survival_analysis.py:279 config_path = project_root/"src"/"config"/"bcss_patient_splits.json"
(khong co dong nao trong src/datasets/bcss_dataset.py)
```

He qua: manifest "151 benh nhan disjoint" chi la **document trang tri** doi voi nganh `train_proposed_sam.py` (Mo hinh Đề xuất - Model C trong bang benchmark!) va `comparative_runner.py` (cong cu ablation 5 seed). Neu nguoi dung thay `data/BCSS-WSSS` bang mot ban BCSS khac ma `training/` chua patch cua benh nhan test, pipeline de xuat sẽ **im lặng học trên bệnh nhân test** và không test nào phát hiện.

Do do: `tests/test_patient_split.py::test_151_patient_split_manifest_zero_leakage` la **test tyr tham chieu** (kiem tra file JSON voi chinh no) - no khong the phat hien ro ri o tang nap du lieu. Test cua toi o muc 3.1 moi la bang chung that.

`tests/test_patient_split.py` - danh gia tong the:
- `test_verify_patient_leakage_catches_overlap` (`:52-60`) - **TOT**, adversarial that, dung `pytest.raises(ValueError)`.
- `test_bcss_dataset_loader_dry_run` (`:74-91`) - **TOT**, nap batch that tu `data/sample_dataset`, assert hinh dang tensor.
- `test_split_patients_kfold` (`:63-71`) - **TOT ve logic nhung doi tuong chay la `["TCGA-P00"..]` tyr sinh**, va `split_patients_kfold` khong duoc dung boi bat ky pipeline huan luyen nao (chi duoc re-export trong `src/datasets/__init__.py`).
- Hypothesis fuzz (`:162-174`) - **VAI TRO**. `test_hypothesis_extract_patient_id_never_crashes` chi assert `isinstance(result, str)`. Voi loi tho `return token.strip()`, no khi nao cung tra ve str -> duoc 100% ngay ca khi regex TCGA hoan toan bi hong.
- `test_extract_patient_id_tcga` (`:17-29`) - **CO LUI SU THAT**: assert `extract_patient_id("GDPH_1234_patch_05.png") == "GDPH"` (dong 29), trong khi docstring cua ham (`src/datasets/patient_split.py:14`) ghi ro `GDPH_1234_patch_01.png -> 'GDPH_1234'`. Test "dong bang docString" bang cach ma hoa loi: `extract_patient_id` split theo `[_+]` dau tien (`:25`) nen GDPH_* bi cham thanh "GDPH" -> **moi patch GDPH cua moi benh nhan khac nhau co cung mot patient ID**, nghia la GDPH/LUAD-HistoSeg bi gep thanh 1 "benh nhan". Test da khoa chet mot loi phan loai thuc su va bao co no.

### 3.3 [T3-survival-cohort-stale] Cohort survival hard-code 23 benh nhan

| Severity | **Medium** | Do tin cay | **Cao - VERIFIED** |
| :--- | :--- | :--- | :--- |

`src/clinical/survival_analysis.py:282-292`

```python
    p = Path(config_path)
    if p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "test_active_patients" in data:
                return list(data["test_active_patients"])
            if "splits" in data and "test_active" in data["splits"]:
                return list(data["splits"]["test_active"])
        except Exception:
            pass

    return list(ACTIVE_TEST_PATIENTS_23)
```

Neu JSON bi hong/thieu khoa, chuong trinh tra ve danh sach hard-code va **khong bao**. Manifest hien tai chua `test_active_patients` (23 phan tu, toi da doc) -> duong fast-path dang chay dung. Rui ro la khi manifest duoc cap nhat (them benh nhan), hard-code list tro thanh stale va bao cao lam sang se phan lop sai benh nhan.

### 3.4 Cac diem chia train/val/test khac - da quet

```
$ grep -rn "train_test_split|GroupKFold|Stratified|shuffle=True" --include=*.py src/
src/training/train_proposed_sam.py:496    train_loader = DataLoader(train_ds, batch_size=..., shuffle=True)
src/training/train_stage1.py / train_stage2.py: shuffle o tang DataLoader, KHONG phai tang split
```

`shuffle=True` o day chi la **xao tron trong mot split da chia xong** ( Dung, chuan PyTorch), khong phai xao tron giua cac split. **KHONG tim thay** `sklearn.model_selection.train_test_split` hoac `GroupKFold` o bat ky dau trong `src/` - tat ca cac split deu theo thu muc/manifest. **DIEM SACH** cua T3 ngoai 3 loi tren.

---

## 4. T4 - Anti-Shallow-Glob [INVARIANT-29]

INVARIANT-29 dua ra HAI yeu cau bat buoc. Ket qua: **ca hai deu vi pham o nhieu noi.**

### 4.1 Yeu cau 1 - Auto-Discovery da tang (`rglob` / `resolve_split_path`)

| Vi tri | Ma lenh | Danh gia |
| :--- | :--- | :--- |
| `src/training/dataset_loader.py:163` | `for p in root_path.rglob("*")` | **DAT** - `find_image_files(..., recursive=True)` |
| `src/colab_pkg/downloader.py:146` | `base_path.glob(f"**/{sp}/img")` | **DAT** - co `resolve_split_dirs` that |
| `src/colab_pkg/visualizer.py:416` | `sorted(out_p.rglob("*"))` | **DAT** |
| `src/datasets/bcss_dataset.py:55` | `sorted(glob.glob(os.path.join(train_dir, "*.png")))` | **KHONG DAT** - nong |
| `src/datasets/bcss_dataset.py:72` | `sorted(glob.glob(os.path.join(img_dir, "*.png")))` | **KHONG DAT** - nong |
| `src/clinical/exhaustive_evaluator.py:394` | `sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))` | **KHONG DAT** - nong, dung chinh xac van mau `img_dir.glob("*.png")` ma INVARIANT-29 cam ten |
| `src/colab_pkg/evaluator.py:1079-1081` | `sorted(real_v_img.glob("*.png"))[:cap]` | **KHONG DAT** |
| `src/colab_pkg/evaluator.py:1085` | `all_real = sorted(real_t_img.glob("*.png"))` | **KHONG DAT** |
| `src/colab_pkg/evaluator.py:1240` | `sorted(real_t_img.glob("*.png"))[:50]` | **KHONG DAT** |
| `src/colab_pkg/evaluator.py:1245` | `sorted(cand_img_dir.glob("*.png"))[:50]` | **KHONG DAT** |
| `src/colab_pkg/evaluator.py:1005` | `m_candidates = list(m_dir.glob(f"{p.stem}.*"))` | **CHAP NHAN DUOC** (tra cứu theo stem trong mot thu muc cu the) |
| `src/colab_pkg/downloader.py:134` | `pngs = list(p.glob("*.png"))` | **CHAP NHAN DUOC** (`_get_dir_stats` tren img-dir da duoc phien dich) - nhung luu y: neu anh nested trong `test/img/img/` thi dem ra 0 va loai ung vien |
| `src/colab_pkg/downloader.py:354-355`, `:543-544`, `:632`, `:658`, `:663`, `:703` | `.glob("*.png")` | **NONG** nhung o sau `resolve_split_dirs` -> rui ro thap |

`[T4-shallow-glob-primary-loader]` | **High** | Do tin cay: Cao (VERIFIED) | `src/datasets/bcss_dataset.py:55`, `:72`; `src/clinical/exhaustive_evaluator.py:394`; `src/colab_pkg/evaluator.py:1079`, `:1085`, `:1240`, `:1245`

Nhan manh: `BCSSDataset` **co** mot it phong thu mot phan (doi `train`<->`training` `:51-53`, doi `val`<->`valid` `:66-70`) va `exhaustive_evaluator` co phong thu `test/img`<->`test` (`:389-392`), nhung **khong noi nao dung `rglob`** va **khong noi nao duoc thiet ke cho `BCSS-WSSS/train/img/*.png` long nheu** ma INVARIANT-29 neu ten cu the.

### 4.2 Yeu cau 2 - Safety-net `len(files) == 0` -> placeholder (`len(dataset) >= 1`)

**KHONG TON TAI o BAT KY dau.**

```
$ grep -rn "num_samples|placeholder|smoke-test|len(files) == 0|>= 1" --include=*.py src/datasets/ src/training/dataset_loader.py src/colab_pkg/evaluator.py
src/datasets/otsu_filter.py:84: if tissue_mask.size >= 1024 and np.any(tissue_mask == 1):
(khong co ket qua nao khac)
```

`len(self.files)` tra ve 0 mot im lang trong `BCSSStage1Dataset.__len__` (`:230`), `BCSSStage2MultiMaskDataset.__len__` (`:308`), `BCSSEvalDataset.__len__` (`:434`), `BCSSDataset.__len__` (`:94`).

He qua: tren Colab, neu `data/BCSS-WSSS.zip` tai chưa xong hoac mang gian doan, DataLoader nem `ValueError: num_samples should be a positive integer value, but got num_samples=0` - chinh xac la dieu INVARIANT-29 tuyet doi cam. Toan bo nganh `BCSSDataset` (loader cua Model De xuat) cung khong co safety net.

### 4.3 Nghia vu liem chinh: safety net co phai la mot vi pham T2?

**Cau tra loi: CO, va day la mot mau thuan thiet ke can nguoi dung quyet.**

INVARIANT-29 yeu cau placeholder khi rong; INVARIANT-07 cam so lieu tu du lieu gia. Neu thuat toan placeholder duoc thiet ke va dung **dung cach** (chi de gi cho DataLoader khong crash trong `mode="smoke"`, co co che danh dau `synthetic=True`, va bi chan ngat khi ghi bao cao) thì cả hai đều thỏa. Hiện tại repo rơi vào tình huống **tệ nhất của cả hai**: safety net absent (vi phạm INVARIANT-29), nhưng **fallback dữ liệu giả thì CÓ và không bị chặn** (`np.zeros((224,224))` ở 2.5, `SyntheticHistologyDataset` ở 2.8, cohort 7 bệnh nhân ở 2.7) - tức các chỗ tạo dữ liệu giả đều không có cờ đánh dấu, còn chỗ cần cờ đánh dấu (chống sập) thì lại không tồn tại.

---

## 5. T5 - Colab Lifecycle [INVARIANT-28] + [INVARIANT-27]

### 5.1 [T5-drive-mount] 23 notebook vi phạm INVARIANT-28

INVARIANT-28: "TUYET DOI NGHIEM CAM viec tu y dua lenh `drive.mount("/content/drive")`".

```
$ grep -rlP "drive\.mount|/content/drive" --include=*.ipynb . | grep -v "/repos/" | wc -l
23

$ grep -rcP "drive\.mount" --include=*.ipynb . | grep -v ":0" | grep -v "/repos/"
./drive_upload_ready/SE374_WSSS_Drive_Code/notebooks/history_runs/Run3_20261020_2055UTC_..._T4_Full4986.ipynb:1
./drive_upload_ready/SE374_WSSS_Drive_Code/notebooks/SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb:1
./drive_upload_ready/SE374_WSSS_Drive_Code/notebooks/SE374_Colab_WSSS_A100_Full4986_ThinDriveRunner_v2.ipynb:1
./drive_upload_ready/SE374_WSSS_Drive_Code/notebooks/SE374_Colab_WSSS_L4_Full4986_ThinDriveRunner_v2.ipynb:1
./drive_upload_ready/SE374_WSSS_Drive_Code/notebooks/SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb:1
./drive_upload_ready/SE374_WSSS_Drive_Code/notebooks/SE374_Colab_WSSS_T4_Sanity150_ThinDriveRunner_v2.ipynb:1
./drive_upload_ready/SE374_WSSS_Drive_Code/SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb:1
./drive_upload_ready/SE374_WSSS_Drive_Code/SE374_Colab_WSSS_{A100,L4,T4}_Full4986_ThinDriveRunner_v2.ipynb:1 (x3)
./drive_upload_ready/SE374_WSSS_Drive_Code/SE374_Colab_WSSS_T4_Sanity150_ThinDriveRunner_v2.ipynb:1
./notebooks/history_runs/Run3_20261020_2055UTC_..._T4_Full4986.ipynb:1
./notebooks/modular_drive/SE374_Colab_WSSS_{A100,L4,T4}_Full4986_ThinDriveRunner_v2.ipynb:1 (x3)
./notebooks/modular_drive/SE374_Colab_WSSS_T4_Sanity150_ThinDriveRunner_v2.ipynb:1
./notebooks/SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb:1
./notebooks/SE374_Colab_WSSS_{A100,L4,T4}_Full4986_ThinDriveRunner_v2.ipynb:1 (x3)
./notebooks/SE374_Colab_WSSS_T4_Sanity150_ThinDriveRunner_v2.ipynb:1
./SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb:1
./SE374_Colab_WSI_LowMag_HighMag_BonusDemo_v2.ipynb:1
```

Trich doan ngu canh that (`SE374_Colab_WSI_LowMag_HighMag_BonusDemo_v2.ipynb`, cell 2):

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
...
print(f'[CANH BAO] Khong mount duoc Drive: {e}')
# 2. Tim kiem thu muc ma nguon SE374_WSSS_Drive_Code tren Google Drive (Kich ban A, B, C)
MYDRIVE = Path('/content/drive/MyDrive')
```

**Phan bien khach quan:** co 2 yeu to gim nhe severity:
1. Code **co** fallback GitHub clone (`# Kich ban D: Neu chua co tren Drive, tu dong clone truc tiep tu GitHub`) - dung mot phan tinh than INVARIANT-28.
2. `drive.mount` duoc goi trong `try:` voi `[CANH BAO]` khi that bai, khong phai lenh truc tiep khong bao che.

**Nhung vi pham van con hieu ly:** INVARIANT-28 cam *dua lenh `drive.mount` vao notebook*, khong cam "lenh mount khong co try/except". `drive.mount()` **dung thuc thiet cho toi khi nguoi duyet mo pop-up OAuth va cap quyen** - tuc nghiep vu "Run All chay 1 mach tu dau den cuoi 100% tu dong, khong bao gio dung lai doi cap quyen" bi pha. Toan bo bon `ThinDriveRunner_v2` (T4/L4/A100/Sanity150) - ten goi gan voi "Drive" - deu dung lenh nay.

`[T5-drive-mount]` | **High** | Do tin cay: Cao (VERIFIED lenh + doc ngu canh) | 23 tep nhu tren

### 5.2 [T5-seven-step-lifecycle] Vong doi 7 buoc - vi pham theo notebook

Do phap: doc JSON, dem so luong theo 10 chuoi dau roi.

| Notebook | 1 GPU | 2 pip | 3 du lieu that | 4 weights | 5 mIoU/Dice | 6 truc quan | 7 ZIP | 7 `files.download` | `drive.mount` |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `01_Train_Stage1_Classifier*.ipynb` (4 ban) | 2 | **0** | 1 | 2 | **0** | 2 | 3 | 1 | 0 |
| `02_Pseudo_Masks_Train_Stage2*.ipynb` (4) | 2 | **0** | 1 | 2 | 2 | 2 | 3 | 1 | 0 |
| `03_Train_Proposed_MedSAM*.ipynb` (4) | 2 | **0** | **0** | 2 | 2 | **0** | 3 | **0** | 0 |
| `04_Exhaustive_Benchmark_Survival*.ipynb` (4) | 2 | **0** | **0** | 2 | 4 | 2 | 3 | **0** | 0 |
| `SE374_Colab_Scientific_Research_Runner.ipynb` | **0** | **0** | 2 | **0** | 4 | **0** | 3 | 1 | 0 |
| `SE374_Colab_WSI_..._BonusDemo{,_v2}.ipynb` (2) | 1 | 1 | 2 | **0** | **0** | 3 | 3 | 1 | **1** |
| `SE374_Colab_WSSS_{T4,L4,A100,Pipeline}.ipynb` | 3 | 1 | 2 | 2 | 4 | 3 | 3 | 1 | 0 |
| `SE374_Colab_WSSS_T4_Sanity150.ipynb` | 3 | 1 | 2 | 2 | 4 | 3 | 3 | 1 | 0 |
| `SE374_Colab_WSSS_*_ThinDriveRunner_v2.ipynb` (4) | 1 | **0** | **0** | **0** | 2 | **0** | 3 | **0** | **1** |

Xac minh doc lap bang cach dot grep cho 4 notebook nghi van nhat:

```
notebooks/03_Train_Proposed_MedSAM_T4.ipynb
  files.download: 0   drive.mount: 0   imshow: 0   matplotlib: 0   cuda.is_available: 5  torch.load: 2
notebooks/04_Exhaustive_Benchmark_Survival_T4.ipynb
  files.download: 0   drive.mount: 0   imshow: 0   matplotlib: 1   cuda.is_available: 3  torch.load: 3
notebooks/SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb
  files.download: 0   drive.mount: 1   imshow: 0   matplotlib: 0   cuda.is_available: 1  torch.load: 0
notebooks/SE374_Colab_Scientific_Research_Runner.ipynb
  files.download: 1   drive.mount: 0   imshow: 0   matplotlib: 0   cuda.is_available: 0  torch.load: 0
```

`[T5-lifecycle-gaps]` | **High** | Do tin cay: Cao (VERIFIED) | Cac tep tren

Cac vi pham cu the theo INVARIANT-27:
1. **`03_Train_Proposed_MedSAM*.ipynb` thieu BUOC 6 (trực quan hóa lâm sàng) và BUOC 7 (tu dong tai ve).** Day la notebook cua **Mo hinh De xuat** - chu de chinh cua do an. Khong co `imshow`/`matplotlib` nao -> khong co heatmap/CAM overlay cho bac si. Khong co `files.download` -> adapter checkpoint (`checkpoints/medsam_wsss_adapter_best.pth`, 86 MB, hien ton tai tren dia) bi **nham trong may ao Colab** khi phien ket thuc.
2. **`04_Exhaustive_Benchmark_Survival*.ipynb` thieu BUOC 7.** Bao cao benchmark + `kaplan_meier_survival_curve.png` khong tu tai ve.
3. **Ca 12 notebook `01`-`04` thieu BUOC 2 (cai thu vien)** - `pip install` = 0. Chung dua vao `sys.path.insert` tu mot zip tai ve, tuc **phai co buoc tai ma nguon thanh cong truoc**; neu buoc do hong, notebook fail voi ModuleNotFoundError o cell 1.
4. **Ca 4 `ThinDriveRunner_v2` thieu BUOC 2, 3, 4, 6, 7** va dung BUOC drive.mount - day la nhiep vu vi pham INVARIANT-27 nang nhat (mot "Thin Runner" hoan toan khong min bach tu kiem soat du lieu).
5. **`SE374_Colab_Scientific_Research_Runner.ipynb` thieu BUOC 1** - `cuda.is_available` = 0, `torch.load` = 0; no goi vao `src.colab_pkg` nen buoc 1/4 nam sau hidden module. Ghi nhan la **rui ro thap hon** so voi ve ngoai, nhung van vi pham yeu cau "kiem tra GPU" cong khai.

### 5.3 So sanh hanh T4 / L4 / A100 / v2 / root-vs-notebooks (drift)

`md5sum` 24 notebook (`notebooks/` vs goc vs `drive_upload_ready/`):

| File | `notebooks/` | goc | Ket luan |
| :--- | :--- | :--- | :--- |
| `01_Train_Stage1_Classifier.ipynb` | `c635f0828c6e` | `7b375fde6160` | **LECH** |
| `01_Train_Stage1_Classifier_L4.ipynb` | `0f5073af9041` | `8d49792b009f` | **LECH** |
| `01_Train_Stage1_Classifier_T4.ipynb` | `c7d96317fdf0` | `c7d96317fdf0` | trung |
| `02_Pseudo_Masks_Train_Stage2_T4.ipynb` | `0410d06713d9` | `0410d06713d9` | trung |
| `03_Train_Proposed_MedSAM_T4.ipynb` | `4f37af83e358` | `4f37af83e358` | trung |
| `04_Exhaustive_Benchmark_Survival_T4.ipynb` | `95d2abe6e783` | `95d2abe6e783` | trung |
| `SE374_Colab_Scientific_Research_Runner.ipynb` | `87bfb7a2cf5a` | `87bfb7a2cf5a` | trung |
| `SE374_Colab_WSI_..._BonusDemo.ipynb` | `266f6763d5a0` | `05e42e21a869` | **LECH (chi o cell outputs)** |
| `02_/03_/04_` ban khong phai T4 (`*_L4.ipynb`, chan `L4`/`.ipynb`) | co trong `notebooks/` | **khong ton tai o goc** | **ma nguon co, ban goc khong** |
| `01_Train_Stage1_Classifier_v2.ipynb` | **khong co** | `co o goc` | **orphan** |

Phan tinh noi dung (khong phai metadata), `python -X utf8` doi chieu tung `cell.source`:

```
01_Train_Stage1_Classifier.ipynb: root cells=17 notebooks cells=17
   differing cell idx: [8, 10, 12, 14, 16]
01_Train_Stage1_Classifier_L4.ipynb: root cells=17 notebooks cells=17
   differing cell idx: [0, 8, 10, 12, 14, 16]
SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb: root cells=17 notebooks cells=17
   differing cell idx: []
```

`[T5-notebook-content-drift]` | **Medium** | Do tin cay: Cao (VERIFIED) | `01_Train_Stage1_Classifier.ipynb` goc vs `notebooks/` khac **5 cell code** (8,10,12,14,16); `_L4` khac **6 cell** (0,8,10,12,14,16). BonusDemo khac byte nhung **source identical** (chi khac outputs) -> khong phai loi.

He qua: 2 ban copy cua *cung mot thi nghiem Stage-1* co sieu tham so / logic khac nhau, va 8 notebook baseline+L4 chi ton tai o mot trong hai cho. Khong co mot "nguồn thật" nào. Khi ngiờn viên tải `01_Train_Stage1_Classifier.ipynb` từ gốc repo nhưng bạn trong `notebooks/` đã đổi, số liệu không tái lập.

**Diem SACH cua 5.3:** doi chieu `src/` vs `drive_upload_ready/SE374_WSSS_Drive_Code/src/`:

```
$ diff -rq src drive_upload_ready/SE374_WSSS_Drive_Code/src | grep -v __pycache__
Only in src/config: .gitkeep
Files src/config/bcss_patient_splits.json and .../src/config/bcss_patient_splits.json differ
Only in src/datasets: .gitkeep
Only in src/losses_metrics: .gitkeep
Only in src/pipeline: .gitkeep
Only in src/utils: .gitkeep
```

Toi da parse JSON ca 2 ban va doi chieu tung khoa: **toan bo 15 khoa va 3 danh sach split (106/22/23) giong het ve noi dung va thu tu** (`same_order=True` cho `splits/train`, `/val`, `/test`, `test_active_patients`, `train`, `val`, `test`). Khac biet chi la ** whitespace/format byte** (json.dump khong cung `indent`). => **KHONG co drift semantic giua ma nguon goc va goi Drive.** Do la mot ket luan Sach quan trong, ghi nhan `scripts/build_drive_package.py` lam tron vai tro cua no ve phan nay.

---

## 6. T6 - Device & Tai Nguyen

### 6.1 Cac diem DAT (sach)

**`weights_only` (rui ro torch>=2.6):** toan bo `src/` deu **quyet khai bao ro** va **co bao ve try/except**:

```
src/colab_pkg/model_loader.py:574       torch.load(checkpoint_path, map_location=dev, weights_only=True)
src/colab_pkg/model_loader.py:576       torch.load(checkpoint_path, map_location=dev, weights_only=False)  # nosec B614
src/models/wsss_inference_methods.py:363/:365      cung motif try weights_only=True -> except -> False
src/colab_pkg/downloader.py:230/:232               cung motif
src/training/generate_pseudo_masks.py:321/:323     cung motif
src/training/train_proposed_sam.py:402/:404        cung motif
src/training/train_stage1.py:390/:392              cung motif
src/training/train_stage2.py:298/:300, :309/:311   cung motif
```

**`map_location`:** 100% cac `torch.load` trong `src/` co `map_location` (`"cpu"` hoac `dev`). **KHONG tim thay** bat ky `torch.load` thieu `map_location` nao. Dat.

**Tensor prompt cua SAM duoc chuyen device DUNG:**

```
src/models/sam_wsss_adapter.py:252   pts_tensor = torch.tensor(pts_list[:total_pts], dtype=torch.float32, device=device)
src/models/sam_wsss_adapter.py:253   lbl_tensor = torch.tensor(lbl_list[:total_pts], dtype=torch.int64,  device=device)
src/models/sam_wsss_adapter.py:261   "active_mask": torch.tensor(batch_active, dtype=torch.bool, device=device)
src/models/sam_wsss_adapter.py:259-260 torch.stack(batch_coords, dim=0)   # stack cua cac tensor DA co device -> preserving device
```

**Seed (INVARIANT-09):** co dinh dung o moi entry point: `src/colab_pkg/colab_runner.py:101-103,109`, `src/pipeline/comparative_runner.py:41-45`, `src/training/train_stage1.py:372-376`, `src/training/train_stage2.py:250-254`. **Va cudnn determinism cung co:** `src/colab_pkg/colab_runner.py:110-111` (`torch.backends.cudnn.deterministic = True`, `benchmark = False`) cung nhu trong notebook `01`/`02`. Do la diem cong that.

**Thieu sot duy nhat ve seed:** `src/training/train_proposed_sam.py` **khong goi ham dat seed nao** (chi co `torch.manual_seed(42+idx)` o `:207` trong `SyntheticHistologyDataset.__getitem__`, khong phai seed toan cuc). Chay 5 seed cho Model De xuat la khong the thuc hien qua entry point nay. Ghi la **Low** vi co the dat seed ngoai lai tu notebook, nhung minh bach thi thieu.

**AMP:** `torch.amp.autocast` va `torch.amp.GradScaler` duoc dung dung API moi (khong dung `torch.cuda.amp` deprecated): `src/colab_pkg/model_loader.py:925`, `src/models/wsss_inference_methods.py:846`, `src/training/train_stage1.py:154,201`, `src/training/train_stage2.py:112-114`, `src/training/train_proposed_sam.py:473`. Cycle `scale->backward->step->update` hoan chinh o ca 3 trainer. `src/colab_pkg/model_loader.py:917-925` co `nullcontext()` khi khong phai CUDA -> dung cach xu ly CPU. Dat.

### 6.2 [T6-no-grad-clip] Khong co gradient clipping o bat ky dau

| Severity | **Medium** | Do tin cay | **Cao - VERIFIED (lenh am tinh that)** |
| :--- | :--- | :--- | :--- |

```
$ grep -rn "clip_grad" --include=*.py src/
(khong co output)
```

`src/training/train_proposed_sam.py:291-294` huan luyen mot **Mask Decoder hop nhat voi SAM image encoder (59.3M params, theo `src/colab_pkg/colab_runner.py:622` assert `sum(p.numel()) == 59339940`)** + gradient accumulation (`accum_steps=2`) + BF16 autocast, **khong `clip_grad_norm_`**. voi focal + dice + iou + entropy regularization cong lai va prompt points co the rong, gradient spike/NaN la rui ro thuc. Tren T4 voi `GradScaler`, mot NaN-step bi scaler bo qua se im lang (`scaler.step` tu dong skip khi inf/NaN), nen that bai hien theo cach "loss khong giam" chu khong crash -> rat kho chan doan.

### 6.3 [T6-pin-memory-un guarded] `pin_memory=True` khong kiem tra CUDA

| Severity | **Low** | Do tin cay | **Cao - VERIFIED** |
| :--- | :--- | :--- | :--- |

`src/training/train_stage1.py:416`, `src/training/train_stage2.py:329`, `src/training/train_stage2.py:341`

```python
        pin_memory=False if dry_run else True,
```

So sanh voi cach lam DUNG o cho khac trong cung repo:

```python
# src/training/dataset_loader.py:521, :538, :571, :601
        pin_memory=pin_memory and torch.cuda.is_available(),
```

`train_stage1.py`/`train_stage2.py` dung mac dinh `True` khi khong dry-run, **khong quan tâm device la gi**. Chay local CPU (Windows, nhu moi truong audit nay) sinh `UserWarning: Non-pin memory... / pin_memory option is not supported` va chiem page-locked RAM thua. Mâu thuẫn nội bộ giữa hai loader trong cùng repo cho thấy chuẩn device-handling chưa nhất quán.

### 6.4 VRAM sizing cho T4 16 GB

Batch size mac dinh:

```
src/training/train_stage1.py:470    --batch-size  default=32
src/training/train_stage2.py:396    --batch-size  default=16
src/training/generate_pseudo_masks.py:399  --batch-size  default=8
src/training/train_proposed_sam.py:427     --batch-size  default=8   ("Batch size per worker.")
```

Stage-2 la DeepLabV3+ ResNet-101 output_stride=16 (theo `src/models/deeplabv3plus.py`, assert 59.3M tham so o `colab_runner.py:622`). Voi `224x224` + ASPP 4 nhan [1,6,12,18] + optimizer AdamW (2 state fp32) va gradient accumulation, batch 16 fp32 uoc 6-9 GB, van an toan tren T4 16 GB. **Ket luan T6 ve batch size: hop ly, khong co vi pham.**

Nhung **khong co bat ky tai lieu uoc VRAM nao trong repo cho cac lenh chay Colab** (khong co `torch.cuda.max_memory_allocated`, khong co `memory_allocated` logging - `grep` cho ra 0 ket qua trong `src/` ve logging VRAM dinh). Trong khi INVARIANT-27 yeu cau "muc tieu ton VRAM dinh theo batch size" va bao cao lai `VRAM Dinh = <50 MB` cua `SE374_FINAL_TECHNICAL_REPORT_WSSS.md:196-197` **khong duoc do bang code that** (xem 2.2). Do la mot khe hoat dong trong cua 6 chieu T6.

---

## 7. T7 - Silent Failure

```
$ grep -rn "except:" --include=*.py src/ | wc -l
0
$ grep -rn "except Exception" --include=*.py src/ | wc -l
49
```

**Khong co bare `except:` nao trong `src/`** - do la mot diem Sach. `warnings.filterwarnings` trong `src/`: **0**.

Danh sach 20 handler "not lai loi" (`except ...: pass` / tra ve gia tri ao), dung python AST-like scan:

```
src\clinical\exhaustive_evaluator.py:545   except ImportError:  -> return {"status": "patient_level_helper_unavailable"}
src\clinical\survival_analysis.py:290      except Exception:    -> pass
src\clinical\survival_analysis.py:681      except Exception:    -> pass
src\clinical\survival_analysis.py:806      except Exception:    -> pass
src\colab_pkg\colab_runner.py:33           except Exception:    -> pass
src\colab_pkg\colab_runner.py:575          except ValueError:   -> pass
src\colab_pkg\downloader.py:129            except ValueError:   -> pass
src\colab_pkg\evaluator.py:49              except ImportError:  -> pass
src\colab_pkg\evaluator.py:181             except Exception:    -> pass
src\models\sam_wsss_adapter.py:315         except Exception:    -> pass
src\pipeline\wsi_lowmag_highmag_pipeline.py:151,166,410,567  except Exception: -> pass
src\utils\deliverable_auditor.py:196,202,207,389,625         except Exception: -> pass
src\visualization\api_server.py:300        except Exception:    -> pass
TOTAL: 20
```

### 7.1 [T7-metric-subset-skew] FWIoU/BF1 duoc do tren mot tap con khong bao gio duoc cong bo

| Severity | **High** | Do tin cay | **Cao - VERIFIED** |
| :--- | :--- | :--- | :--- |

`src/colab_pkg/evaluator.py:173-183`

```python
        try:
            p_fwiou = compute_fwiou(pred, target, num_classes=self.num_classes)
            self.patch_fwiou.append(p_fwiou * 100.0)
            p_bf1 = compute_boundary_f1(pred, target, num_classes=self.num_classes)
            if isinstance(p_bf1, dict):
                self.patch_bf1.append(float(p_bf1.get("BF1", 0.0)))
            else:
                self.patch_bf1.append(float(p_bf1) * 100.0)
        except Exception:
            pass
```

Neu `compute_fwiou` hoac `compute_boundary_f1` nem loi o mot patch (rong `target`, vuot kich thuoc, hoac `scipy` khong co san), **chinh xac patch do bi loai khoi danh sach FWIoU/BF1**, trong van no duoc giu nguyen trong `patch_miou`/`patch_mdice`/`patch_patient_ids`. Ket qua:

- Cột "FwIoU 4-Class" trong bảng benchmark nop Giang vien **duoc tinh tren n khac** so voi cot mIoU, ma khong co thong ke n o dau.
- Khong co `len(self.patch_fwiou) == len(self.patch_miou)` assertion nao trong `src/`.
- Doi chieu bang trong `EXHAUSTIVE_3WAY_BENCHMARK_REPORT.md`: `FwIoU == Pixel_Accuracy_4cls` **chinh xac tung chu so** o ca 3 model (19.95/19.95 va 20.06/20.06). Voi mot FwIoU that (frequency-weighted, khac PA vi so lieu cheu) thi bang tuyet doi 2/2 la mot co hoi rat thap.Day la mot dau hieu bo sung rang cot FwIoU khong mang thong tin doc lap.

He qua lam sang/hoc thuat: trong mot so sanh ablation, "moi metric deu dat tren 4,986 patch" la mot gia dinh mac dinh cua nguoi doc. Handler nay pha vo gia dinh do mot cach im lang.

### 7.2 [T7-auditor-self-disable] Gate INVARIANT-25 tu vo hieu hoa khi anh doc loi

| Severity | **High** | Do tin cay | **Cao - VERIFIED** |
| :--- | :--- | :--- | :--- |

`src/utils/deliverable_auditor.py:389-390` (trong check mau/gradient anh) va `:196,202,207,625`

```python
                        if mean_color_std > 30.0:
                            self.issues.append(DeliverableIssue(
                                category="IMAGE", severity="CRITICAL", ...
                                message="Phat hien hinh ve mau me thieu chuan muc khoa hoc ..."))
                except Exception:
                    pass
```

Nghia la: **neu PIL/Pillow khong mo duoc anh** (duong dan sai, anh DICOM/NIfTI-render khong phai PNG that, file .emf vector, anh > `Image.MAX_IMAGE_PIXELS` bi chan), kiem tra "cam mau se" **bi bo qua va khong sinh ra bat ky issue nao**. Auditor bao "PASSED" cho mot tai lieu chua mot anh ma no chua bao gio nhin thay.

Bang chung hanh vi truc tiep (lenh chay that):

```
$ KMP_DUPLICATE_LIB_OK=TRUE python -X utf8 -m src.utils.deliverable_auditor \
      product/reports/clinical/EXHAUSTIVE_3WAY_BENCHMARK_REPORT.md
[DELIVERABLE QUALITY AUDIT: PASSED] - Toan bo tieu chuan hoc thuat, chat luong hinh anh, chinh ta va cau truc deu DAT 100%!
exit=0
```

**PASSED 100%** cho mot bao cao chua: Model A = `0.00%` moi chi tieu, `total_patches_evaluated = 2`, `t_statistic = inf`, `cohen_d = 0.0`, CI rong bang 0, va survival cohort = 1 benh nhan. Gate INVARIANT-25 (cap 4 tang kiem soat chat luong) hoan toan khong duoc thiet ke de phat hieu lo-gic hoc thuat. Do la **khet niem trong thiet ke cua gate**, khong chi la mot loi code.

### 7.3 Cac handler khong dang ke (phai ghi nhan cho du)

- `src/models/sam_wsss_adapter.py:315` - `except Exception: pass` bao quanh viec chon prompt points bang connected-components (`scipy.ndimage.label`), roi roi nhanh fallback "chọn 1 điểm bất kỳ". Lam chat luong SAM prompt giam mot cach vo sinh. **Medium.**
- `src/pipeline/wsi_lowmag_highmag_pipeline.py:151,166` - chuoi fallback OpenSlide -> tifffile -> PIL. Mot file SVS hong se bi doc bang PIL o **sai scale** (PIL doc level 0 hoac level dau tien), sinh overview o do phong dai sai -> toan bo phan tich low-mag/high-mag tren WSI do se sai ti le ma khong bao. **High ve lam sang**, nhung rui ro nay phu thuoc file dau vao.
- `src/pipeline/wsi_lowmag_highmag_pipeline.py:567` - `slide_handle.close()` trong `finally` - **chap nhan duoc**, day la mot idiom standard.
- `src/utils/deliverable_auditor.py:196-207` - Word COM automation + `pythoncom.CoUninitialize()` - **chap nhan duoc** (best-effort cleanup).
- `src/colab_pkg/colab_runner.py:33` - UTF-8 stdio reconfigure - **chap nhan duoc**.
- `src/visualization/api_server.py:300` - can doc them ngan canh cu the, kha nang anh huong thap (API endpoint).
- `src/colab_pkg/evaluator.py:49` - `except ImportError: pass` cho mot tuychon tro cuu thong ke - **can danh dau**: neu no vo hieu hoa mot ham stat test thi day la 7.1 phien ban khac.

### 7.4 [T7-exhaustive-clause-cohort-escape] Cohort lloc bi bo qua khi co so nho

`src/clinical/exhaustive_evaluator.py:404-408`

```python
            if active_set and pid not in active_set and len(active_set) > 0:
                # Still include if sample dataset or total test count is small
                if len(image_files) > 100:
                    continue
```

Neu co so < 100 patch, **moi benh nhan ngoai cohort van duoc nap vao danh gia**. Tren du lieu that (4,986 patch) nhanh loc hoat dong dung - toi da verify `fs-test patients NOT in manifest test = 0` o 3.1. Nhung day la mot chan ngat co the tu tat chinh no. **Medium.**

---

## 8. T8 - INVARIANT-12 (ASCII hyphen) + CI

### 8.1 [T8-typography-violations] 230 dong vi pham, tuyen bo "0" la SAI

```
$ grep -rnP "[\x{2013}\x{2014}\x{2015}\x{2212}]" --include=*.py --include=*.md --include=*.ipynb \
      --exclude-dir=repos --exclude-dir=.git . 2>/dev/null | wc -l
230
```

Phan bo theo thu muc (lenh dem moi thu muc):

```
src : 0
tests : 0
scripts : 0
notebooks : 0
drive_upload_ready : 0
product : 0
docs : 0
experiments : 0
scratch : 0
.github : 0
root *.md *.ipynb (AGENTS.md, README.md, 11 notebook goc) : 0
```

Và **toàn bộ 230 dòng nằm trong**:

```
./.agents/rules/13_deliverable_quality_and_scientific_visuals_gate.md
./.agents/teamwork/audit_reviewer_parity_invariant12/DISPATCH.md
./.agents/teamwork/checker_team_a/audit_report.md
./.agents/teamwork/team3_reauditor_papers_docs/BRIEFING.md
./.agents/teamwork/team3_reauditor_papers_docs/progress.md
./.agents/teamwork/victory_auditor_5/BRIEFING.md
./.agents/teamwork/victory_auditor_5/handoff.md
./.agents/teamwork/victory_auditor_5/progress.md
./.agents/teamwork/victory_worker_r3/BRIEFING.md
./.agents/teamwork/victory_worker_r4/DISPATCH.md
./graphify-out/GRAPH_REPORT.md  (+ 10 GRAPH_REPORT.md theo ngay trong graphify-out/2026-10-*/ )
./graphify-out/converted/*.md   (16 tep)
./graphify-out-all/converted/*.md  (3 tep)
```

**Phan tinh khach quan:** toan bo noi dung chuc nang cua do an (`src/`, `tests/`, `scripts/`, `notebooks/`, `drive_upload_ready/`, `product/`, `docs/`, `AGENTS.md`, `README.md`) **SACH 100%**. `.agents/` la tin hieu tac tu, `graphify-out*` la san phum may sinh - ve ky la "khong phai san pham nop". **NHUNG**: tuyen bo cu "100% compliance, 0 forbidden unicode dash across all workspace files" la **sai ve mat chu nghia** (no noi "all workspace files"), va `graphify-out/` duoc sinh tu chinh repo nen van la artifact cua repo.

`[T8-typography-overclaim]` | **Low** ve noi dung hoc thuat, **Medium** ve liem chinh bao cao | Do tin cay: Cao (VERIFIED) | 230 dong / 40 tep nhu tren.

### 8.2 [T8-governance-test-blind] `test_governance_invariants.py` co thiet ke khong the phat hien vi pham cua 8.1

| Severity | **High** | Do tin cay | **Cao - VERIFIED** |
| :--- | :--- | :--- | :--- |

`tests/test_governance_invariants.py:1-43`

```python
IGNORED_DASH_PREFIXES = (
    "repos\\", "repos/",
    "graphify-out\\", "graphify-out/",
    "graphify-out-all\\", "graphify-out-all/",
    "temp_graphify_all\\", "temp_graphify_all/",
    ".pytest_cache\\", ".pytest_cache/",
    ".ruff_cache\\", ".ruff_cache/",
    ".git\\", ".git/",
)

def _scan_files_for_bad_dashes(patterns):
    target_files = []
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):        # <--- van de o day
            if not f.startswith(IGNORED_DASH_PREFIXES):
                target_files.append(f)
```

**Loi can ban:** `glob.glob("**/*.md", recursive=True)` **bo qua moi thu muc an (bat dau bang `.`)** theo duong loi cua Python `glob` (no khong match `.agents`). Vi vay `.agents/` khong bi loai bo vi `IGNORED_DASH_PREFIXES` - no **khong bao gio xuat hien trong danh sach duyet**. Test pass khong phai vi sach, ma vi no **mu**. Nhom ignore list la mot **an toan thu cong** che phia tren mot lloc tu dong.

Chung minh: `IGNORED_DASH_PREFIXES` khong chua `.agents`, vay ma `test_no_ai_em_dash_in_markdown` van pass trong run 346-test cua toi -> doan bac bang `glob` dot-suppression.

**Test thuc su enforce gi (cau tra loi cho yeu cau "doc xem no enforce gi"):**

| Test | Enforce that |
| :--- | :--- |
| `test_no_ai_em_dash_in_markdown` | Chi `*.md` khong an. Co dung voi `product/`, `docs/`, `AGENTS.md`. |
| `test_no_ai_em_dash_in_python_and_notebooks` | Tu tu, cho `*.py`/`*.ipynb`. |
| `test_invariants_24_and_25_defined_in_agents_md` | Chi kiem **3 chuoi van ban** co xuat hien trong `AGENTS.md`. **Doan na: dung la test van ban, khong phai test hanh vi**, va no khong kiem 29 INVARIANT con lai. |
| `test_no_raw_phone_photos_in_product_directory` | Kiem **ten file** bat dau bang `media_`/`img_`/`pxl_`/`screenshot`/`dcm_` trong `product/`. **Loi thiet ke:** anh quet dien thoai dat ten `slide_1.png` hay `IMG-001.jpg` (gach noi) se duoc bo qua; va no **khong kiem noi dung anh** (khong do do lech mau, do phan giai) - khac hoan toan voi muc tieu INVARIANT-24. `os.walk("product")` cung gia dinh CWD = goc repo. |

### 8.3 [T8-ci-commands-cannot-run] `.github/workflows/ci.yml` co ton tai nhung khong the chay xuong

| Severity | **High** | Do tin cay | **Cao - VERIFIED (doc) + Cao (phuong phap: so sanh requirements)** |
| :--- | :--- | :--- | :--- |

`.github/workflows/ci.yml` **ton tai** (INVARIANT-03 da dat ve mat hien dien). Noi dung:

```yaml
      - name: Install Lint & Test Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install flake8 pytest Pillow

      - name: 1. Code Style & Linting Check
        run: |
          flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics || true
          flake8 src/ --count --exit-zero --max-complexity=15 --max-line-length=120 --statistics || true

      - name: 2. Academic & Medical Integrity Audit (Zero Mockup)
        run: |
          if grep -rnw 'src/' -e 'TODO: mock' -e 'fakeData' -e 'dummyData' -e 'fake_metric' -e 'simulateTx'; then

      - name: 3. Deliverables & Scientific Visuals Quality Gate
        run: |
          python -m src.utils.deliverable_auditor product/proposal/SE374_MINI_PROPOSAL_WSSS_WSI.md

      - name: 4. Run Unit Tests & Integration Tests
        run: |
          if [ -d "tests" ]; then
            pytest tests/ -v
```

Doi chieu tung lenh voi repo that:

| Buoc | Chay duoc tren repo nay? | Bang chung |
| :--- | :--- | :--- |
| 1 (flake8) | **CO lenh, NHUNG vo nghia** | Ca 2 dong deu `|| true` hoac `--exit-zero`. CI **khong the** fail vi lint. 33 loi `scripts/` va moi loi E9/F82 tuong lai deu in ra roi bo qua. |
| 2 (grep mock) | **CO lenh, NHUNG vo nghia** | Tim 5 chuoi den (`TODO: mock`, `fakeData`, `dummyData`, `fake_metric`, `simulateTx`). Toi da kiem: khong chuoi nao xuat hien trong `src/`. Toan bo 10 loi Zero-Mockup that cua T2 (np.zeros GT, synthetic fallback, cohort hard-code...) deu **khong chua** 5 chuoi nay. Gate la mot **blacklist tu khoa**, khong phai quet logic. |
| 3 (auditor) | **CO - da chay that exit=0** | Toi chay lenh y het: `python -m src.utils.deliverable_auditor product/proposal/SE374_MINI_PROPOSAL_WSSS_WSI.md` -> `PASSED`, exit 0. File ton tai (`product/proposal/SE374_MINI_PROPOSAL_WSSS_WSI.md`). `src/utils/deliverable_auditor.py:636` co `if __name__ == "__main__":`. **DAT.** |
| 4 (pytest) | **KHONG - se hong khi collect** | Ci chi cai `flake8 pytest Pillow`. Cac test import: `torch numpy cv2 fastapi matplotlib hypothesis scipy` + `PIL` (Pillow OK). `requirements.txt:5-44` yeu cau torch, torchvision, monai, timm, opencv-python-headless, scikit-image, numpy, scipy, pandas, scikit-learn, matplotlib, seaborn, lifelines, streamlit, python-docx, pymupdf, openpyxl. **CI cai 3/22 thu vien.** `pytest tests/` se `ERROR tests/test_pda_backbone.py - ModuleNotFoundError: No module named 'torch'` ngay o buoc collect. |

**Khong co `KMP_DUPLICATE_LIB_OK=TRUE`** trong CI. tren `ubuntu-latest` dieu nay khong can thiet (OMP duplicate lib la van de Windows/Anaconda), nen **khong phai la loi CI** - ghi nhan de tranh double-flag. Nhung cung co nghia la CI **khong tai hien duoc** moi truong Windows cua nhom, o do 2 loi T1 xuat phat.

**Khong co buoc CI nao cho 2 trong so 3 chot chan ten tui cua INVARIANT-03:** patient-leakage check (khong co `pytest tests/test_patient_split.py` rieng) va zero-mockup scan thi chi la blacklist tu khoa (buoc 2).

He qua: INVARIANT-03 yeu cau CI "tu dong lint, test va quat anti-mockup". Thuc te CI se **hong o buoc 4 voi ModuleNotFoundError**, va neu buoc do duoc xoa/sua thi buoc 1 va 2 **khong bao gio fail**. Day la CI trang tri.

### 8.4 SACH (T8)

`grep -nP "[\x{2013}\x{2014}\x{2015}\x{2212}]"` tra ve **0** cho `src/`, `tests/`, `scripts/`, `notebooks/`, `drive_upload_ready/`, `product/`, `docs/`, `experiments/`, `scratch/`, `.github/` va tat ca file goc. Ky thuat ASCII-hyphen trong noi dung that cua do an la **dat 100%**.

---

## 9. T9 - Dead Code & Duplication

### 9.1 `src/config/` - 0 file `.py`

```
$ ls -la src/config/
-rw-r--r-- 1 ...   44 Sep 28 18:10 .gitkeep
-rw-r--r-- 1 ... 7643 Oct 25 17:31 bcss_patient_splits.json

$ find src/config -name "*.py" | wc -l
0
```

**Co ai import no khong?** Khong ai `import src.config` (khong the - khong co `__init__.py`), nhung **co 5 noi doc file JSON cua no**:

```
src/training/dataset_loader.py:123        config_path = project_root / "src" / "config" / "bcss_patient_splits.json"
src/datasets/patient_split.py:129         config_path = os.path.join(project_root, "config", "bcss_patient_splits.json")   # (-> src/config/, dung)
src/clinical/survival_analysis.py:279     config_path = project_root / "src" / "config" / "bcss_patient_splits.json"
src/visualization/api_server.py:25,:416   from src.datasets.patient_split import load_patient_splits
notebooks/04_Exhaustive_Benchmark_Survival*.ipynb:169   SPLIT_FILE = Path(CODE_ROOT)/'src'/'config'/'bcss_patient_splits.json'
```

Nen `src/config/` **KHONG phai dead** - no la thu muc du lieu. Nhung:

`[T9-config-module-absent]` | **Medium** | `src/config/` (khong ton tai file nao)

Muc 3 cua `AGENTS.md` mo ta `src/config/` la **"Cấu hình siêu tham số, random seeds, paths, device GPU"**. Thuc te **khong co module thong so nao**. Moi sieu tham so duoc hard-code trong argparse defaults roi rac: `train_stage1.py:470` (batch 32), `train_stage2.py:396` (batch 16), `train_proposed_sam.py:427` (batch 8), `generate_pseudo_masks.py:399` (batch 8). Khong co file `seeds.py` (`DEFAULT_SEEDS` nam trong `src/pipeline/comparative_runner.py`). Khong co `paths.py`, khong co `device.py`. Vi the INVARIANT-05 ("Decoupled Modular Separation") bi pha o tang cau hinh: doi 1 batch size phai sua 4 file khac nhau.

### 9.2 Module khong duoc import o dau (dead code that)

```
$ (python scan: moi module src/*.py, dem so lan duoc neu trong src+tests+scripts+experiments+moi notebook)
  ZERO-REFERENCE: src\visualization\draw_all_proposal_figures.py
  ZERO-REFERENCE: src\visualization\draw_pipeline_diagram.py
```

Chi 2/42 module la zero-reference. Ca 2 deu co `if __name__ == "__main__"` (CLI ve hinh minh hoa cho proposal). Do la **entry point co chu dich, khong phai dead code**. Ghi nhan la Low/informational, va chung **khong duoc CI goi** -> nen cac hinh trong `product/proposal/assets/` khong tai lap tu dong duoc.

**Diem lien quan:** `git status` cho thay `product/proposal/assets/preview/pages/page_1..10.png` **bi sua (`M`)** - day la thuoc tinh cua viec toi chay `pytest tests/` va `scripts/build_drive_package.py` (buoc 5.3 o T5 va muc 9.3), tuc la **bo test ghi de asset cua san pham nop**. Xem 9.4 ve cong bo minh bach.

### 9.3 doi chieu md5 3 ban sao notebook + 2 ban sao `src/`

- `src/` vs `drive_upload_ready/SE374_WSSS_Drive_Code/src/`: **identical semantic** (chi khac `.gitkeep` va whitespace JSON) - xem 5.3. SACH.
- `notebooks/` vs goc: **3 file drift semantic** (`01...`, `01..._L4`, BonusDemo chi khac outputs) - xem 5.3.
- `notebooks/` vs `drive_upload_ready/SE374_WSSS_Drive_Code/notebooks/`: md5 cua `SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb` va 4 file `*_ThinDriveRunner_v2.ipynb` xuat hien o ca 3 cho - toi da doi tuong va kha nang drift la nhu nhau.
- **6 notebook bi "moi" o mot trong hai cho** (ma nguon co, ban goc khong): `02_..._Stage2.ipynb`, `02_..._Stage2_L4.ipynb`, `03_..._MedSAM.ipynb`, `03_..._MedSAM_L4.ipynb`, `04_..._Survival.ipynb`, `04_..._Survival_L4.ipynb`, `SE374_Colab_Scientific_Research_Runner_FIXED.ipynb`, va 4 file `SE374_Colab_WSSS_*.ipynb`.
- **1 orphan o goc:** `01_Train_Stage1_Classifier_v2.ipynb` - khong ban sao nao trong `notebooks/`.

`[T9-triplicate-notebooks]` | **Medium** | 34 `.ipynb` ton tai o 2-3 noi. Mot sua phai sua 3 cho; khong co co che dong bo tu dong trong CI.

### 9.4 [T9-tests-write-into-repo] Test ghi de trang thai repo that

| Severity | **High** | Do tin cay | **Cao - VERIFIED** |
| :--- | :--- | :--- | :--- |

`tests/test_thin_drive_runner.py:96` va `:175`

```python
def test_2_drive_bundle_packager_creates_complete_folder_and_zip(tmp_path):
    """..."""
    summary = packager.build_drive_package()          # <--- tmp_path KHONG BAO GIO DUOC SU DUNG
    assert BUNDLE_DIR.is_dir(), f"Bundle directory not created: {BUNDLE_DIR}"
```

Ham nhận fixture `tmp_path` **nhưng không dùng nó** - `build_drive_package()` viết thẳng vào `drive_upload_ready/` của repo (hằng `BUNDLE_DIR`). `test_3` (`:175`) gọi cùng hàm, không có `tmp_path` luôn.

Đây là vi phạm nghiêm trọng nhất về chất lượng test trong `tests/`: một test ghi đè thư mục *deliverable* thì (a) không idempotent, (b) để lại repo dirty sau mỗi lần chạy, (c) xung đột file-handle gây ra đúng 2 lỗi T1, và (d) không thể chạy song song (`pytest-xdist`) an toàn.

---

## 10. T10 - Chat luong Test + Bug-hunt doi chieu ngoai

### 10.1 Danh gia `tests/` theo testing-best-practices

|ieu chí | Kết luận | Bằng chứng |
| :--- | :--- | :--- |
| **Không phụ thuộc thứ tự chạy** | **ĐẠT** | `grep -rn "pytest.mark.order|randomly|global " tests/*.py` → 0 kết quả. |
| **Không `time.sleep` nondeterminism** | **ĐẠT (sạch hoàn toàn)** | `grep -rn "time.sleep|asyncio.sleep" tests/` → 0 kết quả. |
| **Mock tối thiểu (không mock chính hệ thống)** | **RẤT TỐT** | `grep -rn "MagicMock|patch(|mocker|Mock(" tests/*.py` → **3 kết quả trên 22 file**, chỉ trong `tests/test_scientific_upgrades.py`. 19/22 test file **không mock gì cả**, dùng tensor/PIL/dataset thật. Đây là điểm mạnh nổi bật. |
| **Test HÀNH VI vs chi tiết cài đặt** | **MIXED - có 3 chỗ test cài đặt** | Ví dụ test-cài-đặt: `tests/test_deeplabv3plus.py` assert `"penalty = -1e4"` và `dilation_rates == [1,6,12,18]`; `tests/test_sam_wsss_adapter.py` assert kiến trúc bottleneck `256 -> 64 -> 256`; `tests/test_pda_backbone.py` assert `p=0.25`. Những con số này là *siêu tham số*, không phải hành vi - đổi hyperparameter sẽ đỏ test mà không phát hiện lỗi. `tests/test_audit_team_3.py` (Invariant-12 check) là test hành vi đúng chuẩn. |
| **Đường lỗi + biên (rỗng, None, 0, -1)** | **ĐẠT MỘT PHẦN** | Có: `tests/test_patient_split.py:148-151` (`FileNotFoundError`), `:52-60` (overlap → `ValueError`), `src/colab_pkg/colab_runner.py:638-642` (shape sai → `ValueError`). **Thiếu nghiêm trọng:** **KHÔNG test nào** bao phủ chính các fallback path mà T2 phát hiện: không có test assert `BCSSStage2MultiMaskDataset` phải raise khi `train_PM` trống, không có test cho `np.full(...,4)` GT fabrication, không có test cho `SyntheticHistologyDataset` leak vào báo cáo, không có test cho `load_bcss_patient_splits` fallback 7-bệnh-nhân. Đó là lý do 346 test xanh mà 5 defect Critical vẫn sống. |
| **Không có weak assertion** | **ĐẠT** | `grep -rn "assert True" tests/*.py` → 0. Có 5 chỗ `assert len(x) > 0` (`test_clinical_service.py:219,254`, `test_training_service.py:162,319`, `test_wsi_pipeline_extension.py:271`) - chấp nhận được vì chúng assert tính có thật của dữ liệu, nhưng `test_wsi_pipeline_extension.py:271` (`assert len(grad_enabled_during_inference) > 0`) là tự tham chiếu và nên là `assert grad_enabled_during_inference == [False]`. |
| **Test isolation vs repo state** | **KHÔNG ĐẠT - CRITICAL** | Xem `[T9-tests-write-into-repo]` mục 9.4. |
| **Độ phủ pipeline thật (Anti-Mock-Testing Gate)** | **ĐẠT có điều kiện** | `tests/test_patient_split.py:74-91` load thật từ `data/sample_dataset` và assert shape `(3,224,224)`; `data/BCSS-WSSS` có 31,826 ảnh thật và tôi verify nạp được batch thật (mục 3.1). Nên **không** có "mock illusion" ở tầng data ingestion. |

### 10.2 [T10-max-samples-order-bias] Cắt mẫu theo thứ tự tên file

`src/training/dataset_loader.py:167` rồi `:210-211`, `:298-299`, `:424-425`

```python
    files.sort(key=lambda p: p.name)
    ...
    if max_samples is not None and max_samples > 0:
        filtered_files = filtered_files[:max_samples]
```

Vì `sort(key=p.name)` gom nhóm theo bệnh nhân (TCGA-XX-YYYY đứng trước coordinate), `[:max_samples]` lấy **toàn bộ patch của vài bệnh nhân đầu bảng chứ không phải mẫu ngẫu nhiên**. Sanity-150 trên 23,422 patch vì thế có thể chỉ chứa 1-3 bệnh nhân.

Hậu quả: (a) `mIoU` của sanity run không đại diện; (b) nếu áp cho cả train và test cùng một cohort chữ cái đầu, hai tập có thể **chồng bệnh nhân** dù manifest ban đầu disjoint. Đây là một **leakage có thể phát sinh từ cách lấy mẫu**, mà không test nào kiểm. Mức **Medium** (INFERENCE - logic đọc được rõ, nhưng tôi chưa chạy `max_samples=150` trên toàn 23,422 patch vì tốn thời gian; dữ liệu trên đĩa đã xác nhận disjoint ở mức full-split nên rủi ro là tương lai).

### 10.3 [T10-bughunt] 3 lỗi ngoài đối chiếu - mã repo này CÓ NGUY CƠ mắc

#### (a) `model.load_state_dict(..., strict=False)` âm thầm để lại trọng số khởi tạo ngẫu nhiên

**Chữ ký lỗi trong repo:** `src/clinical/exhaustive_evaluator.py:329` `model.load_state_dict(cleaned, strict=False)` và `:368` `model.load_state_dict(sd, strict=False)`.

`strict=False` **không** bỏ qua size mismatch (vẫn raise `RuntimeError`), nhưng nó **bỏ qua silently `missing_keys` và `unexpected_keys`**. Nếu checkpoint lưu prefix khác (`module.xxx`, `model.xxx`, hay kiến trúc DeepLabV3+ của torchvision vs bản tự viết), các tham số không khớp **giữ giá trị khởi tạo ngẫu nhiên**, hàm trả về object `InconsistentKeys`/namedtuple mà không ai đọc. Đây là cơ chế giải thích chính xác nhất `Pixel_Accuracy_4cls = 0.00%` của Model A.

**Bản vá chuẩn** (theo tinh thần accepted answer của pytorch#24139 / #40859, và hướng xử lý được khuyến nghị trong pytorch#155368):

```python
result = model.load_state_dict(cleaned, strict=False)
if result.missing_keys or result.unexpected_keys:
    raise RuntimeError(
        f"Checkpoint '{ckpt_p}' is architecturally incompatible with {tag}. "
        f"missing={result.missing_keys[:12]} unexpected={result.unexpected_keys[:12]}"
    )
```

Và **không được** fall-through sang model khởi tạo ngẫu nhiên (xem `[T7-evaluator-random-weights-fallback]`).

**Nguồn tham chiếu:**
- https://github.com/pytorch/pytorch/issues/24139 (Allow incompatible shapes in load_state_dict(strict=False))
- https://github.com/pytorch/pytorch/issues/40859 (Option to allow loading state dict with mismatching shapes)
- https://github.com/pytorch/pytorch/issues/155368 (Change error to warning when tensor size mismatch - thảo luận về `warn_on_mismatch`)
- https://discuss.pytorch.org/t/size-mismatch-weights-and-biases/161085

#### (b) `scipy.stats.wilcoxon` / `ttest_rel` trên mẫu n=1 hoặc các chênh lệch bằng 0 cho `t = inf` và `p` vô nghĩa

**Chữ ký lỗi trong repo:** `src/clinical/exhaustive_evaluator.py:571` `stats.ttest_rel(scores_m, scores_a)` và `:573` `stats.wilcoxon(scores_m, scores_a)`, với kết quả JSON thật: `t_statistic = inf`, `cohen_d = 0.0`, `wilcoxon_p_value = 0.5`, `patient_cohort_count = 1`.

Với n = 1 (một bệnh nhân), `ttest_rel` có `sem ≈ 0` → `t = inf`, `p = 0`. SciPy trả về kết quả **không raise**, chỉ `RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation` - chính xác là warning tôi thấy ở mục 1.6. Repo có guard `std_diff_ma > 1e-6 else 0.0` cho Cohen's d (`:579`) nhưng **không có guard nào trước `ttest_rel`**, và không có guard `n >= 2`.

Trong khi đó `src/colab_pkg/evaluator.py:483-489` xử lý ĐÚNG một nửa:

```python
    try:
        if np.all(np.abs(diffs) < 1e-12):
            w_stat, w_pval = 0.0, 1.0
        else:
            w_res = stats.wilcoxon(diffs, zero_method="zsplit", alternative="two-sided")
            ...
    except Exception:
        w_stat, w_pval = 0.0, 1.0
```

Nhưng `except Exception: w_stat, w_pval = 0.0, 1.0` biến **mọi** lỗi thống kê thành `p = 1.0` = "không có ý nghĩa" - một **kết luận khoa học được bịa ra từ một crash**.

**Bản vá chuẩn:**

```python
if len(scores_m) < 2 or len(scores_a) < 2:
    out.update({"t_statistic": float("nan"), "t_test_p_value": float("nan"),
                "test_status": "skipped_insufficient_patients",
                "n_patients": len(scores_m)})
    return out
```

và **không** dùng `except Exception` gán `p = 1.0`; thay bằng `except Exception as e: record status="wilcoxon_error", reason=str(e)` và đặt p = NaN (NaN không thể bị diễn giải thành "không có ý nghĩa").

**Nguồn tham chiếu:**
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html (`zero_method` semantics cho zero-differences)
- https://github.com/scipy/scipy/issues/6805 (error in scipy Wilcoxon signed-rank test for equal series)
- https://stats.stackexchange.com/questions/62110/error-in-scipy-wilcoxon-signed-rank-test

#### (c) `torch.load(weights_only=True)` đổi mặc định ở PyTorch 2.6 - checkpoint chứa object Python

**Chữ ký trong repo:** repo **ĐÃ XỬ LÝ ĐÚNG** - mọi `torch.load` trong `src/` đều try `weights_only=True` rồi fall-back `weights_only=False` (xem 6.1). Ghi nhận là **ĐIỂM MẠNH**. Tuy nhiên `weights_only=False` fall-back vẫn là rủi ro pickle và được `# nosec B614` im lặng - chấp nhận được vì checkpoint là tự sản sinh. **Không có lỗi cần sửa.** Tôi ghi mục này để xác nhận chiều kiểm định đã được chạy và cho kết quả SẠCH, thay vì bịa một finding.

**Nguồn tham chiếu (dùng để xác nhận cách repo xử lý là hợp chuẩn):**
- https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573
- https://stackoverflow.com/questions/79584485/unable-to-torch-load-due-to-pickling-safety-error
- https://github.com/axondeepseg/axondeepseg/issues/994 (torch.load fails with PyTorch >=2.6)

---

## 11. Ma Tran Nghiem Thu (Verification Matrix)

| Chieu |KET LUAN | So defect | Ghi chu |
| :--- | :--- | :-: | :--- |
| **T1** Test suites | **KHONG DAT** | 2 | `2 failed / 346 passed`, flakiness 2/6 run; ruff `src tests scripts` = 33 errors. |
| **T2** Zero-Mockup (INV-07) | **KHONG DAT** | 7 | Bao cao benchmark 2-patch + bang 5-seed thieu log + GT gia + pseudo-mask zeros + nhan yeu bia + cohort hard-code + synthetic training. |
| **T3** Anti-Leakage (INV-08) | **DAT (ket qua) / KHONG DAT (phuong phap)** | 3 | Du lieu that 0% leakage (VERIFIED, 31,826 patch), nhung test tu tham chieu va `BCSSDataset` khong dung manifest. |
| **T4** Anti-Shallow-Glob (INV-29) | **KHONG DAT (ca 2 yeu cau)** | 2 | 7 cho glob nong trong loader chinh; safety-net placeholder **khong ton tai o dau**. |
| **T5** Colab (INV-28/27) | **KHONG DAT** | 3 | 23 notebook `drive.mount`; 4 notebook thieu buoc 6/7; drift noi dung 2 notebook. |
| **T6** Device & tai nguyen | **DAT CO DIEU KIEN** | 2 | weights_only/map_location/SAM-prompt/seed/cudnn **dat**; thieu grad-clip, pin_memory khong guard, khong do VRAM that. |
| **T7** Silent failure | **KHONG DAT** | 4 | Metric-subset skew; auditor tu vo hieu hoa; surrogate cohort; loader scale sai. |
| **T8** INV-12 + CI | **NOI DUNG DAT / CI KHONG DAT** | 3 | 0 vi pham trong `src/tests/notebooks/product/docs`; 230 dong vi pham trong `.agents`+`graphify-out`; test governance mu; CI hong o buoc 4. |
| **T9** Dead code & duplication | **KHONG DAT (nhe)** | 3 | `src/config` 0 file `.py` (thieu module sieu tham so); 34 notebook triplicate; test ghi de repo. |
| **T10** Chat luong test + bug-hunt | **KHONG DAT (mot chieu)** | 2 | Mock toi thieu & khong sleep (rat tot); nhung 0 test bao phu fallback paths, va `[:max_samples]` gay lech mau. |

### tong hop theo severity

| Severity | So luong |
| :--- | :-: |
| **Critical** | **7** |
| **High** | **9** |
| **Medium** | **9** |
| **Low** | **5** |
| **TONG** | **30** |

**Danh sach Critical:**

1. `[T2-benchmark-report-2-patches]` - `product/reports/clinical/EXHAUSTIVE_3WAY_BENCHMARK_REPORT.md:1-40` + `exhaustive_3way_benchmark_results.json` (n=2, Model A 0.00%, t=inf, cohort 1 BN)
2. `[T2-five-seed-table-no-log]` - `product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md:190-197` (log `experiments/benchmarks/benchmark_results.json` khong ton tai; std=0.00; VRAM <50MB vo ly; 2 dong giong het nhau)
3. `[T2-fake-gt-mask]` - `src/clinical/exhaustive_evaluator.py:439` (`np.full(shape, 4)` khi thieu GT)
4. `[T2-zero-supervision-fallback]` - `src/training/dataset_loader.py:323-325` (VAI LAP HANH VI: `train_PM` rong -> `label uniq: [0]`)
5. `[T2-checkpoint-surrogate]` - `src/clinical/exhaustive_evaluator.py:322-347` (model trong so ngau nhien vao bang benchmark)
6. `[T1-flaky-drive-bundle-tests]` - `tests/test_thin_drive_runner.py:96`, `:175` (2/6 lan fail, PermissionError WinError 32)
7. `[T9-tests-write-into-repo]` - `tests/test_thin_drive_runner.py:96` (`tmp_path` bi bo rong, test ghi de `drive_upload_ready/` va `product/proposal/assets/`)

---

## 12. Cong Bo Ve Anh Huong Cua Qua Trinh Audit (Minh Bach)

Toi bi rang buoc READ-ONLY va **da khong sua** bat ky file source, test, hay script nao. Tuy nhien, **hai lenh kiem chung cua toi co tac dong ghi de** den trang thai repo, vi ban than bo test khong isolated (muc 9.4 - day la mot FINDING, khong phai loi cua toi):

1. `python -m pytest tests/ -q -p no:cacheprovider` (muc 1.1) -> goi `scripts/build_drive_package.py` -> **ghi de** `drive_upload_ready/SE374_WSSS_Drive_Code/` va `drive_upload_ready/SE374_WSSS_Drive_Code.zip`.
2. `python -m pytest tests/test_thin_drive_runner.py ...` x6 (muc 1.3, 1.4) -> nhu tren.

`git status --short` sau audit cho thay `M product/proposal/assets/preview/pages/page_{1,2,3,4,5,6,7,8,9,10}.png` (10 asset bi tai tao boi `tests/test_deliverable_auditor.py` hoac draw-*.py trong luong test), cung nhieu `M src/*.py` da ton tai **truoc** toi (trang thai working tree cua repo, khong phai do toi - toi khong dung Edit/Write lenh bat ky tep source nao).

**Kien nghi cho Lead Agent:** `git checkout -- drive_upload_ready/ product/proposal/assets/` de khoi phuc, va **sua `tests/test_thin_drive_runner.py` dung `tmp_path`** (thuoc ve Fixer Swarm, khong phai toi).

Lenh da chay (toan bo read-only ngoai 2 muc tren): `pytest` x8, `ruff check` x6, `grep`/`find`/`ls`/`diff`/`md5sum`/`wc`, `python -X utf8` doi chieu JSON + notebook lifecycle + patient-leakage tren du lieu that + tai lap `DataLoader` that (khong doc/ghi file ngoai thu muc tam va `train_PM` chi duoc doc).

---

## 13. Ket Luan Tong The Gui Lead Agent

**Gia dinh dat ra ("tuyen bo 348/0 CO THE SAI hoac PHONG DAI") duoc chung minh la DUNG.** Ca 4 tuyen bo cua audit 2026-11-02 deu sai hoac phong dai o muc khac nhau:

- `348 passed / 0 failed`: **sai** (348 la so test, khong phai so pass).
- `ruff 0 errors`: **phong dai** (0 trong `src` va `tests`, nhung 33 trong `scripts` va 464 trong toan repo).
- `0 non-ASCII hyphen`: **sai ve chu nghia** (230 dong trong `.agents`/`graphify-out`), **dung ve noi dung** (`src`/`tests`/`notebooks`/`product`/`docs` sach).
- `0.0% patient leakage`: **dung ve ket qua**, nhung **phuong phap chung minh cua bao cao cu khong the kiem chung dieu do** - chi khi nao do truc tiep 31,826 file tren dia nhu toi moi co ket luan co gia tri.

**Dieu quan trong hon nhung con so tren:** bo test 348 test la mot bo test **tot** (mock toi thieu, khong sleep, khong thu tu phu thuoc, co hypothesis fuzz, co du lieu that) - nhung no **hoan toan huong vao `src/` va quyet bo qua `product/`**. 7/7 defect Critical cua toi nam o mot trong hai cho: (a) `product/reports/` (san pham nop), hoac (b) cac nhan `fallback` trong `src/` ma **khong test nao dat lenh goi voi du lieu that/thieu**. Mot swarm audit chi chay pytest va ruff se **luon luon** ket luan "PASS" cho mot repo nhu the. Do la ban chat cua Anti-Mock-Testing Illusion Gate ma INVARIANT-26 da canh bao - va audit truoc do chinh xac la no nan nhan.

**Kien nghi uu tien cho Fixer Swarm (toi KHONG sua - theo Builder-Auditor Separation):**

1. Ngat ngay `product/reports/clinical/EXHAUSTIVE_3WAY_BENCHMARK_REPORT.md` khoi danh sach nop; chay lai `exhaustive_evaluator.py --full` tren 4,986 patch va thay bang.
2. Chan ngat `raise` (khong `return`) o 3 fallback: `dataset_loader.py:323`, `bcss_dataset.py:132`, `exhaustive_evaluator.py:439`.
3. Bo `raise`/ghi co che `checkpoint_loaded: bool` trong `_build_deeplabv3plus` va dua no ra JSON ket qua.
4. Xoa nhanh `except Exception: -> DeepLabV3PlusWithCGM()` (`exhaustive_evaluator.py:337-347`).
5. Sua `tests/test_thin_drive_runner.py` dung `tmp_path` + cho phep `build_drive_package(output_dir=...)`.
6. Cai du `requirements.txt` trong CI (`pip install -r requirements.txt`) va xoa `|| true`/`--exit-zero` khoi buoc lint.

---

**End of Report - AUDITOR 06**
