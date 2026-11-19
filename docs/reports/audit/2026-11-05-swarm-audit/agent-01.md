# SE374.R11 - AUDIT SWARM 2026-11-05 - AGENT 01 (Independent Isomorphic Auditor)

- **Vai tro:** AUDITOR 01 trong swarm 8 tac tu doc lap (prompt dong nhat, zero-decomposition).
- **Pham vi:** Toan bo 10 chieu kiem dinh T1-T10 tren khap repo (khong chon module hep).
- **Repo:** `SE374_WSSS_WORKSPACE` (remote `medical-wsss-consortium/medical-wsss`, default branch `master`, HEAD `2248e7b`).
- **Moi truong:** Windows + Git Bash, Python 3.13.9, pytest 8.4.2, ruff 0.12.0, torch 2.10.0+CPU, Pillow 12.0.0, KHONG GPU.
- **Quy uoc bang chung:** `VERIFIED` = co chay lenh va output that duoc dan; `INFERENCE` = doc ma/phan tich ty luan, chua duoc chay.
- **Ky luac vai tro:** Khong sua source, khong sua test, khong tao script moi. Chi doc, chay lenh read-only, va ghi duy nhat tep nay.

---

## 0. Ket luan dieu hanh (Executive)

Ban audit truoc (`docs/reports/audit/2026-11-02-swarm-audit/01_SWARM_RECHECK_VERIFICATION.md`, muc 1.6, 2.1, 4.1, 5) tuy bo 4 dieu:

| Tuy bo cua ban 2026-11-02 | Thuc nghiem Agent 01 | Ket luan |
| :--- | :--- | :--- |
| `348 passed, 0 failed` | `5 failed, 343 passed, 2 warnings in 455.12s` | **SAI** (dinh so 348 dung, trang thai 0-fail SAI) |
| `Ruff 0 lint errors` | `src/` 0, `tests/` 0, `scripts/` **33 errors** | **SAI MOT PHAN** (scope bi thu hep) |
| `0 non-ASCII hyphens across the entire workspace` | `src|tests|scripts|notebooks|product|docs` = 0; `.agents/` = **10 tep co em/en-dash** | **SAI** (tuyen bo toan-vung) |
| `Exactly 32 .ipynb files` | `find . -name "*.ipynb" -not -path "./repos/*"` = **60** | **SAI** |
| `0.0% patient leakage (151 cases)` | Khop 100% voi dia that: train 106 / val 22 / test 23, pairwise overlap = 0 | **DUNG** |
| CI/CD gate (INVARIANT-03) dang hoat dong | `gh api .../actions/runs -> total_count = 0` (CI chua tung chay) | **SAI** |

**Ngoai ra, phat hien mang tinh he thong:** bo test `tests/` **tuy chinh sua chinh workspace** (`build_variants()` viet de notebook, `build_drive_package()` rmtree + rebuild `drive_upload_ready/`). Day la bang chung truc tiep cho **Anti-Mock-Testing Illusion Gate** (INVARIANT-26): mau "xanh" mot phan do chinh test tu sinh ra.

**Tong hop defect:** 5 Critical, 8 High, 9 Medium, 4 Low (chi tiet muc 1-10).

---

## 1. T1 - Tang 1: Test suites tu dong

### 1.1 Lenh da chay (VERIFIED)

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q -p no:cacheprovider
```

Output ( dong cuoi):

```text
5 failed, 343 passed, 2 warnings in 455.12s (0:07:35)
```

Dinh so test thu duoc = 348 (khop con so cua ban cu), **nhang 5 test FAIL**.

### 1.2 Chu ky 5 loi te nhat (VERIFIED - truncated nguyen van tu pytest)

**(1) `tests/test_colab_notebook.py::test_integration_engine_and_colab_generator_parity`**

```text
scripts\generate_colab_variants.py:337: AssertionError
E   AssertionError: Missing Stage 1 ID in notebooks\SE374_Colab_WSSS_T4_Full4986.ipynb
    gen_module.build_variants()   # tests\test_colab_notebook.py:924
```

**(2) `tests/test_thin_drive_runner.py::test_2_drive_bundle_packager_creates_complete_folder_and_zip`**

```text
scripts\build_drive_package.py:1545: in build_drive_package
    shutil.rmtree(bundle_dir)
E   OSError: [WinError 145] The directory is not empty:
    'C:\Users\...\Y_KHOA\drive_upload_ready\SE374_WSSS_Drive_Code'
```

**(3) `tests/test_thin_drive_runner.py::test_3_new_thin_runner_notebooks_valid_concise_and_modular`**

```text
E   FileNotFoundError: [Errno 2] No such file or directory:
    '...\drive_upload_ready\SE374_WSSS_Drive_Code\notebooks\history_runs\Run3_...T4_Full4986.ipynb'
```

**(4) `tests/test_thin_drive_runner.py::test_7_invariant_12_strict_ascii_hyphens_across_all_packaged_files`**

```text
E   PermissionError: [WinError 32] The process cannot access the file because it is
    being used by another process: '...\drive_upload_ready\SE374_WSSS_Drive_Code\src\pipeline\wsi_lowmag_highmag_pipeline.py'
```

**(5) `tests/test_training_service.py::test_train_stage2_dry_run`** - `RuntimeError` (chi xuat hien trong lan chay full, xem 1.3).

### 1.3 Test thu lai cach ly (VERIFIED - chung minh phu thuoc thu tu / tran thai)

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_thin_drive_runner.py \
  tests/test_training_service.py::test_train_stage2_dry_run \
  "tests/test_colab_notebook.py::test_integration_engine_and_colab_generator_parity" \
  -q -p no:cacheprovider
# -> 3 failed, 9 passed in 101.42s
```

Nghiã là: **2/5 loi bien mat khi chay lan 2**, vi chinh lan chay 1 da viet de file notebook/manifest ma no dan chieu. Day la dang loi "test tu lam xanh minh".

**[T1-pytest-red]** | *Critical* | do tin cay 100% | `docs/reports/audit/2026-11-02-swarm-audit/01_SWARM_RECHECK_VERIFICATION.md:33,136`
He qua: tuyen bo "0 fail" trong bao cáo nghiem thu la sai le that; sinh vien nh nop so lieu tu mot bao cáo xanh gia.

**[T1-tests-mutate-repo]** | *Critical* | 100% | `tests/test_thin_drive_runner.py:32,98,180,493`; `tests/test_colab_notebook.py:921-924`; `scripts/generate_colab_variants.py:14,301-341`; `scripts/build_drive_package.py:1525-1545`

```python
def test_2_drive_bundle_packager_creates_complete_folder_and_zip(tmp_path):   # tmp_path KHONG duoc dung
    summary = packager.build_drive_package()                                  # rmtree THUAT directory that
```

Bang chung VERIFIED: sau lan chay pytest, `find notebooks drive_upload_ready -newermt "-25 minutes"` liet ke 6 notebook trong `notebooks/` + toan bo cay `drive_upload_ready/SE374_WSSS_Drive_Code/` + `SE374_Colab_Scientific_Research_Runner.ipynb` goc bi viet lai.
He qua: (a) bo test khong hiem khiet (khong hermetic), pha huy deliverable khi dev/auditor chi muon do; (b) ket qua phu thuoc thu tu chay va tran thai -> khong the tai lap; (c) **kinh nghiem cua chinh agent nay**: `drive_upload_ready/` bi rmtree mot phan trong luc agent dang doc song song -> sinh ra WinError 5/32/145. Agent KHONG phuc hoi duoc nguyen trang ban dau (tep thuoc .gitignore, khong co baseline trong git); hiên tại bundle đã được rebuild (88 file) va `experiments/` goc van nguyen ven (28 file).

### 1.4 Static analysis (VERIFIED)

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m ruff check src      # All checks passed!
KMP_DUPLICATE_LIB_OK=TRUE python -m ruff check tests    # All checks passed!
KMP_DUPLICATE_LIB_OK=TRUE python -m ruff check scripts  # Found 33 errors.
KMP_DUPLICATE_LIB_OK=TRUE python -m ruff check src tests scripts  # Found 33 errors.
```

Phan bo 33 loi (concise): `I001` x8 (import block chua sap xep), `F401` x7 (import thua: `time`, `urllib.request`, `re`, `pathlib.Path` x3, `os`), `E741` x16 (ten bien `l` mo ho), `F841` x1 (`old_drive_block` gan khong dung). Toan bo nam trong `scripts/`.
**[T1-ruff-scripts]** | *Medium* | 100% | `scripts/*.py` (vi du `scripts/download_papers.py:4-5`, `scripts/update_agents_invariants.py:1`, `scripts/refactor_notebooks_pure_colab.py:42`).
He qua: tuyen bo "0 errors" trong bao cáo la thu hep scope (chi `src/`) de qua gate.

---

## 2. T2 - Zero-Mockup (INVARIANT-07)

### 2.1 Phat hien nang: duong hoi synthetic fallback trong huan luyen MedSAM

**[T2-synthetic-fallback-train]** | *High* | 100% (doc ma + timing) | `src/training/train_proposed_sam.py:485-500`

```python
# Dataloaders: use real data if exists, else synthetic fallback
if os.path.exists(args.data_dir) and not args.dry_run:
    ...
else:
    num_samples = 4 if args.dry_run else 32
    train_ds = SyntheticHistologyDataset(num_samples=num_samples, image_size=224)
    val_ds   = SyntheticHistologyDataset(num_samples=num_samples // 2, image_size=224)
...
total_epochs = 1 if args.dry_run else args.epochs
```

`SyntheticHistologyDataset` (`:195-217`) tra anh `torch.rand(...)` va mask `torch.randint(0,5,...)`. Neu `--data-dir` sai/missing **ma khong co `--dry-run`**, chuong trinh huan luyen du `args.epochs` (mac dinh 25) tren nhieu ngau nhien, roi `save_checkpoint(... "best_mIoU": best_miou, "val_stats": val_stats)` (`:539-550`) va in `[INFO] Completed successfully. Best mIoU: ...`.
He qua lam sang/khoa hoc: metric gui trong `.pth` va log co dang bi dung nhu ket qua that tren BCSS. Cung luc, duong nay vi pham INVARIANT-08 (train/val synthetic khong co Patient ID).
Ghi chu: duong nay **khong** vi pham phan nhap hoi khi file thieu (khac voi muc 2.3), boi no co in dong thong bao.

### 2.2 Proxy CAM duoc dung thay CAM that

**[T2-surrogate-cams]** | *Medium* | 100% | `src/training/train_proposed_sam.py:254-265`

```python
# When CAMs are not precomputed, construct surrogate CAMs from weak labels
surrogate_cams = torch.zeros(b, 4, h, w, device=device)
for c in range(4):
    active_b = weak_labels[:, c].view(b, 1, 1)
    surrogate_cams[:, c] = (images.mean(dim=1) * active_b).clamp(0.0, 1.0)
```

Cam "gia" lay tu **cường độ anh trung binh**, khong phai CAM cua Stage-1 ResNet38d/NetCAM. Neu prompt cua MedSAM duoc bao cao la "CAM-prompted", day la khac voi phuong phap cong bo. **INFERENCE** (chua chay duong nay tren du lieu that).

### 2.3 Fallback mask = 0: metric tinh tren ground-truth gia

**[T2-zero-mask-fallback]** | *Medium* | 100% (doc ma) | `src/training/dataset_loader.py:324-326`, `:456-458`; `src/datasets/bcss_dataset.py:132-133`

```python
# dataset_loader.py:456-458 (BCSSEvalDataset)
else:
    # Fallback zero mask if mask folder is absent
    mask_np = np.zeros((224, 224), dtype=np.int64)
```

Khi thu muc mask sai duong dan (BCSS that dung `mask/`, mot so ban phat hanh dung `label/`), **toan bo** ground-truth thanh "toan background". IoU/Dice luc sau khong crash, chi tra ve so lieu vo nghia. Day la "che loi bang ket qua vo nghia" - dung chi phan mo ta cua muc T2.
**VERIFIED** mot phan: `BCSSEvalDataset(mask_dir='/tmp/khong_co')` khong bao gio loi (xem T4/T3).

### 2.4 Metric bi lang le khi tinh loi -> trung binh tren tap con

**[T2-metric-silent-drop]** | *High* | 100% | `src/colab_pkg/evaluator.py:172-182`

```python
try:
    p_fwiou = compute_fwiou(pred, target, num_classes=self.num_classes)
    self.patch_fwiou.append(p_fwiou * 100.0)
    ...compute_boundary_f1...
except Exception:
    pass
```

FwIoU va Boundary-F1 bi loai khoang le khi loi, nhung danh sach dung de lay trung binh **khong ghi so luong phan tu bi mat**. Ket qua: bang so lieu trong `ablation_study_metrics.csv` (cot FwIoU/BF1) co the la trung binh tren tap con benh nhan, khong co chu thich. Vi pham INVARIANT-07/09 ve minh bach so lieu.

### 2.5 Nguon goc so lieu trich dan SOTA khong kiem chung duoc

**[T2-citation-provenance]** | *High* | 100% (so cross-file) | `experiments/benchmarks/COLAB_3_FULL_RUNS_EVOLUTION_COMPARISON.md:43-44`; `docs/research/SE374_CHRONICLE_RESEARCH_RESTRUCTURING_AND_DEFECT_RESOLUTIONS.md:106-107`

```text
| **WSSS-Tissue (Stage-2 DeepLabV3+ Paper)** | MedIA 2022 | 69.34% | 78.44% | 73.31% | 59.98% | 65.62% |
| **LowMagWSS (40x Benchmark)**            | arXiv 2024 | 69.34% | 78.44% | 73.31% | 59.98% | 65.62% |
```

Hai cong trinh khac nhau (Han et al. 2022 va Dung et al. 2024) duoc gan **5 chi so giong het nhau den 2 chu so thap phan**. Khi kiem tra nguon dang trong repo:

```bash
grep -nE "69\.[0-9]{2}|78\.44|73\.31|59\.98|65\.62" \
  docs/papers/WSSS_Tissue_arxiv_2112.03053.md docs/papers/LowMagWSS_arxiv_2607.10783.md
# -> khong co khop nao
grep -niE "miou|m-iou" docs/papers/WSSS_Tissue_arxiv_2112.03053.md   # -> 0 dong
wc -l docs/papers/WSSS_Tissue_arxiv_2112.03053.md                    # -> 91 dong
```

Nghĩa là: **bao cáo cua de tài** (tệp PDF gốc 201KB-808KB) duoc ingest thanh file `.md` chi 91 dong va **khong he co bang ketqua**. So lieu trich dan do do **khong doi chieu duoc** theo tinh than INVARIANT-07 ("trich dẫn chính xác từ paper gốc") va INVARIANT-19 ("bắt buộc ingest toan văn chi tiết").
Mâu thuân nội bộ: `product/proposal/SE374_MINI_PROPOSAL_WSSS_WSI.md:69` cung cit Han et al. MedIA 2022 nhung bang **mIoU 64.20%** tren BCSS, khac hăn 69.34% trong bảng benchmark.
Xep loai: **High** (trich dan thieu minh chung + mot dong co dangnghi copy-paste), KHONG quy là "bịa số liệu" vì chưa loại trừ khả năng LowMagWSS trích lại baseline cua Han et al. trong bảng của ho.

### 2.6 Cac nghi van DA LOAI (report ca dieu KHONG tim thay)

- `src/colab_pkg/downloader.py:611,647` `allow_mock: bool = False` (mac dinh an toan) va **khong co bat ky caller nao** truyen `allow_mock=True` (VERIFIED: `grep -rn "allow_mock" src scripts` chi ra 2 dong dinh nghia).
- `src/colab_pkg/evaluator.py:221,492` va `src/models/coarse_to_fine_router.py:373`: `np.random.RandomState(seed)` dung cho **bootstrap CI / resampling thong ke hop le**, khong phai gia nghiem.
- `src/clinical/exhaustive_evaluator.py:699,719,723`: khi thieu du lieu song ton tai, he thong **ghi status chu hoa** (`"Insufficient stratified cohorts for logrank test; no dummy patients injected."`) thay vi nhan benh gia. UNH vi INVARIANT-07.
- `src/colab_pkg/colab_runner.py:597-656`: `torch.rand` / `np.random.randint` nam trong `run_package_self_test()` (self-test pipeline), khong bong ra bang ket qua.
- Cac so lieu cot loi (69.34/70.49/76.38...) **co log that**: `experiments/colab_run_2_dual_stage_ensemble/COLAB_EVALUATION_REPORT.md:10-28`, `experiments/colab_run_3_multiscale_sota_v2/ablation_study_metrics.csv:2,4`, `.../COLAB_EVALUATION_REPORT.json:118,120,342`. -> Pham tru "bịa metric" cho pipeline chinh.
- Quet token: `grep -rniE "\b(placeholder|synthetic|dummy|fake|mock|simulated)\b" src --include="*.py"` = 19 khop, **da mo het 19 tep kiem nguat ngữ**, chi muc 2.1-2.4 la rung.

---

## 3. T3 - Patient-level anti-leakage (INVARIANT-08)

### 3.1 Ket luan SACH co bang chung (VERIFIED - phat hien quan trong nhat cua T3)

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -c "<doi chieu manifest vs dia that>"
```

```text
counts files (train/val/test)      23422 3418 4986
unique patients train/val/test     106  22   23
manifest sizes                     106  22   23
REAL train-vs-val patient overlap: [] 0
REAL train-vs-test overlap:        0
REAL val-vs-test overlap:          0
manifest train not present in training dir: 0
manifest val not in val/img:                0
manifest test not in test/img:              0
patients in training dir NOT in manifest train: 0
```

Manifest `src/config/bcss_patient_splits.json` (151 benh nhan, `train_count=106, val_count=22, test_count=23`) **trunkhop 100% voi phan chia dia ly cua `data/BCSS-WSSS`** (0 overlap). Tuyen bo "0.0% leakage" trong ban cu la **DUNG** o tang manifest va tang file that.

### 3.2 Defect: filter benh nhan bi **tat lang le** khi manifest rong

**[T3-filter-disabled-when-empty]** | *High* | 100% (doc ma, dong roi ro) | `src/training/dataset_loader.py:501-512` va `:527-531`

```python
splits = load_bcss_patient_splits(splits_config)
train_patients = set(splits.get("train", []))
...
train_dataset = BCSSStage1Dataset(
    root_dir=train_dir,
    patient_ids=train_patients if train_patients else None,   # <-- NULL HOA FILTER
    ...)
```

`BCSSStage1Dataset` (`:198-208`) chi loc theo benh nhan khi `patient_ids is not None`. Neu manifest bi empty/corrupt-but-parsable (`{"train": []}`), **toan bo 31.826 patch cua ca 151 benh nhan (gom ca 23 benh nhan TEST) roi vao tap train** -> ro ri LEAKAGE hoan toan ma khong co bat ky loi/canh bao nao. Cung kieu go cho `val_dataset` (`patient_ids=val_patients if val_patients else None`).
He qua lam sang: mIoU tang ao, ket luan khoa hoc vo gia tri.
**INFERENCE** (chua dung manifest rong de chay, vi quy dinh khong ghi file trong `src|tests`).

### 3.3 Defect: fallback hard-code 6 benh nhan

**[T3-hardcoded-fallback]** | *High* | 100% | `src/training/dataset_loader.py:127-133`

```python
if not config_path.exists():
    # Fallback to local default patient lists if config file does not exist
    return {
        "train": ["TCGA-A1-A0SK", "TCGA-A2-A04N", "TCGA-A2-A0T2"],
        "val":   ["TCGA-AN-A03X", "TCGA-AN-A0AL"],
        "test":  ["TCGA-AQ-A04H", "TCGA-C8-A1HJ"],
    }
```

Manifest thieu -> he thong van chay "thanh cong" tren 3/2/1 benh nhan, trong khi `src/datasets/patient_split.py:131-132` (han song) thi **raise FileNotFoundError**. Hai ham cung ten chuc nang co han vi khac nhau hoan toan.

### 3.4 Defect: hai ban `extract_patient_id` phan ky

**[T3-duplicate-extract-patient-id]** | *Medium* | 100% | `src/datasets/patient_split.py:20` vs `src/training/dataset_loader.py:63`

```python
# patient_split.py (dung de verifier chong leakage)
tcga_match = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", basename, re.IGNORECASE)
# dataset_loader.py (dung de PHAN CHIA tap)
tcga_match = re.match(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", fname, re.IGNORECASE)
```

`re.search` (bat ky vi tri trong basename) khac `re.match` (dau ten file) va khac luon dau vao (`basename` vs `os.path.splitext(...)[0]`), dong thoi ban dataset tra `token.strip()` khong hoa, ban loader tra `.upper()`. Mot file nhu `Crop_TCGA-A2-A0T2_x.png` se duoc **verifier** coi la `TCGA-A2-A0T2` nhung **loader** coi la `CROP`. Khi do kiem dinh chong leakage (bang `verify_patient_leakage`) danh gia tren mot khong gian ID khac voi khong gian ID ma pipeline thuc chat dung de chia tap -> leakage co the triet de khi qua mat verifier. Vi pham INVARIANT-05/06 (trung lap module).

### 3.5 Phat xet ve `tests/test_patient_split.py`: test co chung minh dieu no assert?

- **Co chung minh that (khong ty tham chieu):** `test_verify_patient_leakage_on_sample_dataset` (`:32-49`) doc file tren `data/sample_dataset` va dat `train=2, val=1, test=1` benh nhan; `test_verify_patient_leakage_catches_overlap` (`:52-60`) kiem duong ERROR (co overlap -> phai `ValueError`). Day la 2 test han vi that.
- **Ty tham chieu mot phan (Medium defect):** `test_151_patient_split_manifest_zero_leakage` (`:94-145`) chi doc manifest roi assert "tap ID roi nhau". Ve mat toan hoc, `set(train) & set(val) == empty` cua **cung mot tep JSON** la **hang dang thuc logic** - no chung minh tep cau truc dung, **khong** chung minh pipeline chia du lieu theo patient. Test cung khong doi chieu voi `data/BCSS-WSSS` (phai chinh phan agent 01 lam, xem 3.1). Khuyet diem nay nang ne vi ban cu (muc 2.2, muc 3.1) dung chinh no lam "bang chung 0.0% leakage".
- **Test duong bien:** `test_load_patient_splits_invalid_path` (`:148-151`) kiem `FileNotFoundError` - dat, ho tren duong loi. Hypothesis fuzz (`:162-174`) dat, do tin cay cao.
- **Thieu hoan toan:** khong test "manifest rong -> filter tat", khong test "mask missing -> zero mask", khong test doi chieu manifest vs du lieu dia (chinh la 3 defect tren).

---

## 4. T4 - INVARIANT-29: chong glob nong va luoi an toan

### 4.1 Vi pham glob nong co tai lap bang lenh (VERIFIED)

**[T4-shallow-glob]** | *High* | 100% | `src/datasets/bcss_dataset.py:48-72`

```python
train_dir = os.path.join(self.root_dir, "training")
if not os.path.isdir(train_dir):
    train_dir = os.path.join(self.root_dir, "train")
pattern = os.path.join(train_dir, "*.png")
self.image_paths = sorted(glob.glob(pattern))       # <-- NONG, khong rglob, khong resolve_split_path
```

Tai lap tren moi tam `tempfile` (khong ghi vao workspace):

```text
real png count under train/img: 3
BCSSDataset len with nested train/img layout = 0
LOADER ERROR -> ValueError : num_samples should be a positive integer value, but got num_samples=0
non-existent root -> len = 0 ; no exception at __init__
```

Nghiã là Dataset **lang le tra danh sach rong** khi cau truc long hon 1 cap, roi PyTorch crash o tang DataLoader (dung nhu quy luat INVARIANT-29 mo ta). Tren may nay duong `training/*.png` may man "phang" (23.422 anh nam ngay trong `training/`) nen loi **bi an**: neu tai BCSS-WSSS goc tu HuggingFace/Challenge (cau truc `train/img/*.png`) tren Colab -> sập.

### 4.2 Luoi an toan (safety-net) cua INVARIANT-29: **KHONG TON TAI**

```bash
grep -rn "def resolve_split_path|placeholder|smoke-test patch|safety net|len(files) == 0" src --include="*.py" -i
# -> 0 khop trong tang dataset (chi co 2 khop o src/utils/deliverable_auditor.py lien quan typography)
```

**[T4-no-safety-net]** | *High* | 100% | Toan bo `src/datasets/`, `src/training/dataset_loader.py`, `src/clinical/exhaustive_evaluator.py`.

### 4.3 Cac loader khac

| Loader | Do tim kiem | Danh gia |
| :--- | :--- | :--- |
| `src/training/dataset_loader.py:154-172` `find_image_files` | `root_path.rglob("*")` (de quy) + filter suffix | **DAT** INVARIANT-29 |
| `src/colab_pkg/downloader.py:139-176` (`resolve` split) | Try truc tiep `sp/img` + `base_path.glob(f"**/{sp}/img")` + loc `_is_mock_path` + 2 pass (>=10anh/>=2benh nhan, roi fallback min) | **DAT**, pham chat cao |
| `src/datasets/bcss_dataset.py:55,72` | `glob.glob("<split>[/<alt>]/*.png")` nong | **VI PHAM** (4.1) |
| `src/clinical/exhaustive_evaluator.py:388-394` | `dataset/test/img` -> fallback `dataset/test` -> `img_dir.glob("*.png")` nong | **VI PHAM** (chi 2 cap co dinh) **[T4-shallow-glob-evaluator]** *Medium* |

### 4.4 Nghia vu doi chieu T2 x T4

Safety-net **duoc thiet ke** nhu placeholder/anh gau moc de DataLoader khong crash. Neu ap dung theo dung ngu nghia cua INVARIANT-29 ("kich hoat placeholder/smoke-test patch"), thi no **vi pham INVARIANT-07** trong `tests/`. **VERIFIED**: `src/colab_pkg/downloader.py:611` (`allow_mock`) da duoc thiet ke dung cach (mac dinh False va **khong caller nao** bat), va trong repo **khong co placeholder naof trong duong danh gia** - do do **chua co vi pham T2 tu huong nay**. Nhung cung chinh vi "khong co safety-net" nen repo roi vao 4.2. De xuat cho Lead: safety-net phai gan co che **aborted = True** va **cấm** ghi metric ra artifact.

---

## 5. T5 - INVARIANT-28 + INVARIANT-27 (vong doi Colab)

### 5.1 Vi pham INVARIANT-28: `drive.mount` trong notebook Colab

**[T5-drive-mount]** | *High* | 100% (VERIFIED grep) | 23 tep `.ipynb` (ngoai `repos/`) chua `drive.mount`, gom **ca 4 runner trong goi ship**:

```bash
grep -rl "drive.mount\|/content/drive" --include="*.ipynb" . | grep -v repos/ | wc -l   # -> 23
```

Trich `notebooks/SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb`, cell 2:

```python
if "google.colab" in sys.modules:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception: pass
MYDRIVE = Path("/content/drive/MyDrive")
DRIVE_ZIPS = [MYDRIVE / "SE374_WSSS_Project/SE374_WSSS_Drive_Code.zip", ...]
```

Va `SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb` (goc + notebooks + drive bundle) co ghi ro `# 1. Mount Google Drive`.
INVARIANT-28 cam tuyet doi dua `drive.mount`/ep OAuth trong notebook. `try/except` khong loai bo duoc pop-up OAuth tuong tac: khi Run All, `drive.mount` van dung de yeu cau cap quyen -> vi pham nguyen tac "100% tu dong, khong dung lai doi cap quyen". Cac notebook *khong* co drive.mount (da liet ke 29 tep, gom `01..04_*_T4/L4`, `SE374_Colab_WSSS_{T4,L4,A100}_Full4986.ipynb`, `T4_Sanity150`) la **SACH** - chung minh doi da co ban "pure local disk" song song, nhung ban vi pham van duoc **ship trong `drive_upload_ready/`** (goi dung cho nguoi dung).

### 5.2 vong doi 7 buoc (INVARIANT-27) - ma tran bo 60 notebook

Da chay phan tich JSON + regex tren **60** notebook va xuat bang (coc: `drive / content_drive / files.download / gpu_check / pip install / gdown / wget / huggingface / torch.load / autocast / GradScaler / zip / MIoU / Dice`). Ket luan tong hop:

| Nhom | GPU check | Cai lib | Tai du lieu THAT | Nap weights | Do mIoU/Dice | truc quan hoa lam sang | ZIP + `files.download` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `01_Train_Stage1_*` (4 ban) | CO | CO | CO (wget+HF) | CO | **THIEU mIoU** (chua giai doan classifier AU AUC/F1) | THIEU | CO |
| `02_Pseudo_Masks_*` (3) | CO | CO | CO | CO | CO (MIoU) | CO | CO |
| `03_Train_Proposed_MedSAM_*` (3) | CO | **THIEU pip** | **THIEU wget/gdown/HF** (chi HF) | CO | **CO Dice, thieu cot MIoU** | CO (autocast/Dice) | **THIEU files.download** |
| `04_Exhaustive_Benchmark_Survival_*` (3) | CO | **THIEU pip** | **THIEU wget/gdown/HF** | **THIEU torch.load** | CO | **THIEU autocast/GradScaler** | **THIEU files.download** |
| `*_ThinDriveRunner_v2` (8 ban/4 phien ban) | CO | **THIEU pip** | **QUA Drive (vi pham 28)** | **THIEU torch.load truc tiep** (uy quyen `src`) | CO | CO (ham `src`) | **KHONG co `files.download` trong notebook** (goi `package_and_download_results()` tu `src/colab_pkg/visualizer.py:399,439`) |
| `SE374_Colab_WSSS_{T4,L4,A100,Pipeline,Sanity150}` (5) | CO | CO | CO (wget+HF+gdown) | CO | CO | CO | CO |
| `SE374_Colab_Scientific_Research_Runner{,_FIXED}` + archive | **THIEU gpu_check** (`torch.cuda.is_available` = "-") | **THIEU pip** | CO | **THIEU torch.load** | CO | CO | CO |
| `BonusDemo` (3 ban) | CO | CO | CO | **THIEU torch.load** | **THIEU MIoU va Dice** | **THIEU** | CO |

**[T5-lifecycle-gaps]** | *Medium* | 100% | Cac o "THIEU" o tren.
Ghi chu cong bang: **khong notebook nao** chua `shutil.make_archive` truc tiep (cot `zip` = "-" tren 60/60); viec dong goi ZIP duoc day sang `src/colab_pkg/visualizer.py:399-439`. Do do buoc 7 "dong goi ZIP va tu tai ve" phu thuoc 100% vao package `src/` duoc nap tu Drive -> nghiep vu vong doi kep chat che voi 5.1.

### 5.3 Drift gitua cac phien ban notebook (VERIFIED md5sum)

```bash
md5sum *.ipynb notebooks/*.ipynb drive_upload_ready/SE374_WSSS_Drive_Code{,/notebooks}/*.ipynb notebooks/modular_drive/*.ipynb
```

| Cặp được đối soát | md5 | Ket luan |
| :--- | :--- | :--- |
| goc `01_Train_Stage1_Classifier.ipynb` vs `notebooks/` cung ten | `7b375fde...` vs `c635f082...` | **DRIFT** |
| goc `01_Train_Stage1_Classifier_L4.ipynb` vs `notebooks/` | `8d49792b...` vs `0f5073af...` | **DRIFT** |
| goc vs `notebooks/`: `01.._T4`, `02.._T4`, `03.._T4`, `04.._T4`, `Scientific_Research_Runner` | giong het | SACH |
| `SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb`: goc / `notebooks/` / `drive_upload_ready/` | `05e42e21` / `266f6763` / `045fa19f` | **3 BAN KHAC NHAU HOAN TOAN** |
| 4 runner `*_ThinDriveRunner_v2`: `notebooks/` == `notebooks/modular_drive/` (hom bo) vs `drive_upload_ready/{,notebooks/}` | `c127184e`/`70533ff0`/`35b09c4b`/`2250fea4` vs `ab370de9`/`b16c971b`/`f39ae743`/`a23f17c7` | **DRIFT**: goi ship cho nguoi dung khac ban trong `notebooks/` |
| `notebooks/SE374_Colab_Scientific_Research_Runner.ipynb` vs `..._FIXED.ipynb` | `87bfb7a2` vs `2b677408` | **DRIFT** (hai "runner chinh" song song, khong ro ban nao sinh ra `SE374_Scientific_Research_Results*.zip`) |
| goc `SE374_Colab_WSI_..._v2.ipynb` (686KB) vs cac ban khac | `c54268fb` | **BAN CHI TON TAI Ở GOC, khong đồng bo** |

**[T5-notebook-drift]** | *Medium* | 100% | Nhu bang tren.
He qua: khong the xac dinh **ban notebook nao** da sinh ra **zip ket qua nao** (`SE374_Colab_Results_DeepLabV3Plus_T4_Full4986_20261020_205548.zip` vs `notebooks/history_runs/Run3_20261020_2055UTC_...`) -> mat chuoi bang chung thuc nghiem (INVARIANT-07/14).

### 5.4 Dinh danh tai nguyen (INVARIANT-27 chieu 2-4) (VERIFIED)

Trong `notebooks/SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb`: `MODE_TAG="T4_Full4986"`, `SANITY_MODE=False`; `BATCH_SIZE = 12` (runner) / `BATCH_SIZE = 16` (nhom 01/02). Metadata `colab.gpuType` duoc generator gan theo variant (`scripts/generate_colab_variants.py`). **THIEU**: khong co uoc tinh VRAM dinh theo batch size va khong co bang thoi gian tung stage duoc ghi trong notebook; `set_reproducible_seed()` (`src/colab_pkg/colab_runner.py:99-124`) co in ten GPU + VRAM **chi khi CUDA san** (tuc la khong co uoc tinh cho truong hop T4 khi code chay CPU-fallback). Danh gia: **Medium**, vi INVARIANT-27 yeu cau bao cao uoc thoi gian truoc khi xin phep.

---

## 6. T6 - Device & tai nguyen (audit tinh + 1 tai lap VERIFIED)

### 6.1 `torch.load` va `weights_only` (PyTorch >= 2.6)

```bash
grep -rn "torch.load" src scripts --include="*.py"
```

- `src/colab_pkg/model_loader.py:574-576`, `src/models/wsss_inference_methods.py:363`, `src/colab_pkg/downloader.py:230-232`: try `weights_only=True` -> `except Exception` -> try `weights_only=False`. **DUNG** huong xu ly (kha phat hien 2.6). Ghi chu an toan: downgrade lang le sang `weights_only=False` tren tep khong tin cay la ru ro thuc thi ma (bandit B614 da duoc `# nosec` tat co y).
- **RU RO TON TAI CO THAT (INFERENCE + VERIFIED one-line):** `repos/SAM2PATH/segment_anything/build_sam.py:105` goi `state_dict = torch.load(f)` **khong co `weights_only`**. Voi torch >= 2.6 mac dinh True -> `_pickle.UnpicklingError: Weights only load failed`. Lỗi này bị nuot boi `except Exception` cua adapter (xem 7.1/10.2) -> duoc xem nhu mot bug thuc te, khong chi la suy doan.

### 6.2 `strict=False` in "[OK] thanh cong" khi 567/680 tensor khong duoc nap

**[T6-strict-false-fake-success]** | *Critical* | 100% (VERIFIED tai lap) | `src/colab_pkg/model_loader.py:582,615-622` (va ban nhat tri `src/models/wsss_inference_methods.py:371`)

```python
res = model.load_state_dict(cleaned_state_dict, strict=strict)   # strict=False mac dinh
if not strict and hasattr(res, "missing_keys"):
    print(f"[OK] Nap trong so Stage 2 thanh cong "
          f"(Missing: {len(res.missing_keys)}, Unexpected: {len(res.unexpected_keys)})")
```

Tai lap (khong ghi file trong workspace, dung `tempfile`):

```text
--> Dang doc checkpoint Stage 2: fake_stage2.pth
--> Mac dinh so lop dau ra: 4 phan lop
[OK] Nap trong so Stage 2 thanh cong (Missing: 567, Unexpected: 1)
```

Mot checkpoint **khong co bat ky key nao truong kiem** (chi 1 key `not_a_real_layer.weight`) duoc bao cao la "nap thanh cong". 567 tensor **van la khoi tao ngau nhien**. He qua: moi ket qua danh gia (bang M1..M7, 69.34/70.49...) co the den tu mot mang **ngau nhien** ma log van in `[OK]`. Doi voi do an y khoa, day la nguyen nhan co the phat sinh "so lieu that tu model gia". Khuyen nghi Lead: `strict=True` mac dinh + **raise** neu `len(missing_keys) > 0`, va in ty le `% nap duoc`.

### 6.3 Tensor prompt cua SAM duoc chuyen device **DUNG**

```python
# src/models/sam_wsss_adapter.py:163, 252-261
device = cams.device
pts_tensor = torch.tensor(pts_list[:total_pts], dtype=torch.float32, device=device)
lbl_tensor = torch.tensor(lbl_list[:total_pts], dtype=torch.int64,  device=device)
results[c] = {"point_coords": torch.stack(batch_coords, 0), "point_labels": torch.stack(batch_labels, 0), ...}
```

`torch.stack` giu device cua thanh phan; dong thoi `:706` dung `.to(coords.device)`. **SACH** (khop voi commit `dc58ed0 fix(pipeline): ... enforce CUDA prompt device transfer`). Cung kieu `surrogate_cams = torch.zeros(b,4,h,w, device=device)` (`src/training/train_proposed_sam.py:257`) dung device.

### 6.4 Seed co dinh (INVARIANT-09)

- `src/colab_pkg/colab_runner.py:101-111`: `random/np/torch.manual_seed` + `cuda.manual_seed_all` + `cudnn.deterministic=True` + `cudnn.benchmark=False`. **DAT**.
- `src/pipeline/comparative_runner.py:41-45`, `src/training/train_stage1.py:372-376`, `train_stage2.py:250-254`: dat day du seed (rieng comparative_runner **khong** dat `cudnn.deterministic`).
- **THIEU (INFERENCE):** `src/training/train_proposed_sam.py` **khong co ham seed nao** trong `main()`; augmentation cua `BCSSStage2MultiMaskDataset` (`dataset_loader.py:343,361,368`) dung `torch.rand*()` phu thuoc seed toan cuc -> khong the tai lap neu chay trinh `train_proposed_sam.py` doc lap. **[T6-no-seed-sam-trainer]** *Medium*.
- **THIEU:** khong co bat ky dong goi **multi-seed (5 seeds) + $\mu \pm \sigma$** nao trong `src/` hay notebook; INVARIANT-09/`product/proposal/...:59` co cam ket "5 random seeds" nhung benchmark hien tai (`experiments/colab_run_*`) chi la **single run**. **[T6-no-multi-seed]** *High* (do tin cay 100% ve su thieu hut; INFERENCE ve viec no co duoc bao cao o noi khac).

### 6.5 AMP / batch size / VRAM cho T4 16GB

- `torch.amp.autocast(device_type=...)` duoc goi co kiem tra (`model_loader.py:917-925`, `wsss_inference_methods.py:838-846`, `train_stage1.py:154,201`, `train_proposed_sam.py:273-274,473`). Cac duong AMP trong inference **khong** can GradScaler (suy luan) va co dieu kien `use_amp and cuda` -> an toan CPU.
- `num_workers`: `dataset_loader.py:506-507,555-556,586-587` tu dong giam 4 -> 2 tren Windows (hợp lý); `train_stage1.py:415` `num_workers=0 if dry_run else 2`. **KHONG co gioi han thuat toan VRAM**: khong co try/except `torch.cuda.OutOfMemoryError` o bat ky dau, khong co cache clearing, khong co tu dong giam batch khi OOM. **[T6-no-oom-guard]** *Medium*. Voi WSI gigapixel (`src/pipeline/wsi_lowmag_highmag_pipeline.py`, 795 dong) va T4 16GB, day la ru ro sập pipeline giua chung.

---

## 7. T7 - Silent failure

### 7.1 Handler bien loi that thanh ket qua vo nghia

**[T7-wilcoxon-p1.0]** | *High* | 100% | `src/colab_pkg/evaluator.py:414-415`, `:421-422`, `:489-490`; `src/clinical/exhaustive_evaluator.py:574-575`, `:590-591`

```python
try:
    t_res = stats.ttest_1samp(diffs, 0.0)
except Exception:
    t_stat, t_pval = 0.0, 1.0
try:
    w_res = stats.wilcoxon(diffs, zero_method="zsplit", alternative="two-sided")
except Exception:
    w_stat, w_pval = 0.0, 1.0
...
out.update({"wilcoxon_pvalue": w_pval, "significant_p05": bool(w_pval < 0.05), ...})
```

Khi thong ke that su loi (thoat vi, NaN, scipy khong cai dat...), he thong **che tao** `p = 1.0` va `significant_p05 = False`, sau do ghi vao `COLAB_EVALUATION_REPORT.json`/`ablation_study_metrics.csv` (cot cuoi `4.489539e-17`, `3.092023e-02`...). Nguoi doc **khong the phan biet** "khong co y nghia thong ke" va "loi thong ke". Vi pham INVARIANT-07 (minh bach so lieu) va INVARIANT-10 (phan tich sai le).

**[T7-adapter-silent-surrogate]** | *Critical* | 100% (VERIFIED) | `src/models/sam_wsss_adapter.py:749-767`

```python
if checkpoint_path is not None and os.path.exists(checkpoint_path):
    try:
        from segment_anything import sam_model_registry
        sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        self.image_encoder, self.prompt_encoder = sam_model.image_encoder, sam_model.prompt_encoder
        self.is_surrogate = False
    except Exception:
        if not use_surrogate_if_missing: raise
        self.image_encoder = SurrogateImageEncoder(out_dim=256)
        self.prompt_encoder = SurrogatePromptEncoder(embed_dim=256)
        self.is_surrogate = True
else:
    self.image_encoder = SurrogateImageEncoder(out_dim=256)   # <-- cung lang le
    ...
```

Tai lap that:

```text
RESULT: is_surrogate = True | encoder = SurrogateImageEncoder
(khong co exception noi ra, khong co mot dong canh bao nao cho checkpoint Bi HONG)
SWALLOWED EXCEPTION TYPE: builtins.ModuleNotFoundError
No module named 'segment_anything'
```

va `grep -rn "is_surrogate" src tests scripts notebooks --include="*.py"` (ngoai tep dinh nghia) va grep trong `notebooks/03_Train_Proposed_MedSAM_T4.ipynb`: **0 consumer**. He qua nghiem trong nhat toan bo dot audit nay: "phuong phap de xuat dung MedSAM" co the thuc chat duoc huan luyen/danh gia tren **mot encoder CNN ngau nhien**, va khong mot tai lieu, log, hoac assert nao cho phep phat hien. Duong `else` (checkpoint missing) con lang le hon nua.

### 7.2 Cac handler khac

| Vi tri | Hanh vi | Danh gia |
| :--- | :--- | :--- |
| `src/clinical/survival_analysis.py:290-291, 681-682, 806-807` | `except Exception: pass` | Medium (loading manifest/ancillary) |
| `src/pipeline/wsi_lowmag_highmag_pipeline.py:151-152, 166-167, 410-411, 567-568` | `except Exception: pass` trong doc patch/WSI | High (mat patch cua benh nhan ma khong dem so) |
| `src/colab_pkg/colab_runner.py:33-34` | `reconfigure` UTF-8 fail -> pass | **chap nhan duoc** (chi log) |
| `src/colab_pkg/downloader.py:231-232` | weights_only retry | chap nhan (co ly do chinh dang) |
| `src/visualization/api_server.py` (3 handler), `src/utils/deliverable_auditor.py` (8 handler) | chu yeu bao ve tang doc file | Low |

**KHONG tim thay (bao cao dieu khong phat hien):** 0 `except:` tran; 0 `contextlib.suppress`; 0 `warnings.filterwarnings` trong `src/` va `scripts/`.

---

## 8. T8 - INVARIANT-12 + CI

### 8.1 Quet ky tu cam (VERIFIED)

```bash
grep -rIlP "[\x{2013}\x{2014}\x{2015}\x{2212}]" --include="*.py" --include="*.md" --include="*.ipynb" \
  --exclude-dir=repos --exclude-dir=.git --exclude-dir=graphify-out* .
```

- `src/`, `tests/`, `scripts/`, `experiments/`, `product/`, `docs/`, `notebooks/`, `drive_upload_ready/`, `AGENTS.md`, `README.md`, `requirements.txt`: **0 khop**.
- **NHUNG** `.agents/` (bi `.gitignore`): **10 tep markdown co em/en-dash/horizontal-bar**, vi du:

```text
.agents/rules/13_deliverable_quality_and_scientific_visuals_gate.md:32:
    ... Quet dau gach noi: Chan em-dash (U+2014) va en-dash (U+2013),
.agents/teamwork/checker_team_a/audit_report.md:44,46,49,51,54,56  (6 dong)
.agents/teamwork/victory_auditor_5/BRIEFING.md                     (11 dong)
```

- `docs/templates/.../chapters/main/chapter1.tex|chapter2.tex|chapter3.tex`: co dau (mau LaTeX cua UIT, ben ngoai `*.py|*.md|*.ipynb`).

**[T8-dash-claim-overstated]** | *Medium* | 100% | Ban cu `01_SWARM_RECHECK_VERIFICATION.md:128` ghi "**0 non-ASCII hyphens found across the entire workspace**" - sai pham vi: `.agents/*.md` (chinh la bo luat cua he thong) chua ky tu cam.

### 8.2 Blind spot cua `tests/test_governance_invariants.py`

**[T8-governance-test-blindspots]** | *Medium* | 100% | `tests/test_governance_invariants.py:4-12,17-31,36-43,55-62`

```python
def _scan_files_for_bad_dashes(patterns):
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):      # <-- glob KHONG buoc vao thu muc bat dau bang '.'
            if not f.startswith(IGNORED_DASH_PREFIXES): ...
violations = _scan_files_for_bad_dashes(["**/*.md"])  # .agents/**, .github/** bi bo qua HAN TOAN
```

Ba khuyet diem cu the:
1. Python `glob` **bo qua ten an (bat dau bang `.`)** -> `.agents/` (noi chua chinh rule 13 ve typography) never scanned. Day la loi "test tu choi minh".
2. Chi scan `*.md`, `*.py`, `*.ipynb`; **khong** scan `*.html`, `*.tex`, `*.csv`, `*.json`, `*.txt` -> slide Marp HTML (INVARIANT-12 ghi ro "slide bao cao") va `experiments/**/*.csv` nam ngoai kiem dinh.
3. `IGNORED_DASH_PREFIXES` dung `f.startswith("repos\\")` -> phu thuoc cua so lay lenh (`\` vs `/`); khi chay tu thu muc khac, che do loai tru mat hieu luc (khong an toan ve platform).
4. `test_no_raw_phone_photos_in_product_directory` (`:55-62`) chi kiem **tien to ten file** (`media_`, `img_`, `pxl_`, `screenshot`, `dcm_`) -> mot anh photo doi ten `fig1.png` van qua gate; INVARIANT-24/25 yeu cau kiem **noi dung anh** (deliverable_auditor co lam dieu nay, nhung test nay khong goi auditor).

**SACH (ghi nhan):** `BAD_DASH_CHARS` phuat them U+2012 (ngang hang), rong hon danh sach trong INVARIANT-12; `test_invariants_24_and_25_defined_in_agents_md` (`:46-52`) enforce su ton tai cua luat va duong dan `deliverable_auditor.py`.

### 8.3 CI/CD (`.github/workflows/ci.yml`) - gate chua bao gio chay

| Kiem tra | Bang chung | Ket luan |
| :--- | :--- | :--- |
| Workflow ton tai va active | `git ls-files .github` -> `.github/workflows/ci.yml`; `gh workflow list` -> `SE374 Medical AI CI Pipeline active 392939520` | Co |
| **Da tung chay?** | `gh api repos/medical-wsss-consortium/medical-wsss/actions/runs --jq '.total_count'` -> `0`; `gh run list --limit 5 --json ...` -> `[]` | **KHONG BAO GIO** |
| Trigger khop branch? | `on.push.branches: [main, dev]`, `on.pull_request.branches: [main, dev]` **nhung** `gh api ... --jq '.default_branch'` -> `master`, `git branch -vv` -> `* master` | **SAI** -> se **khong bao gio** duoc kich hoat ke ca khi push |
| Cai dependency cho pytest? | `pip install flake8 pytest Pillow` (dong 27). `requirements.txt:5-6` yeu cầu `torch>=2.2.0`, `torchvision>=0.17.0`. 19/22 tep trong `tests/` `import torch` o dong dau | **FAIL at collection** (`ModuleNotFoundError: torch`) **INFERENCE** (chua mo phong CI tren ubuntu) |
| Lint co that rang buoc? | `flake8 ... --count --exit-zero ... \|\| true` (dong 33-34) | **Gate vo nghia** (luon pass) |
| Zero-mockup grep co bat duoc gi? | `grep -rnw 'src/' -e 'TODO: mock' -e 'fakeData' -e 'dummyData' -e 'fake_metric' -e 'simulateTx'` -> chay local: **0 khop** trong khi `src/` co `SyntheticHistologyDataset`, `allow_mock`, `surrogate_cams`, `np.zeros((224,224))` mask fallback | **Gate no-op**: token khong ton tai trong ma nguon |
| Deliverable gate | `python -m src.utils.deliverable_auditor product/proposal/SE374_MINI_PROPOSAL_WSSS_WSI.md` chay local: `[DELIVERABLE QUALITY AUDIT: PASSED] ... DAT 100%!`, `EXIT=0` | **Hoat dong duoc** (duong dan + module co that) |
| `KMP_DUPLICATE_LIB_OK` trong CI | khong co | **KHONG can** (ubuntu, OpenMP conflict la dac thu Windows). De xuat them cho job Windows neu mo them matrix. |
| CI co chay duoc tren repo nay (pytest)? | Neu CI duoc kich hoat dung branch, no se goi `pytest tests/` -> kich hoat `build_variants()` va `build_drive_package()` (mutate repo tren runner) va **fail 3 test** o muc 1.3 | **FAIL + tac dong phu** |

**[T8-ci-never-runs]** | *Critical* | 100% (VERIFIED qua gh CLI) | `.github/workflows/ci.yml:5-9`.
He qua: INVARIANT-03 ("CI/CD tu dong lint, test va quoc anti-mockup") **hoan toan khong hoat dong**; moi "xanh" CI trong bao cao la khong co bang chung.

---

## 9. T9 - Dead code & duplication

### 9.1 `src/config/` (0 file `.py`)

```bash
ls -la src/config           # .gitkeep (44B) + bcss_patient_splits.json (7643B)
grep -rn "src\.config|from src import config" src tests scripts notebooks --include="*.py"   # -> 0 khop
```

Ket luan: **khong phai dead code**, no la tep JSON duoc doc **bang duong dan chuoi** boi 3 noi doc lap: `src/training/dataset_loader.py:123` (`project_root/src/config/...`), `src/datasets/patient_split.py:129` (`dirname(dirname(__file__))/config/...` - cung ket qua), `src/clinical/survival_analysis.py:279`. **Medium defect (duplicated path logic + khong co `src/config/__init__.py` nen khong the import/nhieu nguon doi chieu).** `[T9-config-path-triplication]`

### 9.2 Trung lap module nang nhat: 2 ban sao dong bo mot nua cua engine

```bash
grep -rn "^class DeepLabV3Plus|^class MultiMethodWSSSEngine|^class WSSSTissueStage1NetCAM|^class ASPP" src --include="*.py"
diff -u src/models/wsss_inference_methods.py src/colab_pkg/model_loader.py | grep -c "^[+-]"
```

- `src/models/wsss_inference_methods.py` (1798 dong) va `src/colab_pkg/model_loader.py` (1805 dong) cung dinh nghia `ASPP`, `_ASPPModule`, `Decoder`, `Bottleneck`, `DeepLabV3Plus`, `ResNet101`, `ResBlock`, `ResBlockBot`, `WSSSTissueStage1NetCAM`, `OtsuClinicalGating`, `safe_torch_load`, `build_and_load_deeplabv3plus`, `build_and_load_stage1_netcam`, `load_wsss_models`, `MultiMethodWSSSEngine`, `MultiMethodWSSSEngineV2` -> **16 class/ham trung ten**.
- **497 dong khac biet**. Drift thuc ve ma nguon (khong chi docstring):
  - `src/models/wsss_inference_methods.py:105` docstring noi "**7** methods (M1->M7)"; `src/colab_pkg/model_loader.py:11` noi "**5** (M1..M5)". Hai ban mo ta pham vi thuc nghiem khac nhau cho **cung ten lop**.
  - Trong `WSSSTissueStage1NetCAM.forward`: `y = self.fc8(x_pool).view(x.size(0), -1)` (ban `models`) vs `.view(conv6.size(0), -1)` (ban `colab_pkg`).
  - `safe_torch_load(checkpoint_path, device: torch.device)` (ban `models`, device bat buoc kieu Tensor) vs `safe_torch_load(checkpoint_path, device="cpu")` (ban `colab_pkg`, chap chuoi).
- Hau qua dinh tuyen: `src/pipeline/colab_runner.py` (153 dong shim) re-export tu **`src.colab_pkg`**; trong khi `src/pipeline/wsi_lowmag_highmag_pipeline.py:36`, `src/training/generate_pseudo_masks.py:29`, `train_stage1.py:26`, `train_stage2.py:30`, `src/clinical/exhaustive_evaluator.py:325,361` lai dung **`src.models.wsss_inference_methods`**. **Cung mot "phuong phap de xuat M1..M7" nhung ton tai 2 trien khai khac nhau 497 dong**, khong co chiuat nao khai bao ban nao la nguyen ban cho so lieu cong bo. Vi pham INVARIANT-05/06.

**[T9-duplicate-engine-drift]** | *High* | 100% (VERIFIED diff + grep) | nhu tren.

### 9.3 Notebook phien ban trung lap (ngoai 5.3)

`SE374_WSSS_Drive_Code/` chua **hai ban sao giong het nhau** cua cung 5 notebook (`<root>/` va `<root>/notebooks/`, md5 trung tung cap) + `SE374_WSSS_Drive_Code.zip` ben trong chinh no -> package tự chua zip cua no. Tong hop: **60 tep `.ipynb`** trong repo (ban cu noi 32), gom `notebooks/` (24), `notebooks/archive|history_runs|modular_drive` (7), `drive_upload_ready/**` (22), goc (10).

### 9.4 `scripts/`: 13 tep, phan lon la one-off scratch (vi pham INVARIANT-15)

```bash
ls scripts/*.py
# build_drive_package.py  build_t4_l4_notebook_suites.py  download_papers.py
# fix_bfloat16_and_dataset_count.py  fix_dataset_path_resolution.py  fix_forward_cam_inversion.py
# fix_medsam_prompt_call.py  generate_colab_variants.py  generate_thin_colab_notebooks.py
# inspect_v2_notebook.py  refactor_notebooks_pure_colab.py  update_agents_invariants.py
```

6/12 tep co tien to `fix_`/`inspect_`/`refactor_` la **scratch script dung mot lan** (VD `fix_bfloat16_and_dataset_count.py` thura 8 loi `E741`), vi pham truc tiep INVARIANT-15 ("TUYET DOI NGHIEM CAM tao cac file Python dung mot lan... lam rac thu muc du an"). `inspect_v2_notebook.py` va `update_agents_invariants.py` **khong duoc import boi bat ky test nao** (`grep -rn "import scripts" tests/` chi thay `generate_colab_variants`, `build_t4_l4_notebook_suites`, `build_drive_package`).

**[T9-once-off-scripts]** | *Medium* | 100%.

### 9.5 Module khong duoc import (dead code)

| Module | Nguoi dung | Danh gia |
| :--- | :--- | :--- |
| `src/visualization/draw_all_proposal_figures.py` (451 dong), `draw_pipeline_diagram.py` (237) | tim kiem `import` trong `src/tests/scripts/notebooks` -> 0 | Sinh hinh cho proposal; **co the la dead sau khi chay mot lan** (INFERENCE - can xac nhan qua git log). `[T9-fig-scripts-dead]` *Low* |
| `src/clinical/exhaustive_evaluator.py` (1091 dong) | `src/pipeline/*`, `tests/` | Song |
| `src/datasets/bcss_dataset.py` (`BCSSDataset`) | `src/training/train_proposed_sam.py:487` va `tests/` | Song, nhung **khong duoc pipeline chinh dung** (pipeline chinh dung `dataset_loader.py`) -> 2 tang dataset song song **[T9-dual-dataset-layer]** *Medium* |
| `src/losses_metrics/metrics.py` (525) / `losses.py` | `src/training/*` | Song |
| `src/utils/deliverable_auditor.py` | `ci.yml`, `tests/` | Song (duoc CI dung) |
| `src/colab_pkg/colab_runner.py` (728) + `src/pipeline/colab_runner.py` (shim 153) + `src/pipeline/comparative_runner.py` | 3 lop dieu phoi cho cung mot vai tro | Vi pham INVARIANT-02 phan "ranh gioi ro rang" **[T9-triple-orchestrator]** *Medium* |
| `scripts/__pycache__/*.pyc` (4 tep), `src/**/__pycache__` | - | Rac build, co trong `.gitignore` (SACH ve git) |

### 9.6 Import thua (VERIFIED qua ruff)

`time`, `urllib.request` (`scripts/download_papers.py:4-5`), `re`, `pathlib.Path` (`scripts/fix_dataset_path_resolution.py:2-3`, `scripts/refactor_notebooks_pure_colab.py:4`), `os` (`scripts/update_agents_invariants.py:1`); `Path` (`refactor_notebooks_pure_colab.py:4`); ham chet `old_drive_block` (`:42`).

---

## 10. T10 - Chat luong test + doi chieu nguon ben ngoai

### 10.1 Danh gia theo testing-best-practices

**Test HANH VI vs test chi tiet cai dat:**
- **Tot (hanh vi that):** `tests/test_benchmark_components.py` (7 test: streaming accumulator, confusion-matrix `bincount`, per-class mIoU/Dice), `tests/test_deeplabv3plus.py` (11 test: ASPP dilation `[1,6,12,18]`, CGM penalty `-1e4`, hinh dang decoder), `tests/test_pda_backbone.py` (14), `tests/test_adversarial_clinical_stress.py` (13: Otsu duoi dich chieu/sao che, GroupNorm vs BatchNorm voi batch=1) - day la duong bien co y nghia lam sang.
- **Kem (test cai dat noi bo / ty tham chieu):** nhom `tests/test_colab_notebook.py` (30) va `tests/test_colab_pro_notebooks.py` (71) - **101/348 test (29%)** kiem tra notebook bang cach **doc chuoi JSON/AST** ("cell 12 phai chua `class DeepLabV3Plus`", "khong duoc con HSV naive", `test_integration_engine_and_colab_generator_parity`). Nhung assert nay **khong do hanh vi nao**; chung chi buoc code trung voi mot doan text, va la nguyen nhan truc tiep cua defect **[T1-tests-mutate-repo]**. Chi so bao phu hanh vi that cua bo test do: **~71%**.
- **Mock/monkeypatch:** `tests/test_training_service.py` 23 khop mock/patch, `tests/test_wsi_pipeline_extension.py` 9, `tests/test_scientific_upgrades.py` 8, `tests/test_colab_notebook.py` 6. Muc do vaa pham (INFERENCE: can doc tung test de ket luan "mock chinh he thong dang test"); tuy nhien `test_training_service` duoc dat ten "dry_run" va co that goi `main()` cua `train_stage1/stage2` -> **co chay duong that**.
- **Phu thuoc thu tu / trang thai:** **VERIFIED** (muc 1.3): 2 test fail khi chay full nhung pass khi chay sau do. Nguyen nhan: test truoc **ghi de** notebook/manifest ma test sau doc.
- **Khong xac dinh (nondeterminism):** `grep -rn "time.sleep" tests/*.py` -> **0 khop** (SACH). Nhung 10/22 tep dung `np.random.*` / `torch.rand*`; **8 tep khong dat seed truoc khi dung** (`test_audit_team_2.py:286`, `test_colab_notebook.py:162,210,832,839,976`, `test_benchmark_components.py:127`, `test_pda_backbone.py:268`, `test_scientific_upgrades.py:166,308,331-340`): chi 2 tep (`test_adversarial_clinical_stress.py:79,129`, `test_audit_team_1.py:302` qua seed 2026, `test_audit_team_4.py:290`) co `np.random.seed(...)`. Cac test con lai **pass may man** vi chi kiem bat bat doxuong/ngưỡng, nhung do la **rui ro flakiness** va vi pham INVARIANT-09 ve tai lap. **[T10-unseeded-random-tests]** *Medium*.
- **Duong bien / loi:** **CO** va tot: `test_load_patient_splits_invalid_path` (FileNotFoundError), `test_verify_patient_leakage_catches_overlap` (ValueError), `test_adversarial_clinical_stress` (PDA/batch=1), `test_audit_team_4.py` (tensor vuong/khong vuong, phuong sai 0), `test_patient_split.py` 2 test Hypothesis (fuzz regex + `st.text()` arbitrary UTF-8). **THIEU**: khong test `patient_ids = None/set rong` (3.2), khong test `mask_dir` khong ton tai (2.3), khong test dataset rong -> DataLoader (4.1), khong test `strict=False` voi checkpoint sai (6.2), khong test `except Exception -> p=1.0` (7.1). **4/5 defect nang nhat cua bao cao nay deu nam o vung "khong co test"**.

### 10.2 Doi chieu chu ky loi voi nguon ben ngoai (accepted answer + ban chuan)

#### Chu ky A: `ValueError: num_samples should be a positive integer value, but got num_samples=0`

- **Loai loi:** `ValueError` tir `torch.utils.data.dataloader` (RandomSampler).
- **Ma repo mac phai (duoc tai lap):** `BCSSDataset` tra `len == 0` khi layout long 1 cap (`src/datasets/bcss_dataset.py:55,72`), `exhaustive_evaluator.discover_test_patches` tra `[]` (`:394`), `find_image_files` tra `[]` khi root khong ton tai (`dataset_loader.py:156-158`) - **khong noi nao raise som**.
- **Y kien duoc chap nhan (accepted):** StackOverflow #71615089: *"The problem is that the dataset is empty. The datapath may be wrong or preprocessing might be causing problems ending up with no object in [the dataset]"*; PyTorch Forum #56041: *"Make sure the `__len__` method of your Dataset returns a valid number of samples (>0)"*.
  - https://stackoverflow.com/questions/71615089/valueerror-num-samples-should-be-a-positive-integer-value-but-got-num-samples
  - https://discuss.pytorch.org/t/num-samples-should-be-a-positive-integer-value-but-got-num-samples-0/56041
- **Ban chuan hoa cho repo (khong dung placeholder gia, thoa man dong thoi INVARIANT-07):**

```python
# dau cuoi __init__ cua moi Dataset trong src/
if not self.image_paths:                      # hoac self.files
    raise RuntimeError(
        f"[INVARIANT-29] Dataset rong: split={self.split!r} root={self.root_dir!r}. "
        f"Da thu {candidates}. Kiem tra cau truc thu mục (train/img vs train/*.png)."
    )

# va mot ham dq tim dđa tầng duoc dung CHUNG cho moi loader (thay cho glob nong)
def resolve_split_path(root: Path, split: str) -> Path:
    for pat in (f"{split}/img", f"{split}/images", f"{split}", "training/img"):
        if (root / pat).is_dir():
            return root / pat
    for cand in sorted(root.glob(f"**/{split}/img")):      # de quy da tang
        return cand
    raise FileNotFoundError(f"Khong tim thay split {split!r} duoi {root}")
```

Ly do chon `raise` thay vi placeholder: placeholder vi pham INVARIANT-07; INVARIANT-29 yeu cau "khong crash im lang", khong yeu cau "khong crash". Neu bat buoc giu pipeline song cho demo, **tam** moi artifact metric (`aborted=True` + cam `export_evaluation_artifacts`).

#### Chu ky B: `_pickle.UnpicklingError: Weights only load failed` (torch >= 2.6)

- **Loai loi:** `ModuleNotFoundError`/`_pickle.UnpicklingError` bi **gom** thanh mot `except Exception` vo danh.
- **Ma repo mac phai:** `repos/SAM2PATH/segment_anything/build_sam.py:105` `state_dict = torch.load(f)` (khong truyen `weights_only`), duoc goi tu `src/models/sam_wsss_adapter.py:752`; toan bo loi bi `except Exception:` (`:755`) nuot va chuyen sang `SurrogateImageEncoder` (`:759-762`) **khong log**.
- **Phat kien truyen thong:** tu PyTorch 2.6, mac dinh `weights_only=True` -> checkpoint chua object pickled (config Namespace, numpy scalar) **that bai** (issues #118092, #138985, #133163, #150998, #179904, #149313).
  - https://github.com/pytorch/pytorch/issues/149313
  - https://github.com/pytorch/pytorch/issues/179904
  - https://github.com/pytorch/pytorch/issues/118092
  - https://github.com/pytorch/pytorch/issues/138985
- **Ban xu ly duoc duy tri/chap nhan:** (i) goi roi rang `torch.load(f, map_location="cpu", weights_only=True)`, (ii) neu that bai, **dang ky whitelist** bang `torch.serialization.add_safe_globals([...])` thay vi tat hoan toan, (iii) neu bat buoc `weights_only=False` thi **chi** cho tep cua chinh minh, va **luon in loi goc**.
- **Ban cho `sam_wsss_adapter.py` (thoa man ca T6 va T7):**

```python
if checkpoint_path is not None and os.path.exists(checkpoint_path):
    try:
        from segment_anything import sam_model_registry
        sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    except Exception as exc:                                   # KHONG duoc im lang
        if not use_surrogate_if_missing:
            raise
        logger.warning(
            "[MedSAM] Khong nap duoc trong so chinh thuc %r (%s: %s). "
            "Chuyen sang SurrogateImageEncoder NGAU NHIEN. "
            "MOI ket qua tu day KHONG duoc dung de bao cao la 'MedSAM de xuat'.",
            checkpoint_path, type(exc).__name__, exc,
        )
        ...
        self.is_surrogate = True
        self.surrogate_reason = f"{type(exc).__name__}: {exc}"
```

va: **`train_proposed_sam.py` + `evaluator` + `export_evaluation_artifacts` phai raise neu `model.is_surrogate`** khi nguoi dung dat co che "proposed = MedSAM"; them `assert not model.is_surrogate` trong `tests/test_sam_wsss_adapter.py`.

#### Chu ky C: `RuntimeError: Expected all tensors to be on the same device` (prompt/SAM)

- **Kiểm tĩnh:** day la chu ky ma repo **da tung mac phai** va **da fix** (commit `dc58ed0 fix(pipeline): eliminate dummy fallbacks and enforce CUDA prompt device transfer`). Hien tai `src/models/sam_wsss_adapter.py:252-261` tao prompt tensor **voi `device=cams.device`** nen duong SACH.
- **Rui ro con lai (INFERENCE, Medium):** `src/training/train_proposed_sam.py:269-270` lay `coords/labels` tu `prompt_dict[primary_class]` voi `primary_class = 0` **cung dich** (khong phai `argmax(weak_labels)`) va **khong kiem tra** `active_mask` - neu ca 4 lop deu bi gate tat (`active_mask` False toan bo), `coords` la padding `(0.0, 0.0)/-1` -> mask dau ra co the rong, va Dice = 0 bi ghi nhu "ket qua cua phuong phap". Doi chieu: https://github.com/facebookresearch/segment-anything/issues/597 ("Tensors must have same number of dimensions: got 2 and 3" - lien quan hinh dang `point_coords` khi khong batch hoa).

### 10.3 Cac hang muc SACH cua T10

- `grep -rn "time.sleep" tests/*.py` -> 0 khop (khong co sleep-based nondeterminism).
- `tests/test_patient_split.py` dung **Hypothesis** (`@given(st.from_regex(...))`, `st.text()`) - day la property-based test that, khong mock.
- `tests/test_colab_notebook.py` co **AST compile** tung cell cua notebook (kiem cu phap that), khong chi grep chuoi.
- `tests/test_api_server.py` (8) kiem endpoint FastAPI + base64 + DZI manifest qua `TestClient` -> hanh vi tang API that.

---

## 11. Ma tran nghiem thu tong hop (Agent 01)

| # | Chieu | Tuy bo ban 2026-11-02 | Bang chung cua Agent 01 | Ket luan |
| :--- | :--- | :--- | :--- | :--- |
| T1 | pytest 348/0 | `5 failed, 343 passed` (455.12s) | `ruff check scripts` -> 33 errors | **FAIL** |
| T2 | Zero-mockup | `SyntheticHistologyDataset` fallback; `p=1.0` mask loi; 567 missing weights van in `[OK]`; 2 paper trung 5 chi so, khong doi chieu duoc voi `docs/papers/*.md` | Co log that cho M1..M7 | **FAIL (co dieu kien)** |
| T3 | 0.0% leakage | Manifest = dia that, 0 overlap (106/22/23) | Nhung filter bi tat khi manifest rong; 2 `extract_patient_id` phan ky | **DAT (nhan) / FAIL (cau bao ve)** |
| T4 | Anti-shallow-glob | `bcss_dataset.py` glob nong (VERIFIED crash); `exhaustive_evaluator` glob nong; **0 safety-net trong src** | `dataset_loader.rglob` va `downloader` da dat | **FAIL** |
| T5 | INVARIANT-28/27 | 23 notebook co `drive.mount` (ke ca goi ship); 6 notebook drift goc/notebooks; BonusDemo 3 ban khac nhau; 4 ThinDriveRunner trong `drive_upload_ready` khac ban trong `notebooks` | vong doi 7 buoc thieu o 4 nhom | **FAIL** |
| T6 | Device/tai nguyen | `strict=False` fake-success (VERIFIED); SAM prompt dung device (SACH); weights_only retry dat | Thieu seed cho `train_proposed_sam`, thieu multi-seed 5 seeds, thieu OOM guard | **FAIL mot phan** |
| T7 | Silent failure | `except Exception` x56; 5 cho tra `p=1.0`; adapter surrogate im lang (VERIFIED) | 0 `except:` tran, 0 `contextlib.suppress` | **FAIL** |
| T8 | INV-12 + CI | 0 khop trong `src..docs`; **10 tep `.agents/*.md` co em/en-dash**; `gh api -> runs total_count = 0`; trigger `[main,dev]` vs branch `master` | deliverable_auditor chay duoc (exit 0) | **FAIL** |
| T9 | Dead/dup | 16 class trung ten gitua 2 file (497 dong khac); `src/config/` 3 noi doc duong dan khac nhau; 6 one-off `fix_*.py`; 60 `.ipynb` (ban cu noi 32) | | **FAIL** |
| T10 | Chat luong test | 29% test la text-assert notebook; 8 tep random khong seed; 2 test pass-fail tuy thu tu; thieu test duong loi cho 4/5 defect nang | Hypothesis + AST + FastAPI TestClient dat | **DAT CO DIEU KIEN** |

**Dem theo severity:** Critical **5** (T1-pytest-red, T1-tests-mutate-repo, T7-adapter-silent-surrogate, T6-strict-false-fake-success, T8-ci-never-runs) - High **8** (T2-synthetic-fallback-train, T2-metric-silent-drop, T2-citation-provenance, T3-filter-disabled-when-empty, T3-hardcoded-fallback, T4-shallow-glob, T4-no-safety-net, T5-drive-mount) - Medium **9** va Low **4** (xem chi tiet trong tung muc).

**Phan loai theo do tin cay:** 21/26 defect co **bang chung VERIFIED** (kem lenh + output nguyen van); 5 defect con lai la **INFERENCE** (T2-surrogate-cams, T3-filter-disabled [can dung manifest rong], T5-lifecycle-gaps [mot so o dang tong hop], T6-no-multi-seed [chua kiem tra het product/reports], T10-mock-do [chua doc het tung test]).

**Cam ket Zero-Decomposition:** Toan bo 10 chieu duoc Agent 01 doc lap thuc hien tren khap repo (src, tests, scripts, notebooks, drive_upload_ready, product, docs, experiments, data, .github), khong phan ra cho agent khac.

**Tuyet trinh Lead:** (1) giao Fixer Swarm thutu theo Critical -> High; (2) bat buoc **tach board test repo-mutating** sang `tmp_path` + tao `conftest.py` fixture `isolate_workspace`; (3) CI can sua `branches: [master]` va `pip install -r requirements.txt` (neu khong, moi "gate" van la chut khong ton tai); (4) sinh lai toan bo `experiments/**` sau khi fix 2.3/6.2/7.1, vi cac gia tri do co the da duoc sinh ra tu checkpoint khong nap duoc trong khi **log van bao "thanh cong"**.

---

## 12. Cac lenh tai lap (phuc vu Peer-Swap)

```bash
# T1
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q -p no:cacheprovider
KMP_DUPLICATE_LIB_OK=TRUE python -m ruff check src tests scripts
KMP_DUPLICATE_LIB_OK=TRUE python -m ruff check scripts --output-format concise
# T3 (doi chieu manifest voi dia that - khong ghi file)
KMP_DUPLICATE_LIB_OK=TRUE python -c "import glob,os,json,collections;from src.training.dataset_loader import extract_patient_id as e; ..."
# T4 (tai lap crash INVARIANT-29, dung tempfile ngoai workspace)
KMP_DUPLICATE_LIB_OK=TRUE python -c "<tem du lieu nested train/img -> BCSSDataset -> DataLoader>"
# T5/T9
md5sum *.ipynb notebooks/*.ipynb drive_upload_ready/SE374_WSSS_Drive_Code/*.ipynb notebooks/modular_drive/*.ipynb
grep -rl "drive.mount" --include="*.ipynb" . | grep -v repos/
diff -u src/models/wsss_inference_methods.py src/colab_pkg/model_loader.py | grep -c "^[+-]"
# T6 (tai lap fake checkpoint)
KMP_DUPLICATE_LIB_OK=TRUE python -c "import tempfile,os,torch;from src.colab_pkg.model_loader import build_and_load_deeplabv3plus as b; ..."
# T7 (tai lap silent surrogate)
KMP_DUPLICATE_LIB_OK=TRUE python -c "from src.models.sam_wsss_adapter import MedSAM_WSSS_Adapter as M; ..."
# T8
grep -rIlP "[\x{2013}\x{2014}\x{2015}\x{2212}]" --include="*.py" --include="*.md" --include="*.ipynb" --exclude-dir=repos .
gh api repos/medical-wsss-consortium/medical-wsss/actions/runs --jq '.total_count'
```

*Nguon ngoai trich trong T10:* [SO 71615089](https://stackoverflow.com/questions/71615089/valueerror-num-samples-should-be-a-positive-integer-value-but-got-num-samples), [PyTorch Forum 56041](https://discuss.pytorch.org/t/num-samples-should-be-a-positive-integer-value-but-got-num-samples-0/56041), [pytorch#149313](https://github.com/pytorch/pytorch/issues/149313), [pytorch#179904](https://github.com/pytorch/pytorch/issues/179904), [pytorch#118092](https://github.com/pytorch/pytorch/issues/118092), [pytorch#138985](https://github.com/pytorch/pytorch/issues/138985), [segment-anything#597](https://github.com/facebookresearch/segment-anything/issues/597).

---
*Bao cao Agent 01 - ton trong INVARIANT-12 (chi dung ASCII hyphen 0x2D), INVARIANT-32 (xuat ra tep, khong do van ban dai vao chat).*
