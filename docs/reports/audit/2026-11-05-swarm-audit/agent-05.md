# AUDIT 05 - Báo Cáo Kiểm Định Độc Lập (Swarm 8 Tác Tử)

**Đồ án:** SE374.R11 - Weakly-Supervised Semantic Segmentation (BCSS-WSSS + TCGA BRCA WSI)
**Tác tử:** AUDITOR 05 (độc lập, prompt đồng nhất, thực thi trọn vẹn T1-T10 trên toàn repo)
**Ngày audit:** 2026-11-05
**Repo:** `SE374_WSSS_WORKSPACE`
**Môi trường:** Windows + Git Bash, Python 3.13.9, pytest 8.4.2, ruff 0.12.0, torch 2.10.0+CPU, **KHÔNG GPU**
**Chế độ:** CHỈ đọc + chạy lệnh read-only. Không sửa source, không sửa test, không tạo script mới.
**Khuôn kết luận:** Mọi mục ghi rõ `VERIFIED` (có lệnh + output thật) hoặc `INFERENCE` (suy đoán từ đọc mã).

---

## 0. Tổng Quan Chỉ Số

| Hạng mục | Kết luận |
| :--- | :--- |
| pytest sạch (một lượt, không cạnh tranh) | **348 passed / 0 failed / 2 warnings, 268.16s** - Tuyên bố tiền audit **ĐÚNG** |
| pytest khi có 2 lượt chạy chồng nhau | **2 failed** (test_2+test_3, rồi test_2+test_7 ở lượt tách rời) - **KHÔNG tất định** |
| `ruff check src` | 0 errors - **ĐÚNG** |
| `ruff check src tests scripts` | **33 errors** (toàn bộ trong `scripts/`) |
| `ruff check . --exclude repos` | **464 errors** |
| Dấu gạch bị cấm U+2013/2014/2015/2212 | **0** trong file git-tracked `.py/.md/.ipynb`; **230 dòng** trong thư mục gitignored |
| Patient leakage BCSS trên đĩa | **0.0%** - manifest khớp đĩa 100% - **ĐÃ KIỂM CHỨNG THẬT** |
| Số liệu metric trong `product/` | Có log đối chứng trong `experiments/` - **ĐẠT** |
| **Tổng defect phát hiện** | **40** (Critical 6, High 15, Medium 16, Low 3) |

**Nhận định tổng:** Tiền audit **không bịa số liệu headline** (348/0, ruff src 0, leakage 0.0%, hyphen 0 trong phạm vi git-tracked đều tái lập được). Tuy nhiên tiền audit **phạm lỗi scope**: mỗi chỉ số "0" đều chỉ đúng với phạm vi hẹp nhất đã chọn (`src/` thay vì cả repo; git-tracked `.py/.md/.ipynb` thay vì "toàn bộ workspace"), và **hoàn toàn bỏ sót 3 defect Critical ở tầng hành vi runtime** mà không một test nào hiện có chạm tới: dữ liệu giả thay dự đoán thật (`SyntheticHistologyDataset`), encoder MedSAM giả không bị báo cáo, và pseudo-mask/GT-mask toàn zeros được âm thầm sinh ra.

---

## 1. T1 - Tầng 1: Bộ Test Tự Động

### [T1-01] Tuyên bố 348/0 TÁI LẬP ĐƯỢC - nhưng chỉ khi chạy một mình | Severity: Low (info) | VERIFIED

Lệnh (một lượt, không có tiến trình python khác):

```
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q -p no:cacheprovider
```

Output thật (bản đầy đủ, đã kết thúc):

```
........................................................................ [ 20%]
........................................................................ [ 41%]
........................................................................ [ 62%]
........................................................................ [ 82%]
............................................................             [100%]
348 passed, 2 warnings in 268.16s (0:04:28)
EXIT=0
```

Đối chiếu tuyên bố tiền audit ("348 passed, 0 failed, 2 warnings, 104.15s"): **số test khớp tuyệt đối (348/0/2)**. Thời gian lệch lớn (268s vs 104s, và 522s ở lượt có cạnh tranh) - tiền audit ghi 104.15s là không tái lập được trên máy này, nhưng đây là khác biệt phần cứng nên không kết luận là sai.

2 cảnh báo đúng như tiền audit mô tả, trích nguyên văn:

```
tests/test_clinical_service.py::test_exhaustive_benchmark_evaluator_dry_run_execution
  <python_env>\Lib\site-packages\scipy\stats\_axis_nan_policy.py:423:
  RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation.
  This occurs when the data are nearly identical. Results may be unreliable.

tests/test_colab_notebook.py::test_stage1_netcam_architecture_and_forward_cam
  <workspace_root>\repos\WSSS-Tissue\tool\infer_utils.py:6:
  DeprecationWarning: scipy.misc is deprecated and will be removed in 2.0.0
```

### [T1-02] `tests/test_thin_drive_runner.py` KHÔNG hermetic, phá hủy và viết lại repo state | Severity: High | VERIFIED

Ba lượt chạy cho ra **ba tập lỗi khác nhau** trên cùng một commit - chứng cứ trực tiếp về thiếu cách ly test.

Lệnh 1 - chạy toàn suites trong khi một lượt pytest khác còn sống:

```
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q -p no:cacheprovider
```
```
E       FileNotFoundError: [Errno 2] No such file or directory:
'<workspace_root>\drive_upload_ready\SE374_WSSS_Drive_Code\notebooks\history_runs\Run3_20261020_2055UTC_MultiScale_Pyramid_D4_M1_to_M7_T4_Full4986.ipynb'
...
E       FileNotFoundError: [WinError 3] The system cannot find the path specified:
'<workspace_root>\drive_upload_ready\SE374_WSSS_Drive_Code\src\__init__.py'
FAILED tests/test_thin_drive_runner.py::test_2_drive_bundle_packager_creates_complete_folder_and_zip
FAILED tests/test_thin_drive_runner.py::test_3_new_thin_runner_notebooks_valid_concise_and_modular
2 failed, 346 passed, 2 warnings in 522.22s (0:08:42)
```

Lệnh 2 - chạy riêng lẻ file đó:

```
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_thin_drive_runner.py -q -p no:cacheprovider
```
```
E   PermissionError: [WinError 32] The process cannot access the file because it is being used
by another process:
'<workspace_root>\drive_upload_ready\SE374_WSSS_Drive_Code\src\models\backbone.py'
FAILED tests/test_thin_drive_runner.py::test_2_drive_bundle_packager_creates_complete_folder_and_zip
FAILED tests/test_thin_drive_runner.py::test_7_invariant_12_strict_ascii_hyphens_across_all_packaged_files
2 failed, 8 passed in 63.55s
```

Lệnh 3 - chạy lại toàn suites một mình: **348 passed / 0 failed** (xem [T1-01]).

Nguyên nhân gốc, `tests/test_thin_drive_runner.py:180` và `test_7`:

```python
packager.build_drive_package()
```

và `scripts/build_drive_package.py:1051`:

```python
if dest_src_dir.exists():
    shutil.rmtree(dest_src_dir)
```

**Bằng chứng phụ (rất nặng):** số file notebook trong repo THAY ĐỔI do tác dụng phụ của test. Trước khi chạy test: `find . -name "*.ipynb" -not -path "./repos/*" | wc -l` = **47**. Ngay trong lúc test chạy: **59**. Một bộ test làm biến đổi inventory của chính đối tượng nó audit thì không thể dùng làm cổng nghiệm thu.

**Hệ quả:** Tiền audit nghiệm thu "348/0" dựa trên một suites mà độ xanh/đỏ phụ thuộc thứ tự và trạng thái file hệ thống. Chốt chặn Anti-Mock-Testing của [INVARIANT-26] yêu cầu kiểm chứng hành vi nạp dữ liệu thật; ở đây hành vi thật là "test xóa và rebuild thư mục repo đang được test khác đọc".

### [T1-03] "ruff 0 errors" chỉ đúng khi scope hẹp nhất | Severity: Medium | VERIFIED

```
python -m ruff check src              -> All checks passed!
python -m ruff check tests            -> All checks passed!
python -m ruff check src tests scripts -> Found 33 errors.  (13 fixable)
python -m ruff check . --statistics --exclude repos -> Found 464 errors. (385 fixable)
```

Phân bố 464 lỗi toàn repo:

```
193  I001  unsorted-imports
 93  F811  redefined-while-unused
 53  E701  multiple-statements-on-one-line-colon
 36  F541  f-string-missing-placeholders
 35  F401  unused-import
 25  E401  multiple-imports-on-one-line
 19  E741  ambiguous-variable-name
  6  F821  undefined-name
  4  F841  unused-variable
```

Điển hình trong `scripts/`: `scripts/refactor_notebooks_pure_colab.py:42 F841`, `scripts/update_agents_invariants.py:1 F401`. Tiền audit ghi "0 lint errors across all 42 modules in `src/`" - xác nhận `find src -name "*.py" | wc -l` = **42**, nên phát biểu đó **chính xác nhưng bị đóng khung**; nó không hề đề cập 33 lỗi `scripts/` và 464 lỗi toàn repo.

### [T1-04] `ruff.toml` tự tắt rule ngay trong test | Severity: Medium | VERIFIED

```toml
[lint]
select = ["E", "F", "W", "I"]
ignore = ["E402", "E501", "W291", "W293"]

[lint.per-file-ignores]
"tests/*" = ["F401", "F841"]
```

`F401`/`F841` được miễn trừ **đúng trên tập test** - tức import thừa và biến chết trong 25 file test không bao giờ bị phát hiện. Bộ rule cũng không gồm `B` (bugbear) hay `SIM`, nên các lỗi logic kinh điển (loop variable捕获, `except` quá rộng) nằm ngoài phạm vi kiểm.

### [T1-05] 6 lỗi F821 `DRY_RUN` là dương tính giả nhưng tố cáo kiểu code phá static analysis | Severity: Low | VERIFIED

```
notebooks\03_Train_Proposed_MedSAM.ipynb:cell 15:14:52: F821 Undefined name `DRY_RUN`
   | DRY_RUN = False if 'DRY_RUN' not in globals() else DRY_RUN
   |                                                    ^^^^^^^ F821
```
(tương tự cho `03_..._L4`, `03_..._T4`, `04_Exhaustive_Benchmark_Survival{,.ipynb _L4 _T4}` - tổng 6)

`INFERENCE`: ternary lười nên nhánh `else` chỉ chạy khi tên đã tồn tại -> không crash runtime. Nhưng mẫu code "tự tham chiếu tên mình qua `globals()`" lặp lại khắp notebook khiến ruff/IDE/mọi SAST đều mù ở đúng chỗ này.

---

## 2. T2 - Zero-Mockup ([INVARIANT-07])

### [T2-01] CRITICAL - Huấn luyện trên nhiễu random khi thiếu dữ liệu, và số liệu vẫn được in + lưu checkpoint | Severity: Critical | tin cậy RẤT CAO | VERIFIED (đọc mã) + INFERENCE (đường kích hoạt)

`src/training/train_proposed_sam.py:485-494`:

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

`src/training/train_proposed_sam.py:195-217` - bản thân dataset giả:

```python
class SyntheticHistologyDataset(Dataset):
    """Deterministic synthetic histology dataset for dry-run and local smoke testing."""
    ...
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(42 + idx)
        img = torch.rand(3, self.image_size, self.image_size)
        mask = torch.randint(0, 5, (self.image_size, self.image_size), dtype=torch.long)
        mask[:4, :] = 255
        mask[-4:, :] = 255
        weak_label = (torch.rand(4) > 0.4).float()
        return img, mask, weak_label
```

Ba chi tiết làm đây là lỗi nghiêm trọng chứ không phải "tiện ích dry-run":

1. Nhánh `else` phục vụ **hai điều kiện hoàn toàn khác nhau**. Khi `--dry-run` **không** được bật mà `args.data_dir` chỉ đơn giản **không tồn tại**, code rơi vào nhánh synthetic với `num_samples = 32` **và** `total_epochs = args.epochs` (dòng 499, mặc định `--epochs 25`), `max_steps = None` (dòng 500). Tức là một **cuộc huấn luyện đầy đủ 25 epoch trên mặt nạ noise ngẫu nhiên**.
2. Kết quả vẫn được in như số liệu thật, `src/training/train_proposed_sam.py:505-510`:
```python
        print(
            f"Epoch {epoch:02d} [{elapsed:.1f}s]: "
            f"Train Loss={train_stats['loss']:.4f} ... "
            f"Val mIoU_fg={val_stats['mIoU_fg']:.4f} (TUM={val_stats['iou_TUM']:.4f}, ...)"
        )
```
3. Checkpoint ghi thẳng `best_mIoU` và `val_stats` **không có trường nào đánh dấu nguồn dữ liệu**, dòng 540-549:
```python
        save_checkpoint(
            state={
                "epoch": epoch,
                ...
                "best_mIoU": best_miou,
                "val_stats": val_stats,
            },
```

Đường kích hoạt là hiện thực chứ không hypothetical: `--data-dir` mặc định là `data/BCSS-WSSS` (dòng 423) và `.gitignore` có dòng `data/BCSS-WSSS/` (xác nhận bằng `grep -n "^data" .gitignore`). Một clone mới trên Colab **mặc định không có** `data/BCSS-WSSS` -> `os.path.exists` False -> toàn bộ "Proposed MedSAM" chạy trên noise.

**Hệ quả học thuật:** một con số mIoU sinh ra từ mask `torch.randint` không phải kết quả nghiên cứu mà là bằng chứng giả mạo, đúng điều [INVARIANT-07] định tội "0 điểm và đình chỉ đồ án".

### [T2-02] CRITICAL - Encoder MedSAM giả (surrogate chưa huấn luyện) được thay thế âm thầm và lá cờ `is_surrogate` là state chết | Severity: Critical | tin cậy RẤT CAO | VERIFIED

`src/models/sam_wsss_adapter.py:744-768`:

```python
        # Dual-mode initialization: Official SAM if available and file exists, else Surrogate
        self.is_surrogate = True
        self.image_encoder: nn.Module
        self.prompt_encoder: nn.Module

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            try:
                from segment_anything import sam_model_registry
                sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
                self.image_encoder = sam_model.image_encoder
                self.prompt_encoder = sam_model.prompt_encoder
                self.is_surrogate = False
            except Exception:
                if not use_surrogate_if_missing:
                    raise
                self.image_encoder = SurrogateImageEncoder(out_dim=256)
                self.prompt_encoder = SurrogatePromptEncoder(embed_dim=256)
                self.is_surrogate = True
        else:
            self.image_encoder = SurrogateImageEncoder(out_dim=256)
            self.prompt_encoder = SurrogatePromptEncoder(embed_dim=256)
            self.is_surrogate = True
```

`SurrogateImageEncoder` (dòng 610-615) tự mô tả là "**without downloading 375 MB foundation weights**" - một CNN `Conv2d`+`BatchNorm` **khởi tạo ngẫu nhiên**, không phải MedSAM.

Bằng chứng lá cờ chết -grep toàn repo:

```
=== is_surrogate usage ===
src/models/sam_wsss_adapter.py:745:        self.is_surrogate = True
src/models/sam_wsss_adapter.py:755:                self.is_surrogate = False
src/models/sam_wsss_adapter.py:761:                self.is_surrogate = True
src/models/sam_wsss_adapter.py:765:            self.is_surrogate = True
```

Bốn site **gán**, **zero** site **đọc**. Và:

```
=== surrogate in reports/experiments ===
(end)
```

Từ "surrogate" không xuất hiện **một lần nào** trong `experiments/`, `docs/reports/`, `product/`. Không log, không trường JSON, không cột CSV, không guard.

Ba cổng vào đều mặc định kích hoạt surrogate:
- `src/training/train_proposed_sam.py:444` với `--checkpoint-sam` default `None` (dòng 424) -> `checkpoint_path is None` -> surrogate.
- `src/clinical/exhaustive_evaluator.py:352-356`: `checkpoint_path=str(ckpt_p) if ckpt_p.is_file() else None, use_surrogate_if_missing=True` -> surrogate nếu thiếu file.
- `notebooks/03_Train_Proposed_MedSAM_T4.ipynb:412` (và bản L4 dòng 412, bản gốc dòng 417): `model_medsam = MedSAM_WSSS_Adapter(num_classes=5).to(DEVICE)` - **không truyền checkpoint_path** -> luôn surrogate.

Nghiêm trọng hơn, `notebooks/03_Train_Proposed_MedSAM_T4.ipynb` cell 10 còn có **bản sao inline tự viết** shadow lấy lớp trong `src`:

```python
try:
    from src.models.sam_wsss_adapter import MedSAM_WSSS_Adapter
    print("[IMPORT] Successfully imported MedSAM_WSSS_Adapter from src.models")
except ImportError:
    # Production-grade self-contained MedSAM WSSS Adapter implementation
    class MedSAM_WSSS_Adapter(torch.nn.Module):
        def __init__(self, num_classes=5, embed_dim=256):
            super().__init__()
            self.encoder_proj = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                ...
```

Nhánh fallback này tự nhận là "Production-grade" nhưng là **CNN khởi tạo từ đầu, không một gram trọng số foundation**. Nếu `src` không import được trên Colab (sai `sys.path`, thiếu `__init__.py`), notebook vẫn chạy trọn vẹn và báo cáo kết quả dưới tên "Proposed MedSAM".

**Hệ quả lâm sàng/học thuật:** đây là **phương pháp đề xuất** của đồ án. Nếu nó thực chất là một CNN random, thì mọi so sánh "proposed vs baseline" và mọi kết luận khoa học trong báo cáo mất nghĩa.

### [T2-03] CRITICAL - Pseudo-mask và ground-truth mask giả toàn zeros được sinh ra không lỗi, không cảnh báo | Severity: Critical | tin cậy RẤT CAO | VERIFIED BẰNG HÀNH VI THẬT

`src/training/dataset_loader.py:312-328`:

```python
    def _load_mask(self, mask_dir: Path, filename: str) -> np.ndarray:
        mask_path = mask_dir / filename
        if mask_path.exists():
            ...
        elif self.fallback_mask:
            # Fallback to zeros (class 0 or background) if layer mask is missing
            return np.zeros((224, 224), dtype=np.int64)
        else:
            raise FileNotFoundError(f"Pseudo-mask not found at: {mask_path}")
```

`fallback_mask` mặc định **True** (dòng 273), và `create_stage2_dataloader` (dòng 558-564) **không bao giờ truyền** `fallback_mask=False`. Tương tự cho tập đánh giá, `src/training/dataset_loader.py:456-458`:

```python
        else:
            # Fallback zero mask if mask folder is absent
            mask_np = np.zeros((224, 224), dtype=np.int64)
```

Chứng minh hành vi trên **trạng thái thật của repo**:

```
$ ls train_PM/PM_bn7 | wc -l  -> 0
$ ls train_PM/PM_b5_2 | wc -l -> 0
$ $ls train_PM/PM_b4_5 | wc -l-> 0
$ real training imgs          -> 23422
$ KMP_DUPLICATE_LIB_OK=TRUE python -c "... BCSSStage2MultiMaskDataset(img_dir='data/BCSS-WSSS/training', pm_root_dir='train_PM', is_train=False, max_samples=2)[0] ..."
stage2 dataset len: 2
label shape (224, 224) unique values [0] sum 0
label_a shape (224, 224) unique values [0] sum 0
label_b shape (224, 224) unique values [0] sum 0
```

Nghĩa là: **với đúng cây thư mục hiện tại**, Stage 2 DeepLabV3+ sẽ "học" với cả 3 lớp supervision đều là background thuần, và `BCSSEvalDataset` sẽ tính mIoU/Dice/FwIoU/Pixel-Accuracy **đối chiếu với ground truth bịa ra**. Không exception, không cảnh báo, không đếm số mẫu bị fallback.

Chướng ngại "shape mismatch sẽ lộ ra" cũng không tồn tại: ảnh BCSS-WSSS đo được đúng **224x224** (`Image.open(...).size` = `(224, 224)`) - trùng khớp con số cứng 224 trong zeros, nên lỗi **vô hình về mặt hình dạng lẫn số trị**.

### [T2-04] HẠNG MỤC SẠCH - Số liệu công bố trong `product/` có log đối chứng thật | VERIFIED

Lệnh đối chiếu từng giá trị trích trong proposal với `experiments/`:

```
70.82 -> in 4 experiment files
70.96 -> in 1 experiment files
70.75 -> in 1 experiment files
51.20 -> in 2 experiment files
64.20 -> in 2 experiment files
67.80 -> in 2 experiment files
68.90 -> in 2 experiment files
70.53 -> in 8 experiment files
```

`experiments/colab_run_3_multiscale_sota_v2/COLAB_EVALUATION_REPORT.json` có đầy đủ metadata ngữ cảnh (`run_id`, `timestamp_utc`, `hardware_gpu: "Tesla T4"`, `mode_tag`, `evaluated_patches: 4986`) và số trị full-precision đặc trưng của phép tính thật, không phải số do người/AI tự nghĩ:

```json
"mIoU": 70.53363009767762,
"mDice": 82.53642920666911,
"per_class_iou": {"TUM": 79.37309292633807, "STR": 73.8206942195482, ...}
```

Các bảng số trong proposal đều được **gắn nhãn minh bạch là kỳ vọng**: `SE374_MINI_PROPOSAL_PHASED_ROADMAP_WSSS.md:142` ("Kỳ Vọng mIoU (%)"), `:207` ("Expected Results"), `SE374_MINI_PROPOSAL_WSSS_WSI.md:173,175` ("Kết Quả Kỳ Vọng Định Lượng"). Không tìm thấy metric gán cứng nào được trình bày như kết quả thật trong `product/`.

Ghi chú kỹ thuật (không phải defect): `Macro_F1_4cls` trùng khít `mDice_4cls` (82.5364...). Đây là **đúng toán học** cho multi-class (F1 vĩ mô per-class == Dice), không phải lỗi sao chép.

### [T2-05] Chế độ mock trong downloader tiếp tục với thư mục rỗng | Severity: Medium | VERIFIED (đọc mã)

`src/colab_pkg/downloader.py:644-654`:

```python
        except Exception as exc:
            if allow_mock:
                print(f"[THÔNG BÁO] Chế độ mock cho phép tiếp tục ({exc}).")
                mask_dir.mkdir(parents=True, exist_ok=True)
                rgb_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise RuntimeError(...)
```

Tạo hai thư mục **rỗng** rồi cho pipeline đi tiếp; `samples` trả về `[]` (dòng 669-671 chỉ lặp trên `mask_files`). Không sinh dữ liệu giả nên không vi phạm [INVARIANT-07] trực tiếp, nhưng biến lỗi tải dữ liệu thành "cohort trống" hợp lệ.

---

## 3. T3 - Patient-Level Anti-Leakage ([INVARIANT-08])

### [T3-01] HẠNG MỤC SẠCH - Leakage 0.0% của BCSS là THẬT, đã kiểm chứng trên đĩa | VERIFIED

Không tin manifest, đo trực tiếp trên 23,422 ảnh thật:

```
disk train/val/test patient counts: 106 22 23
disk pairwise overlaps: 0 0 0
manifest==disk train: True  val: True  test: True
symdiff train: [] val: [] test: []
```

Và qua loader thật:

```
MANIFEST train/val/test counts: 106 22 23
overlap t/v: set() t/te: set()
```

Tuyên bố "0.0% patient leakage (151 cases)" của tiền audit là **chính xác về mặt dữ liệu**. Đây là phát hiện quan trọng cần ghi nhận: kiểm soát leakage đạt được **nhờ cấu trúc thư mục do tác giả BCSS-WSSS cung cấp sẵn**, chứ không nhờ code (xem [T3-06]).

### [T3-02] Fallback bệnh nhân cứng 7 người khi thiếu manifest, không báo lỗi | Severity: High | VERIFIED BẰNG HÀNH VI

`src/training/dataset_loader.py:127-133`:

```python
    if not config_path.exists():
        # Fallback to local default patient lists if config file does not exist
        return {
            "train": ["TCGA-A1-A0SK", "TCGA-A2-A04N", "TCGA-A2-A0T2"],
            "val": ["TCGA-AN-A03X", "TCGA-AN-A0AL"],
            "test": ["TCGA-AQ-A04H", "TCGA-C8-A1HJ"],
        }
```

Chứng minh:

```
$ python -c "from src.training.dataset_loader import load_bcss_patient_splits as L; \
  print(L('Z:/definitely_missing_manifest.json'))"
FALLBACK (file missing) counts: 3 2 2
FALLBACK ids: {'train': ['TCGA-A1-A0SK', 'TCGA-A2-A04N', 'TCGA-A2-A0T2'], 'val': ['TCGA-AN-A03X', 'TCGA-AN-A0AL'], 'test': ['TCGA-AQ-A04H', 'TCGA-C8-A1HJ']}
```

Không `FileNotFoundError`, không `warnings.warn`, không log. Đối chiếu: `src/datasets/patient_split.py:131-132` (bản loader song song) **có** raise `FileNotFoundError`. Hai hàm cùng nhiệm vụ, một im lặng trả dữ liệu sai, một báo lỗi.

Giảm nhẹ: `scripts/build_drive_package.py:1074` có lọc `.json` khi copy `src/`, và kiểm tra thực tế cho thấy `drive_upload_ready/SE374_WSSS_Drive_Code/src/config/bcss_patient_splits.json` **có tồn tại**, nên đường Colab chuẩn không kích hoạt fallback. Defect vẫn nguy hiểm cho mọi đường chạy khác (clone GitHub trực tiếp, `git clone` sparse, thư mục làm việc tay).

### [T3-03] Manifest rỗng ==> bộ lọc bệnh nhân bị TẮT hoàn toàn ==> leakage 100% im lặng | Severity: High | VERIFIED (mã) + INFERENCE (hành vi)

`src/training/dataset_loader.py:511` và `:529`:

```python
        patient_ids=train_patients if train_patients else None,
```

kết hợp `:198-200`:

```python
        allowed_set: Optional[Set[str]] = (
            {p.upper() for p in patient_ids} if patient_ids is not None else None
        )
```

và `:203-208` (nếu `allowed_set is None` thì **mọi file đều được nhận**). Chuỗi hệ quả: nếu JSON tồn tại nhưng danh sách `train`/`val` **rỗng** (thiếu key `"splits"`, file bị cut-off, schema đổi tên key - xem [T3-04]) thì `train_patients = set()` -> falsy -> truyền `None` -> **không lọc gì cả** -> tập train nuốt trọn cả 151 bệnh nhân, bao gồm 22 val + 23 test. Leakage toàn phần, báo cáo vẫn in số liệu.

Chứng minh phân biệt hai trạng thái (đã chạy):

```
with patient_ids=set() -> 0 files
```

`set()` (rỗng nhưng không phải `None`) lọc sạch -> crash; còn `None` thì mở toang. Đúng ra cả hai phải là lỗi. Hành vi này **không có test nào bao phủ**.

### [T3-04] Loader `src/` không hề gọi `verify_patient_leakage` | Severity: High | VERIFIED

```
=== load_bcss_patient_splits / load_patient_splits defs ===
src/datasets/patient_split.py:29:def verify_patient_leakage(
src/datasets/patient_split.py:103:def load_patient_splits(
src/training/dataset_loader.py:113:def load_bcss_patient_splits(
```

`src/datasets/patient_split.py:147` có chốt chặn:

```python
    # Enforce patient-level anti-leakage verification
    verify_patient_leakage(train_ids, val_ids, test_ids)
```

Nhưng `src/training/dataset_loader.py:135-151` - hàm mà **actual pipeline** dùng - chỉ parse JSON và return, **không gọi verify**. Hai hàm tải cùng một manifest, một có kiểm định một không. Pipeline sản xuất số liệu đi qua bản không kiểm định.

### [T3-05] CRITICAL - Bốn bản sao `extract_patient_id` trong `src/` cho ra BỐN danh tính khác nhau cho cùng một patch | Severity: Critical | tin cậy RẤT CAO | VERIFIED BẰNG HÀNH VI

Các định nghĩa:

```
src/colab_pkg/downloader.py:76:def extract_patient_id(filename: str) -> str:
src/colab_pkg/evaluator.py:58:        def extract_patient_id(fn: str) -> str:
src/datasets/patient_split.py:7:def extract_patient_id(filepath_or_name: str) -> str:
src/training/dataset_loader.py:51:def extract_patient_id(filepath_or_name: Union[str, Path]) -> str:
```

Khác biệt cốt lõi: bản `patient_split.py:20` dùng **`re.search`** trên **basename có đuôi**; bản `dataset_loader.py:63` và `downloader.py:79` dùng **`re.match`** (neo đầu chuỗi); chỉ `patient_split.py:16` và `dataset_loader.py:60` gọi `os.path.basename`, còn `downloader.py:79` **không** và fallback bằng `filename.split("_")[0]`.

Trích bảng đối chiếu thật:

```
FILENAME                                                   patient_split    dataset_loader   downloader
patch_TCGA-A2-A0T2_001.png                                 TCGA-A2-A0T2     PATCH            patch
TCGA-A2-A0T2_1024_2048_[1 0 0 1].png                       TCGA-A2-A0T2     TCGA-A2-A0T2     TCGA-A2-A0T2
TCGA-D8-A27F-DX1_xmin98787_ymin6725_MPP-0.2500+0.png       TCGA-D8-A27F     TCGA-D8-A27F     TCGA-D8-A27F
GDPH_1234_patch_01.png                                     GDPH             GDPH             GDPH
a/TcgA-E2-A152_512_1024.png                                TCGA-E2-A152     TCGA-E2-A152     a/TcgA-E2-A152
slide_01_TCGA-C8-A1HJ+002+[1 0 0 0].png                    TCGA-C8-A1HJ     SLIDE            slide
```

Ba hệ quả trực tiếp:
1. Dòng 1 và 6: **cùng một bệnh nhân thật** ánh xạ tới `TCGA-A2-A0T2` / `PATCH` và `TCGA-C8-A1HJ` / `SLIDE`. `verify_patient_leakage` lấy giao tập các nhãn hiệu; khi hai phía của một split dùng hai hàm khác nhau, giao tập rỗng **giả tạo** -> leakage trở nên **vô hình với chính bộ máy phát hiện leakage**. Đây là lỗi phá vỡ toàn bộ [INVARIANT-08] ở tầng công cụ.
2. Dòng 5: `downloader` giữ nguyên thành phần đường dẫn `a/TcgA-E2-A152` vì không basename. Chuỗi ký tự này **khác cả hoa thường** so với hai bản trên (`.upper()`), nên mọi phép so sánh đều sai.
3. Dòng 6: docstring của cả hai module (`dataset_loader.py:57-58`, `patient_split.py:12-14`) **tuyên bố hỗ trợ** format LUAD-HistoSeg `TCGA-XX-XXXX+001+[1 0 0 0].png`, nhưng chỉ hỗ trợ khi TCGA đứng ở vị trí 0; có tiền tố là hỏng.

Còn cộng thêm 7 bản sao inline trong notebook (`notebooks/SE374_Colab_WSSS_{T4,L4,A100}_Full4986.ipynb:304/304/304`, `..._T4_Sanity150.ipynb:309`, `..._Pipeline.ipynb:305`, `history_runs/Run1:261`, `Run2:347`).

### [T3-06] `BCSSDataset` không lọc theo Patient ID và không đọc manifest | Severity: High | VERIFIED

`src/datasets/bcss_dataset.py:48-79` (toàn bộ hàm `_load_dataset`) chỉ nối đường dẫn + `glob`, **không tham chiếu** `load_patient_splits`, `patient_split`, hay manifest nào:

```python
        if is_train_split:
            train_dir = os.path.join(self.root_dir, "training")
            ...
            pattern = os.path.join(train_dir, "*.png")
            self.image_paths = sorted(glob.glob(pattern))
```

Và đây chính là dataset được phương pháp đề xuất dùng, `src/training/train_proposed_sam.py:487-489`:

```python
        train_ds = BCSSDataset(root_dir=args.data_dir, split="train")
        val_ds = BCSSDataset(root_dir=args.data_dir, split="valid")
```

`self.patient_ids` được **thu thập** (dòng 59, 76) nhưng **không bao giờ được dùng để lọc hay kiểm tra**. An toàn leakage hiện tại hoàn toàn là **ôm hàm ơn cấu trúc thư mục của nhà cung cấp dữ liệu** (đã chứng minh ở [T3-01]), không phải một bất biến do code thực thi. Nếu người dùng tải một bản BCSS-WSSS đóng gói khác (ví dụ gộp cả val vào `training/`, hoặc dùng layout `BCSS-WSSS/train/img`), leakage xảy ra tức khắc và **không có một dòng code nào trong `src/` phát hiện**.

### [T3-07] Test leakage tự tham chiếu: chứng minh JSON rời rạc, không chứng minh hệ thống rời rạc | Severity: Medium | VERIFIED

`tests/test_patient_split.py:94-145` (`test_151_patient_split_manifest_zero_leakage`) là test **có thật** và **có hành vi**: nó đọc manifest, kiểm tra tổng 151, kiểm tra 3 giao tập rỗng. Nhưng phạm vi của nó **dừng lại ở tệp JSON**. Không một assertion nào đi tới `create_stage1_dataloaders` / `BCSSStage1Dataset` để xác nhận **bộ lọc đã được áp dụng**, nên [T3-02], [T3-03], [T3-04], [T3-06] đều lọt.

Test hành vi mạnh nhất, `:32-49`, chạy trên `data/sample_dataset` - đo được tổng cộng **4 ảnh train / 2 val / 2 test** với chỉ **2+1+1 = 4 bệnh nhân**:

```python
    assert result["train_patients_count"] == 2
    assert result["val_patients_count"] == 1
    assert result["test_patients_count"] == 1
```

Chứng minh "không leakage" trên một bộ 4 bệnh nhân nhân tạo có cấu trúc thư mục đã chia sẵn là bằng chứng rất yếu.

Test tự khóa sai lầm, `:28-29`:

```python
    f4 = "GDPH_1234_patch_05.png"
    assert extract_patient_id(f4) == "GDPH"
```

Docstring của chính module bị test (`src/datasets/patient_split.py:14`) ghi `GDPH_1234_patch_01.png -> 'GDPH_1234'`, còn `src/training/dataset_loader.py:58` cũng ghi `GDPH_1234_patch_01.png -> 'GDPH_1234'`. Hành vi thật là `GDPH` (đã đo ở bảng [T3-05]). Test **assert giá trị sai và biến nó thành hợp thức**, đồng thời mâu thuẫn với hợp đồng tài liệu của chính nó.

`INFERENCE`: vì `patient_id` dùng `re.split(r"[_+]", fname)[0]`, với tên `GDPH_1234_patch_01` toàn bộ hậu tố bệnh nhân (`1234`) bị bỏ -> **mọi patch của mọi bệnh nhân GDPH ánh xạ về cùng một ID "GDPH"** -> nhiều bệnh nhân thật bị coi là một, làm méo cả số lượng bệnh nhân lẫn tính disjoint.

### [T3-08] Quét mở rộng: không có `train_test_split`, `GroupKFold`, `shuffle=True` chia theo patch | VERIFIED (sạch)

`src/training/dataset_loader.py:519` và `:569` dùng `shuffle=True` cho DataLoader - đây là xáo **trong nội bộ một split đã lọc theo bệnh nhân**, không phải xáo qua ranh giới split, nên **không vi phạm** [INVARIANT-08]. `split_patients_kfold` (`src/datasets/patient_split.py:76-100`) dùng `random.Random(seed)` shuffle trên **danh sách Patient ID đã `sorted(set(...))`** -> đúng tầng bệnh nhân, deterministic. Không tìm thấy `train_test_split` hay `StratifiedKFold` nào trong `src/`.

---

## 4. T4 - Anti-Shallow-Glob ([INVARIANT-29])

### [T4-01] `resolve_split_path` do luật nêu tên không tồn tại trong `src/` | Severity: High | VERIFIED

```
=== resolve_split_path ===
./01_Train_Stage1_Classifier.ipynb:190
./01_Train_Stage1_Classifier_L4.ipynb:184
./01_Train_Stage1_Classifier_T4.ipynb:184
./01_Train_Stage1_Classifier_v2.ipynb:184
./02_Pseudo_Masks_Train_Stage2_T4.ipynb:199
./03_Train_Proposed_MedSAM_T4.ipynb:165
./notebooks/01_Train_Stage1_Classifier*.ipynb ...
```

**Zero** kết quả trong `src/`. Hàm đa tầng chỉ sống trong notebook -> `src/` là package được Colab import, nhưng lại là nơi **không có** cơ chế nhận diện thư mục lồng nhau mà [INVARIANT-29] bắt buộc.

### [T4-02] Glob nông hiện diện dày đặc trong `src/` | Severity: High | VERIFIED

```
src/clinical/exhaustive_evaluator.py:394:        image_files = sorted(list(img_dir.glob("*.png")