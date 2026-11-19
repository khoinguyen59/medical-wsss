# KIỂM ĐỊNH ĐỘC LẬP - TÁC TỬ 04 (SWARM RECHECK 2026-11-05)

- Phạm vi: TOÀN BỘ hệ thống, TRỌN 10 chiều T1-T10. Không chia nhỏ module.
- Phương pháp: đọc mã + chạy lệnh READ-ONLY trên máy chủ (Windows, anaconda3, Python 3.x, CPU-only).
- Ràng buộc đã tuân thủ: mọi lệnh python chạm torch/numpy đều prepend `KMP_DUPLICATE_LIB_OK=TRUE`; không sửa source; không sửa test; không tạo script mới trong `src/`, `tests/`, `scripts/`; không có GPU nên mọi đường dẫn GPU chỉ audit tĩnh.
- Quy ước ký hiệu:
  - `VERIFIED` = có lệnh đã chạy và output thật được dán kèm.
  - `INFERENCE` = suy luận từ đọc mã, chưa chạy được trên máy này.
  - Lệnh đều chạy tại `SE374_WSSS_WORKSPACE`.

---

## 0. KẾT LUẬN NGẮN (đếm theo severity)

| Severity | Số defect |
|---|---|
| CRITICAL | 6 |
| HIGH | 7 |
| MEDIUM | 6 |
| LOW | 4 |
| Tổng | 23 |

Ba kết luận trọng tâm:

1. Tuyên bố "348 passed / 0 failed" của đợt kiểm định 2026-11-02 **không tái lập được một cách ổn định**: 4 lần gọi `pytest tests/ -q` trên cùng một working tree cho ra **3 kết quả khác nhau** (4 failed, 1 failed, 0 failed). `VERIFIED`.
2. Tuyên bố "Ruff reported 0 lint errors" chỉ đúng khi thu hẹp phạm vi vào `src/`; phạm vi toàn repo là **464 errors**. `VERIFIED`.
3. Bảo đảm "0.0% patient leakage" **đúng trên dữ liệu thật hôm nay** (tôi xác minh lại và xác nhận), NHƯNG cơ chế bảo vệ nó bị hỏng ở nhiều nơi: một nhánh fallback tạo mask toàn số 0 khiến `mIoU = 100.0` được suy ra và được chọn làm best checkpoint, và hàm khám phá ảnh đệ quy của repo hút cả **mask ground-truth lẫn output dự đoán của model** vào cohort đánh giá. `VERIFIED`.

---

## 1. ĐỐI CHIẾU TUYÊN BỐ CŨ vs SỐ ĐO LẠI

Tệp được đem ra kiểm định: `docs/reports/audit/2026-11-02-swarm-audit/01_SWARM_RECHECK_VERIFICATION.md`.

| # | Tuyên bố 2026-11-02 | Đo lại 2026-11-05 | Phán quyết |
|---|---|---|---|
| 1 | "348 passed, 0 failed, 2 warnings" trong 104.15s | Lần 1: `1 failed, 347 passed, 2 warnings in 105.78s`; lần 2: `348 passed, 2 warnings in 97.66s`; trước đó trong phiên: `4 failed, 344 passed`. Test trượt đổi theo lần chạy. | **SAI (phóng đại độ ổn định)** |
| 2 | "Ruff reported 0 lint errors across all 42 modules in src/" | `src/` = 0 thật. Toàn repo = **Found 464 errors** (33 trong `scripts/`, 386 trong `notebooks/`, 228 trong `drive_upload_ready/`; trùng lặp theo hệ số vì một số tệp bị đếm ở nhiều scope). | **ĐÚNG nhưng THU HẸP sai phạm vi** |
| 3 | "0.0% Strict Leakage (151 Cases)" | Đo lại trên `src/config/bcss_patient_splits.json`: 106/22/23 bệnh nhân, pairwise overlap = 0/0/0; disk-vs-manifest mỗi split khớp tuyệt đối số TCGA ID (train 23422 png, val 6836, test 19949). | **ĐÚNG (tôi xác nhận, xem mục 3.3)** |
| 4 | "0 non-ASCII hyphens across the entire workspace" | 5 tệp loại `.py/.md/.ipynb` mà test nhìn thấy: **0 vi phạm** (đúng). Nhưng `glob("**/*.md", recursive=True)` của test **mù với thư mục ẩn**: 730 tệp nằm trong dot-directory không được quét, trong đó **48 vi phạm thuộc tài sản dự án** (`.agents/...`). | **SAI về phạm vi quét** |
| 5 | "Exactly 32 .ipynb" | `notebooks/` (cả subtree) = 24+1+3+4 = **32** -> đúng nếu chỉ tính `notebooks/`. Toàn cây: 82 tệp `.ipynb` (10 ở gốc, 18 trong `drive_upload_ready/`, 25 trong `repos/`, 29 còn lại trong `notebooks/`). | **ĐÚNG một phần (định nghĩa phạm vi không nêu)** |
| 6 | Victory Matrix "PASS (100%)" cho chiều Anti-Mockup | Có 3 cơ chế dữ liệu/giả lập thật đang hoạt động mặc định trong `src/` (mục 2.1, 2.2, 2.3). | **SAI** |

Trích output nguyên văn của hai lệnh chính:

```text
$ KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q --tb=no
1 failed, 347 passed, 2 warnings in 105.78s (0:01:45)
...
=========================== short test summary info ===========================
FAILED tests/test_governance_invariants.py::test_no_ai_em_dash_in_markdown - ...
```

```text
$ KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q --tb=line -p no:cacheprovider
348 passed, 2 warnings in 97.66s (0:01:37)
```

```text
$ KMP_DUPLICATE_LIB_OK=TRUE python -m ruff check . --output-format concise | tail -3
scripts\update_agents_invariants.py:1:8: F401 [*] `os` imported but unused
Found 464 errors.
[*] 385 fixable with the `--fix` option (1 hidden fix can be enabled with the `--unsafe-fixes` option).
```

Số test function tĩnh: **265** `def test_` trên **22** tệp `tests/test_*.py`; pytest báo 348 vì có parametrize/hypothesis nở rộng. `VERIFIED`.

---

## 2. DEFECT CRITICAL

### 2.1 [T2-synthetic-fallback] | CRITICAL | VERIFIED

Vị trí: `src/training/train_proposed_sam.py:485-494`

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

`SyntheticHistologyDataset` (`src/training/train_proposed_sam.py:195-217`) sinh ảnh bằng `torch.manual_seed(42+idx)` rồi `torch.rand` / `torch.randint(0, 5, ...)`; nhãn phân đoạn là ngẫu nhiên, không có ý nghĩa mô học.

Điều kiện vào nhánh là `os.path.exists(args.data_dir)` với default `--data-dir data/BCSS-WSSS` (dòng 423). Chỉ cần thiếu dữ liệu (Colab mount lỗi, GDown thiếu quyền, đường dẫn khác) là toàn bộ vòng training phương pháp đề xuất chạy trên ảnh nhiễu ngẫu nhiên, và ở dòng 528-533 cùng 541 vẫn in và lưu `"best_mIoU"` vào checkpoint như một kết quả khoa học.

Lệnh tái hiện (đã chạy, không có dữ liệu thật trong `data_dir` giả):

```text
$ KMP_DUPLICATE_LIB_OK=TRUE python -c "...import SyntheticHistologyDataset; d=SyntheticHistologyDataset(num_samples=4,image_size=224); print(d[0][0].shape, torch.unique(d[0][1]))"
torch.Size([3, 224, 224]) tensor([0, 1, 2, 3, 4])
```

Hệ quả: một con số mIoU từ ảnh nhiễu có thể đi vào bảng kết quả. Đây chính là điều INVARIANT-07 (Zero-Mockup) cấm.

Ghi chú thêm: `src/training/train_proposed_sam.py:257-260` dựng `surrogate_cams` từ `images.mean(dim=1) * weak_labels` thay vì CAM thật. `VERIFIED` bằng đọc dòng lệnh; `INFERENCE` về tác động số học vì chưa chạy GPU.

### 2.2 [T2-zeros-gt-perfect-metric] | CRITICAL | VERIFIED

Vị trí 1: `src/training/dataset_loader.py:324-326` (mặc định `fallback_mask: bool = True` quanh dòng 273)

```python
        elif self.fallback_mask:
            # Fallback to zeros (class 0 or background) if layer mask is missing
            return np.zeros((224, 224), dtype=np.int64)
        else:
            raise FileNotFoundError(f"Pseudo-mask not found at: {mask_path}")
```

Vị trí 2: `src/training/dataset_loader.py:445-458` - `BCSSEvalDataset.__getitem__` kiểm tra từng tệp; nếu tên mask không khớp tên ảnh thì **không có lựa chọn nào ngoài việc bù số 0, và không phát cảnh báo**:

```python
        mask_path = self.mask_dir / img_path.name
        if mask_path.exists():
            ...
        else:
            # Fallback zero mask if mask folder is absent
            mask_np = np.zeros((224, 224), dtype=np.int64)
```

Vị trí 3 (bệ phóng cho defect): `src/losses_metrics/metrics.py:71-88` tính mIoU trung bình **chỉ trên các lớp có mặt**:

```python
        iou = np.zeros(self.num_classes, dtype=np.float64)
        valid_classes = denominator > 0
        iou[valid_classes] = tp[valid_classes] / denominator[valid_classes]
        # mIoU over present classes
        miou = float(np.mean(iou[valid_classes])) if np.any(valid_classes) else 0.0
```

Khi toàn bộ cohort có mask rỗng, chỉ class 0 là "present" và `IoU_0 = 1.0` => mIoU = 100.

Vị trí 4 (chuyển thành "kết quả tốt nhất"): `src/training/train_stage2.py:336` dựng `val_loader = create_eval_dataloader(...)` (đi đúng vào đường mask-rỗng), rồi `src/training/train_stage2.py:368-370`:

```python
        current_miou = eval_metrics.get("mIoU", 0.0)
        if current_miou >= best_miou or dry_run:
            best_miou = current_miou
            torch.save(
```

Lệnh tái hiện (đã chạy, output thật):

```text
$ KMP_DUPLICATE_LIB_OK=TRUE python -c "<feed all-zero GT + all-zero pred qua MedicalSegmentationMetrics>"
GT all-zero (fallback)       mIoU=100.0 mDice=100.0 IoU_TUM=100.0 IoU_STR=0.0 PA=100.0
GT with real tumour+stroma   mIoU=21.1 mDice=25.8 IoU_TUM=63.3 IoU_STR=0.0 PA=63.3
```

Hệ quả: đây là kịch bản tồi tệ nhất của Anti-Mock-Testing Illusion Gate. Một hỏng hóc I/O thuần túy (đổi tên tệp mask, sai `pm_root_dir`, mount hờ) biến model **không segment được gì** thành model **điểm tuyệt đối**, và `train_stage2.py` sẽ lưu nó làm best checkpoint rồi in ra như kết quả cuối kỳ. Không có bất kỳ assertion nào chặn `mIoU == 100.0` hoặc kiểm tra `np.unique(target)` có ít nhất 2 lớp.

### 2.3 [T2-surrogate-medsam-default] | CRITICAL | VERIFIED

Vị trí: `src/models/sam_wsss_adapter.py:609-650` (định nghĩa) và `735-766` (chọn khởi tạo).

```python
class SurrogateImageEncoder(nn.Module):
    """Lightweight 4-stage residual convolutional encoder matching ViT-B output shape (B, 256, H/16, W/16).

    Guarantees deterministic execution and exact shape parity on CPU and GPU
    without downloading 375 MB foundation weights.
```

```python
        use_surrogate_if_missing: bool = True,
...
        self.is_surrogate = True
...
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            try:
                from segment_anything import sam_model_registry
                sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
                ...
                self.is_surrogate = False
            except Exception:
                if not use_surrogate_if_missing:
                    raise
                self.image_encoder = SurrogateImageEncoder(out_dim=256)
                ...
        else:
            self.image_encoder = SurrogateImageEncoder(out_dim=256)
            self.prompt_encoder = SurrogatePromptEncoder(embed_dim=256)
            self.is_surrogate = True
```

Ba lỗi gộp lại: (a) mặc định `use_surrogate_if_missing=True`; (b) `checkpoint_path=None` là mặc định, nên **construction mặc định luôn ra surrogate**; (c) `except Exception` nuốt mọi lỗi nạp thật (thiếu package, checksum sai, hết dung lượng) và thay bằng surrogate mà không raise, không log ở mức WARNING.

Lệnh tái hiện (đã chạy):

```text
$ KMP_DUPLICATE_LIB_OK=TRUE python -c "<construct mặc định>"
DEFAULT construction -> is_surrogate = True
image_encoder class   = SurrogateImageEncoder
prompt_encoder class  = SurrogatePromptEncoder
total params = 7.87M  (MedSAM ViT-H official is ~636M)
=== strict=False count === 9
```

Bằng chứng chết (dead governance flag):

```text
$ grep -rn "is_surrogate" src/ --include=*.py
src/models/sam_wsss_adapter.py:745:        self.is_surrogate = True
src/models/sam_wsss_adapter.py:755:                self.is_surrogate = False
src/models/sam_wsss_adapter.py:761:                self.is_surrogate = True
src/models/sam_wsss_adapter.py:765:            self.is_surrogate = True

$ grep -c "is_surrogate" notebooks/*.ipynb  -> (không tệp nào)
$ grep -rn "is_surrogate" tests/            -> (không kết quả)
```

Hệ quả: cờ trung thực tồn tại nhưng **không một downstream nào đọc nó**: không evaluator, không notebook, không test, không deliverable auditor. Báo cáo cuối kỳ có thể công bố số "MedSAM + Residual Adapter" trong khi thực chất là một CNN 7.87M tham số tự chế. Đối chiếu: tài liệu báo cáo (§6.1) nói về "tensor weights" của MedSAM; `strict=False` với 9 khóa bị bỏ qua cho thấy ngay cả khi có checkpoint thật, ánh xạ trọng số cũng không được xác minh.

### 2.4 [T3-test-dir-contains-predictions] | CRITICAL | VERIFIED

Thư mục dữ liệu test của BCSS-WSSS chứa chính output của model và biểu đồ đánh giá:

```text
$ for d in data/BCSS-WSSS/test/img data/BCSS-WSSS/test/mask data/BCSS-WSSS/test/pred data/BCSS-WSSS/test/pred_palette data/BCSS-WSSS/test/miou_out data/BCSS-WSSS/val/img data/BCSS-WSSS/val/mask; do echo -n "$d : "; find "$d" -maxdepth 1 -name "*.png" | wc -l; done
data/BCSS-WSSS/test/img : 4986
data/BCSS-WSSS/test/mask : 4986
data/BCSS-WSSS/test/pred : 4986
data/BCSS-WSSS/test/pred_palette : 4986
data/BCSS-WSSS/test/miou_out : 4
data/BCSS-WSSS/val/img : 3418
data/BCSS-WSSS/val/mask : 3418
```

Và helper khám phá ảnh **được mã nguồn dùng làm đường vào chuẩn** (`src/training/dataset_loader.py:154-172`, mặc định `recursive: bool = True`, dùng `root_path.rglob("*")`) hút tất cả vào một cohort:

```text
$ KMP_DUPLICATE_LIB_OK=TRUE python -c "from src.training.dataset_loader import find_image_files; ..."
find_image_files(test, recursive=True) = 19949
   img 4986
   mask 4986
   pred 4986
   pred_palette 4986
   miou_out 4
   test 1
```

Nghĩa là cohort "test" 4,986 ảnh bị nhân 4 lần bằng cách đưa vào: mask ground-truth, **projected predictions của chính model**, ảnh palette trực quan hóa và 4 biểu đồ PNG (`Precision.png`, `Recall.png`, `mIoU.png`, `mPA.png`) cùng một tệp `1.png` nằm rời.

Định danh bệnh nhân của các tệp rác cũng bị biến thành "bệnh nhân":

```text
$ KMP_DUPLICATE_LIB_OK=TRUE python -c "non-TCGA pseudo-patient scan"
non-TCGA pseudo-patient files: 5
  1 1
  1 PRECISION
  1 RECALL
  1 MIOU
  1 MPA
--- examples ---
  data/BCSS-WSSS\test\1.png
  data/BCSS-WSSS\test\miou_out\Precision.png
  data/BCSS-WSSS\test\miou_out\Recall.png
  data/BCSS-WSSS\test\miou_out\mIoU.png
  data/BCSS-WSSS\test\miou_out\mPA.png
```

Cộng hưởng với nhánh fall-through của `src/training/dataset_loader.py:478-488` (`else: train_dir = data_path`, tức là quét **gốc** dataset) và `:500-513` (`patient_ids=train_patients if train_patients else None`, tức là bộ lọc biến mất khi manifest trả về danh sách rỗng), đo được mức rò rỉ tối đa:

```text
$ KMP_DUPLICATE_LIB_OK=TRUE python -c "<find_image_files('data/BCSS-WSSS') + classify by manifest>"
find_image_files(ROOT, recursive) = 50207
{'test': 19949, 'training': 23422, 'val': 6836}
if train_dir=ROOT and patient_ids=None: {'unknown': 5, 'train': 23422, 'val_or_test': 26780}
=> 26780 patches from val/test would enter TRAIN
```

Docstring của chính hàm này (dòng 475) cam kết "Ensures 0.0% patient-level leakage between train and val splits." Con số 26,780 bẻ gãy cam kết đó ở mức kiến trúc, kể cả khi hôm nay chưa ai gọi đúng cấu hình đó.

### 2.5 [T7-evaluator-selfheal-tunes-on-test] | CRITICAL | VERIFIED

Vị trí: `src/colab_pkg/evaluator.py:1069-1100` và `1108-1162`.

```python
    # Self-heal if test_files is empty or accidentally resolved to the 2-image repo sample_dataset on Colab
    if len(test_files) == 0 or (
        Path("/content").is_dir()
        and any("sample_dataset" in str(p) for p in (list(val_files) + list(test_files)))
    ):
        from .downloader import ensure_bcss_dataset, resolve_split_dirs
        real_root, real_t_img, real_t_mask = ensure_bcss_dataset()
        _, real_v_img, real_v_mask = resolve_split_dirs(real_root, split="val")
        cap = max(len(val_files), 50)
        ...
        else:
            all_real = sorted(real_t_img.glob("*.png"))
            pmap: Dict[str, List[Path]] = defaultdict(list)
            for p in all_real:
                pmap[extract_patient_id(p.name)].append(p)
            upids = sorted(pmap.keys())
            if len(upids) >= 2:
                nv = max(1, len(upids) // 3)
                vp, tp = set(upids[:nv]), set(upids[nv:])
                val_files[:] = [p for pid in vp for p in pmap[pid]][:cap]
                test_files[:] = [p for pid in tp for p in pmap[pid]][:cap]
            ...
            val_mask_dir = real_t_mask
            test_mask_dir = real_t_mask
```

Ba vi phạm phạm trù trong cùng một khối:

1. "Validation" được **cắt ra từ thư mục TEST** (`all_real = sorted(real_t_img.glob("*.png"))`), không phải từ `val/`.
2. Cả `val_mask_dir` lẫn `test_mask_dir` cùng trỏ `real_t_mask` (dòng 1099-1100): hai vai trò khoa học khác nhau dùng chung một nguồn ground-truth.
3. Vòng sweep siêu tham số chạy trên "pseudo-val" đó (dòng 1108-1157, biến `min_island_area` và `m6_cam_fusion_alpha`), rồi dòng 1158 in ra:

```text
--> BƯỚC 2: Đánh giá độc lập trên tập Held-out Test với siêu tham số đã khóa (Zero-Tuning)
```

Nhãn "Zero-Tuning held-out test" là **sai bản chất**: tham số đã được chọn trên một phần của chính tập test. Đây là selection bias kinh điển, đủ để vô hiệu hóa mọi tuyên bố tổng quát hóa chéo-tập (LUAD-HistoSeg / GDPH) và mọi giá trị p của log-rank phía sau.

Điều kiện kích hoạt chỉ cần `len(test_files) == 0`, tức là chỉ cần một lỗi mount/glob nhỏ là protocol khoa học tự động biến thành tuning-on-test mà không raise, không in WARNING.

### 2.6 [T2-report-provenance-file-missing] | CRITICAL | VERIFIED

Tệp `product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md`, mục 6.1 (khoảng dòng 188-196) tuyên bố số liệu "được ghi nhận thật từ tensor weights trong tệp `experiments/benchmarks/benchmark_results.json`".

```text
$ ls -la experiments/benchmarks/
-rw-r--r-- 1 ... 4921 Oct 22 20:26 COLAB_3_FULL_RUNS_EVOLUTION_COMPARISON.md
-rw-r--r-- 1 ... 1119 Oct 22 20:26 README.md

$ find . -name "benchmark_results.json" -not -path "./.git/*"
(không có kết quả)

$ git ls-files | grep -i benchmark_results
product/reports/clinical/exhaustive_3way_benchmark_results.json
```

Tệp duy nhất cùng họ tên là một artifact khác và **không chứa trường nào của bảng §6.1**:

```text
keys = ["timestamp", "evaluation_device", "benchmark_models", "statistical_significance", "clinical_survival_analysis"]
seed 0 | std 0 | mean 2 | "Baseline CAM" 0 | "ClassMix" 0 | "FPS" 0 | mIoU 6 | len 6451
```

Bảng §6.1 yêu cầu 4 cấu hình x 5 seed với `mean ± std` và FPS; tệp được viện dẫn không tồn tại và tệp gần nhất không có `seed`, `std`, hay tên cấu hình. Kết quả tìm được của §6.1 không thể tái dựng từ bất kỳ artifact nào trong repo.

Tương quan nội bộ cũng tự mâu thuẫn (đọc tệp báo cáo, `VERIFIED` bằng grep chuỗi):
- Hàng "Baseline CAM" và hàng "ClassMix" trùng khít từng số: $6.32\pm4.96$, $8.80\pm6.38$, $27.93\pm22.43$. Hai phương pháp khác nhau cho ra năm seed giống hệt nhau về mặt chữ số là không có thực.
- Cột "Stroma IoU" = $0.00\pm0.00$ cho **cả bốn** cấu hình, tức lớp STR không bao giờ được dự đoán đúng; nhưng §6.3 vẫn so sánh với literature ladder 51.20 -> 73.10 và kết luận "khoảng cách ... dưới 2.14% mIoU".
- §6.2 viết tỉ lệ dương-tính-giả mô "vượt 15%" cho "Baseline và ClassMix", trong khi bảng của chính nó cho ClassMix xấp xỉ 8%.

Phán quyết: đây không phải lỗi chính tả số liệu; đây là khoảng trống provenance. Không thể kiểm định độc lập nếu không có tệp nguồn.

---

## 3. DEFECT HIGH

### 3.1 [T1-flaky-suite] | HIGH | VERIFIED (hiện tượng) + INFERENCE (nguyên nhân)

Cùng một lệnh, trên cùng một working tree, bốn lần cho ba kết quả:

| Lần | Lệnh | Kết quả |
|---|---|---|
| A (đầu phiên) | `pytest tests/ -q --tb=no` | `4 failed, 344 passed` |
| B | `pytest tests/ -q --tb=no` | `1 failed, 347 passed` - FAILED `tests/test_governance_invariants.py::test_no_ai_em_dash_in_markdown` |
| C | `pytest tests/ --tb=long -q` | tất cả pass (grep short-summary rỗng) |
| D | `pytest tests/ -q --tb=line -p no:cacheprovider` | `348 passed, 2 warnings` |

Kiểm tra tính cô lập:

```text
$ python -m pytest tests/test_governance_invariants.py -q --tb=long
4 passed in 2.23s

$ python -m pytest tests/ -q --tb=line -k "em_dash"
5 passed, 343 deselected in 7.65s
```

Đo lại ngay sau lần trượt: số vi phạm dash trong tập tệp mà test nhìn thấy = **0**. Vậy test trượt vì **tại thời điểm đó tồn tại một tệp `.md` có dash cấm rồi bị ghi đè/ xóa** - cơ chế race trên filesystem, không phải lỗi logic ổn định.

Nguyên nhân ứng viên (INFERENCE, có chỉ điểm):
- `tests/test_thin_drive_runner.py:94` (`test_2`) và `:489` (`test_7`) **đều gọi** `packager.build_drive_package()`; `scripts/build_drive_package.py:1545` mở đầu bằng `shutil.rmtree(bundle_dir)` trên thư mục thật `drive_upload_ready/` (được khai báo ở `tests/test_thin_drive_runner.py:42-45` là `PROJECT_ROOT / "drive_upload_ready"`). Đây vừa là nguồn flakiness, vừa là hành vi **tự phá hủy artifact bàn giao** trong lúc chạy test (khớp với ghi chú "Audit side effects" của dự án).
- Song song, các tác tử swarm khác đang viết `docs/reports/audit/2026-11-05-swarm-audit/agent-0*.md` vào đúng thư mục mà `**/*.md` quét.

Không có `pytest-timeout`, không có `-p no:randomly` config, không có isolation marker; do đó suite không đạt chuẩn "deterministic gate" cho một báo cáo khoa học.

### 3.2 [T1-ruff-scope-and-fixability] | HIGH | VERIFIED

```text
$ python -m ruff check . --output-format concise | tail -3
Found 464 errors.
[*] 385 fixable with the `--fix` option
```

Phân bố theo thư mục đã đo: `scripts/` 33, `notebooks/` 386, `drive_upload_ready/` 228. Cấu hình `ruff.toml` chọn `E,F,W,I`, bỏ qua `E402,E501,W291,W293`, giới hạn 120 ký tự, target py310, và **per-file-ignore `tests/* = ["F401","F841"]`** (tắt kiểm tra import biến thừa ngay trong test, nơi sinh ra phần lớn F401/F841).

Vấn đề: tuyên bố "0 lint errors" chỉ đúng cho `src/`. Repo không có bước ruff nào trong CI (CI chỉ chạy flake8 với `--exit-zero` và `|| true`, xem 3.6), nên 464 lỗi tồn tại mà không có gate nào chặn. 385 lỗi trong số đó self-fixable, tức là nợ kỹ thuật thuần túy.

### 3.3 [T3-xác-nhận-đúng] | không phải defect (xem mục 5.1) - nhưng có 1 HIGH kèm theo

`src/training/dataset_loader.py:127-133` trả về danh sách bệnh nhân **cứng** khi manifest vắng mặt:

```python
    if not config_path.exists():
        # Fallback to local default patient lists if config file does not exist
        return {
            "train": ["TCGA-A1-A0SK", "TCGA-A2-A04N", "TCGA-A2-A0T2"],
            "val": ["TCGA-AN-A03X", "TCGA-AN-A0AL"],
            "test": ["TCGA-AQ-A04H", "TCGA-C8-A1HJ"],
        }
```

Manifest thật (`src/config/bcss_patient_splits.json`) có 106/22/23 bệnh nhân. Fallback này chỉ giữ 3/2/2, tức là nếu đường dẫn manifest sai trên Colab, training âm thầm chỉ dùng 3 bệnh nhân và mọi con số "151 cases / 106 patients" trong báo cáo mất nghĩa. Không raise, không warning.

Đối chiếu tích cực: `src/datasets/patient_split.py:103-154` (`load_patient_splits`) **đòi** manifest và không có fallback lặng lẽ - đây là phiên bản đúng; tồn tại song song như một bản sao thứ hai của cùng logic.

### 3.4 [T6-resume-optimizer-device] | HIGH | INFERENCE (không có GPU)

`src/training/train_proposed_sam.py:400-410`:

```python
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # nosec B614
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

`map_location="cpu"` đưa toàn bộ optimizer state (momentum, exp_avg, exp_avg_sq) về CPU; `optimizer.load_state_dict` không di chuyển lại về device của param. Trên T4/L4, resume từ checkpoint GPU sẽ đụng `RuntimeError: Expected all tensors to be on the same device...`. Tôi không có GPU nên không chạy được; đây là INFERENCE, nhưng nó khớp chính xác với bug-hunt bên ngoài ở mục 6.3.

Ngoài ra `strict=False` (dòng 405) che mất mọi khóa trọng số không khớp; tôi đo được 9 khóa bị bỏ qua ở đường surrogate.

`src/training/train_stage2.py:305-315` đi theo hướng an toàn hơn (chỉ khôi phục state_dict/epoch/mIoU rồi **dựng lại** optimizer + scheduler) nhưng lại mất Adam moment - một trade-off không được ghi chú ở đâu cả.

### 3.5 [T7-weights-only-blanket-fallback] | HIGH | VERIFIED (đếm) + INFERENCE (tác động)

```text
$ grep -rn "torch.load" src/ --include=*.py | wc -l
31
$ grep -rn "weights_only=False" src/ --include=*.py | wc -l
9
```

Cả 9 site đều theo mẫu `try: weights_only=True / except Exception: weights_only=False`, ví dụ `src/colab_pkg/model_loader.py:567-576` (`safe_torch_load`), `src/colab_pkg/downloader.py:230,232`, `src/training/train_proposed_sam.py:402-404`, `src/training/train_stage2.py:298,300,309,311`.

Hai hệ quả:
- `except Exception` quá rộng: PyTorch 2.6 đổi mặc định `weights_only`, nhưng cùng khối except cũng nuốt cả `FileNotFoundError`, `UnpicklingError: invalid load key, '\x1f'` (tệp tải về bị cắt giữa chừng - rất hay gặp với gdown/Colab), và `PermissionError`. Sau đó nó retry với unpickle **toàn quyền**, biến một tệp tải hỏng thành một checkpoint "hợp lệ" hoặc thành một vector phản serialization.
- Bản sao trùng lặp: `safe_torch_load` tồn tại y bản ở `src/models/wsss_inference_methods.py:360-365` và `:838-846` (INVARIANT-25/14 về chống trùng lặp).

### 3.6 [T8-ci-vacuous] | HIGH | VERIFIED

`.github/workflows/ci.yml` (56 dòng, ubuntu-latest, Python 3.10):

```yaml
      - name: Install Lint & Test Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install flake8 pytest Pillow

      - name: 1. Code Style & Linting Check
        run: |
          flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics || true
          flake8 src/ --count --exit-zero --max-complexity=15 --max-line-length=120 --statistics || true
...
      - name: 4. Run Unit Tests & Integration Tests
        run: |
          if [ -d "tests" ]; then
            pytest tests/ -v
          else
            echo "No tests directory found yet. Skipping pytest."
          fi
```

Bốn khiếm khuyết:
1. Cả hai bước lint đều không có khả năng làm đỏ build (`--exit-zero` và `|| true`).
2. CI **không cài torch/numpy/requirements**: mới chỉ `flake8 pytest Pillow`. Mọi test chạm `torch` (đa số suite 348 test) sẽ `ModuleNotFoundError`. Nghĩa là bước 4 gần như chắc chắn chưa từng xanh trên CI; không có `requirements.txt` được `pip install -r`, không có `torch` index.
3. Bước "Zero Mockup" chỉ grep năm chuỗi ký tự cố định. Chạy nguyên văn trên repo:

```text
$ grep -rnw 'src/' -e 'TODO: mock' -e 'fakeData' -e 'dummyData' -e 'fake_metric' -e 'simulateTx'; echo "exit=$?"
exit=1
$ grep -rc "SyntheticHistologyDataset" src/training/train_proposed_sam.py
4
$ grep -rc "SurrogateImageEncoder" src/models/sam_wsss_adapter.py
3
$ grep -rc "fallback_mask" src/training/dataset_loader.py
3
```

CI in "PASSED: Zero mockup verified" trong khi ba identifier mock thật xuất hiện 4/3/3 lần. Gate lệch hoàn toàn khỏi cơ chế thật (mục 2.1-2.3).

4. Không có bước kiểm tra patient-leakage, không có `KMP_DUPLICATE_LIB_OK`, không có ruff, không có ma trận notebook-vs-source drift.
5. `.gitignore` loại khỏi tầm CI: `:32 drive_upload_ready/` (bundle bàn giao), `:47 product/reports/SE374_MULTI_AGENT_AUDIT_AND_DEFECT_RESOLUTION_REPORT.md` (chính một báo cáo kiểm định), `:64 product/reports/*.log`. Báo cáo bị gitignore thì không được bảo vệ bởi CI.

### 3.7 [T9-bốn-bản-extract_patient_id] | HIGH | VERIFIED

```text
$ grep -rn "def extract_patient_id" src/ --include=*.py
src/colab_pkg/downloader.py:76:def extract_patient_id(filename: str) -> str:
src/colab_pkg/evaluator.py:58:        def extract_patient_id(fn: str) -> str:
src/datasets/patient_split.py:7:def extract_patient_id(filepath_or_name: str) -> str:
src/training/dataset_loader.py:51:def extract_patient_id(filepath_or_name: Union[str, Path]) -> str:
```

Bốn triển khai độc lập cho một quyết định khoa học (mỗi bệnh nhân là một đơn vị phân tách). So sánh hành vi thật:

```text
input | loader | split | evaluator
TCGA-A1-A0SK-01Z-00-DX1_X.png | TCGA-A1-A0SK | TCGA-A1-A0SK | TCGA-A1-A0SK
bcss_slide_001_tumor.png | BCSS | bcss | bcss
IMG_0001.png | IMG | IMG | IMG
tcga-a1-a0sk.png | TCGA-A1-A0SK | TCGA-A1-A0SK | TCGA-A1-A0SK
```

`src/training/dataset_loader.py:51-68` dùng `re.match` + `.upper()`; `src/datasets/patient_split.py:7-25` dùng `re.search` và **không** hoa hóa. Hệ quả đo được:

```text
$ python -c "[A('bcss_slide_%03d_tumor.png'%i) for i in range(3)]; len(set(...))"
['BCSS', 'BCSS', 'BCSS']
distinct count = 1
```

Với mọi kiểu tên không phải TCGA barcode, bản `dataset_loader` **nén toàn bộ slide thành một "bệnh nhân" duy nhất** (`.upper()` trên prefix chung). Khi đó `verify_patient_leakage` sẽ báo 0% rò rỉ một cách tầm thường nhưng sai: nó so khớp các khóa nhân tạo, không còn là bệnh nhân. Hai bản còn lại giữ `bcss` chữ thường và cho khóa đúng.

Trên dữ liệu BCSS hôm nay (100% TCGA barcode) bốn bản trùng nhau - đó là lý do kiểm định 2026-11-02 thấy "0.0%":

```text
train manifest= 106 | disk TCGA ids(from training/) = 106 | diff = [] | non-TCGA ids = ['1','Precision','Recall','mIoU','mPA']
val   manifest=  22 | disk TCGA ids = 22 | diff = []
test  manifest=  23 | disk TCGA ids = 23 | diff = []
pairwise overlaps train-val 0  train-test 0  val-test 0
```

Nhưng bất kỳ đổi tên file/chuyển dataset nào (LUAD-HistoSeg, GDPH - đều được nêu trong tài liệu dự án) cũng kích hoạt phân kỳ. Đây là nợ kiến trúc, không phải lỗi số học hiện tại.

---

## 4. DEFECT MEDIUM

### 4.1 [T8-invariant12-dot-blindspot] | MEDIUM | VERIFIED

`tests/test_governance_invariants.py:17-31`:

```python
def _scan_files_for_bad_dashes(patterns):
    target_files = []
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            if not f.startswith(IGNORED_DASH_PREFIXES):
                target_files.append(f)
```

`glob` với `**` bỏ qua thành phần đường dẫn bắt đầu bằng `.`. Đo trực tiếp:

```text
test-visible files: 5364
files on disk (incl dot-dirs): 6095 | dot-dir files invisible to test: 730
confirmed NOT matched by test glob: 730
bad-dash hits inside dot-dirs: 444
```

Lọc bỏ `repos/` (bên thứ ba), số vi phạm **thuộc tài sản dự án mà test không bao giờ thấy**:

```text
PROJECT-OWNED dot-dir violations missed by test: 48
  .agents\rules\13_deliverable_quality_and_scientific_visuals_gate.md:32 U+2014,U+2013 | - Quet dau gach noi: Chan em-dash [<U+2014>] (U+2014) va en-dash [<U+2013>] (U+2013)
  .agents\teamwork\audit_reviewer_parity_invariant12\DISPATCH.md:21 U+2014,U+2013,U+2015,U+2212 | Inspect markdown files, python files, notebooks...
  .agents\teamwork\checker_team_a\audit_report.md:44 U+2014,U+2013,U+2015 | # Forbidden dashes: em-dash ([<U+2014>] \u2014), en-dash ([<U+2013>] \u2013), horizontal
  ... (tổng 48 lượt chạm, gồm U+2014, U+2013, U+2015, U+2212)
```

Nghịch lý: chính tệp `.md` **định nghĩa quy tắc INVARIANT-12** chứa cả bốn ký tự bị cấm.

Điểm công bằng: các vi phạm này là "code sample mô tả ký tự cấm" chứ không phải văn xuôi luận án. Vì vậy severity là MEDIUM, không CRITICAL. Tài liệu bàn giao (`product/`, `src/`, `notebooks/`, `docs/`) sạch 100% (mục 5.3).

Hai lỗi cấu trúc còn lại của test:
- `tests/test_governance_invariants.py:46-52` chỉ làm **kiểm tra chuỗi** trên `AGENTS.md` (assert sự tồn tại của nhãn quy tắc dạng mở ngoặc vuông + `INVARIANT-24]`), không chứng minh hành vi của INVARIANT-24/25.
- `IGNORED_DASH_PREFIXES` chỉ khớp khi đường dẫn là tương đối. Test phụ thuộc CWD của pytest; chạy từ thư mục khác, mọi prefix ignore đều vô hiệu và suite nổ 444+ lỗi giả.

### 4.2 [T4-shallow-glob] | MEDIUM | VERIFIED (đếm) + INFERENCE (hệ quả)

INVARIANT-29 cấm glob nông. Số vi phạm `.png`-shallow trong `src/`:

```text
src/datasets/bcss_dataset.py:54-55   pattern = os.path.join(train_dir, "*.png"); glob.glob(pattern)
src/datasets/bcss_dataset.py:71-72   pattern = os.path.join(img_dir, "*.png");  glob.glob(pattern)
src/clinical/exhaustive_evaluator.py:394
src/colab_pkg/evaluator.py:1078-1081, 1085
src/colab_pkg/downloader.py:134, 354-355, 543-544, 632, 658, 703
```

`src/datasets/bcss_dataset.py` là đáng kể nhất: `train_dir/*.png` và `val/img/*.png` đều không đệ quy, và `if not root_path.exists()` thì trả `[]` (hoặc để `self.image_paths` rỗng) mà không raise.

Đây là mặt trái của mục 2.4: hoặc glob quá nông (bỏ sót con, sinh dataset rỗng), hoặc rglob quá sâu (nuốt cả `pred/`). Repo không có một helper khám phá dữ liệu **có kiểm soát lớp thư mục** nào; `find_image_files` không loại `pred`, `pred_palette`, `mask`, `miou_out`.

Bản chất hai mặt của cùng một lỗi là lý do cả hai nửa vi phạm cùng tồn tại: `find_image_files` tồn tại (dòng 154-172) nhưng `src/datasets/bcss_dataset.py` dùng `glob.glob` nông riêng, nghĩa là INVARIANT-29 "đã khai báo, chưa được thực thi".

### 4.3 [T5-notebook-drift] | MEDIUM | VERIFIED

Bảng md5 giữa nguồn và bundle bàn giao (`scripts/build_drive_package.py` là công cụ đồng bộ hóa):

| notebook | notebooks/ md5 | drive_upload_ready md5 | kết quả |
|---|---|---|---|
| SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb | 266f6763d5a0 | 045fa19fd2a9 | DRIFT |
| SE374_Colab_WSSS_A100_Full4986_ThinDriveRunner_v2.ipynb | c127184ed9d3 | ab370de9b4eb | DRIFT |
| SE374_Colab_WSSS_L4_Full4986_ThinDriveRunner_v2.ipynb | 70533ff0bc68 | b16c971b8c1e | DRIFT |
| SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb | 35b09c4b3f66 | f39ae74384ec | DRIFT |
| SE374_Colab_WSSS_T4_Sanity150_ThinDriveRunner_v2.ipynb | 2250fea4059d | a23f17c7a55b | DRIFT |

```text
paired=5 drift=5
```

100% các cặp notebook bàn giao bị lệch so với nguồn. INVARIANT-28 (chống drift notebook) bị vi phạm trên toàn bộ giao diện chạy Colab; người hướng dẫn sẽ chạy một bản khác bản đã audit. Không có test nào so md5/hash giữa `notebooks/` và `drive_upload_ready/`.

### 4.4 [T5-lifecycle-inconsistency] | MEDIUM | VERIFIED

Ma trận 7-bước vòng đời Colab, quét tự động trên 9 notebook `SE374_Colab_WSSS_*.ipynb`:

| notebook | drive.mount | torch.cuda | files.download | gdown | nvidia-smi | np.zeros mask | số cells |
|---|---|---|---|---|---|---|---|
| SE374_Colab_WSSS_A100_Full4986.ipynb | - | Y | Y | Y | Y | Y | 19 |
| SE374_Colab_WSSS_A100_Full4986_ThinDriveRunner_v2.ipynb | Y | Y | - | - | - | - | 17 |
| SE374_Colab_WSSS_L4_Full4986.ipynb | - | Y | Y | Y | Y | Y | 19 |
| SE374_Colab_WSSS_L4_Full4986_ThinDriveRunner_v2.ipynb | Y | Y | - | - | - | - | 17 |
| SE374_Colab_WSSS_Pipeline.ipynb | - | Y | Y | Y | Y | Y | 19 |
| SE374_Colab_WSSS_T4_Full4986.ipynb | - | Y | Y | Y | Y | Y | 19 |
| SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb | Y | Y | - | - | - | - | 17 |
| SE374_Colab_WSSS_T4_Sanity150.ipynb | - | Y | Y | Y | Y | Y | 19 |
| SE374_Colab_WSSS_T4_Sanity150_ThinDriveRunner_v2.ipynb | Y | Y | - | - | - | - | 17 |

Nhận xét:
- Tồn tại **hai họ protocol ngược nhau**: họ `Full4986` dùng `gdown` + `files.download` và **không** `drive.mount`; họ `ThinDriveRunner_v2` dùng `drive.mount` và **không** gdown/nvidia-smi. Không có tài liệu nào nói họ nào là chân lý; cả hai đều được bàn giao.
- Bước "xác nhận device" (`torch.cuda`) có mặt ở cả 9 -> tốt.
- Bước `nvidia-smi` (bằng chứng loại GPU thật cho hội thẩm) vắng mặt ở 4/9 notebook, trong đó có cả bản ThinDriveRunner chính.
- Cột `np.zeros mask` = Y ở 5 notebook họ cũ: trực tiếp nối với cơ chế ở mục 2.2.
- Không notebook nào đặt `KMP_DUPLICATE_LIB_OK`; điều này chỉ ảnh hưởng Windows local, không ảnh hưởng Colab, nhưng làm mất khả năng tái lập local.

### 4.5 [T7-cohort-truncation] | MEDIUM | VERIFIED (đọc mã, có dòng) + INFERENCE (mức độ)

`src/colab_pkg/evaluator.py` có ba tầng cắt cohort mà không ghi "đã cắt bao nhiêu":

```python
        cap = max(len(val_files), 50)
        ...
            val_files[:] = sorted(real_v_img.glob("*.png"))[:cap]
            test_files[:] = sorted(real_t_img.glob("*.png"))[:cap]
```

cộng thêm hai `[:50]` cứng quanh dòng 1240 và 1245, và quanh dòng 1269-1271:

```python
            if not m_path.is_file():
                continue
```

Với cohort test thật 4,986 ảnh, một `[:50]` âm thầm giảm mẫu 99 lần; mỗi ca bị `continue` bỏ khỏi thống kê lâm sàng (ICC/CCC/Bland-Altman/log-rank) mà không có Missing Data Report nào. Đây là điểm mạnh nhất về tính toàn vẹn thống kê sau mục 2.5.

### 4.6 [T10-test-tự-chứng-manifest] | MEDIUM | VERIFIED

`tests/test_patient_split.py:94-145`: test đọc manifest rồi **khẳng định lại chính manifest** (đọc `load_patient_splits`, so với `json.load` cùng tệp) và chấp nhận hai đáp án khác nhau bằng `in (...)`:

```python
    assert len(splits["train"]) in (91, 106)
    assert len(splits["val"]) in (30, 22)
    assert len(splits["test"]) in (30, 23)
```

Một test không thể thất bại với bất kỳ manifest nào có hai bộ số đều được, không chứng minh được gì ngoài "đọc file thành công". Đây là ví dụ textbook của Mock-Testing Illusion: pytest xanh nhưng không có hành vi được kiểm chứng.

Phần tốt của tệp này (để công bằng): dòng 32-49 dùng `sample_dataset` thật, dòng 52-60 chủ động bơm overlap rồi `pytest.raises(ValueError)`, dòng 162-175 fuzz bằng `hypothesis`. Bốn test đó là test thật.

### 4.7 [T6-unbound-local-suspect] | MEDIUM | INFERENCE

`src/datasets/bcss_dataset.py:132-133`:

```python
        else:
            mask_tensor = torch.zeros((img_np.shape[0], img_np.shape[1]), dtype=torch.int64)
```

`img_np` được gán trong nhánh đọc mask phía trên; nếu `self.transform` là transform trả về Tensor (cách dùng phổ biến) và mask thiếu, `img_np` chưa từng được binds -> `UnboundLocalError`. Tôi suy luận từ đọc luồng nhánh, **chưa chạy được** vì đường này cần `data/BCSS-WSSS` + transform thật (tôi không tạo script trong `tests/`/`scripts/` theo ràng buộc). Ghi rõ INFERENCE.

Cùng locus: `BCSSDataset.__len__` không có lưới an toàn khi `self.image_paths == []` (xem mục 5.5 cho chứng minh lỗi `num_samples=0`).

### 4.8 [T8-deliverable-auditor-fence-blind] | MEDIUM | VERIFIED

`src/utils/deliverable_auditor.py` là gate của INVARIANT-24/25, nhưng nó **không bỏ qua fenced code block**, nên mọi báo cáo có bằng chứng lệnh đều bị phạt oan. Chứng minh bằng probe tối thiểu (tệp tạm ngoài repo, không tạo tệp mới trong `src/`, `tests/`, `scripts/`):

```text
noi dung probe.md:
  dong 5-8 : fence ```python, ben trong co dong comment bat dau bang dau `#`
  dong 10  : heading `### Sub heading here`
  dong 12-14: fence ```text, ben trong co dong `pytest tests/ -q`

ket qua chay `python -m src.utils.deliverable_auditor probe.md`:
1. [WARNING] [INTEGRITY] probe.md:7  - Cong thuc toan LaTeX ($) khong dong mo can bang.
2. [WARNING] [INTEGRITY] probe.md:13 - Cong thuc toan LaTeX ($) khong dong mo can bang.
3. [WARNING] [STRUCTURE] probe.md:10 - Cau truc tieu de nhay cap bat hop ly (tu H1 len H3).
```

Cơ chế:
- Dòng 13 nằm **trong** fence ```text nhưng vẫn bị đếm `$` của dấu nhắc shell -> cảnh báo LaTeX giả.
- Dòng 6 (`# this is a python comment` trong fence ```python) bị hiểu là **H1**, nên `###` ở dòng 10 bị kết luận "nhảy bậc H1 -> H3" -> cảnh báo cấu trúc giả.
- Chính tôi gặp lại đúng hai loại giả dương này trên báo cáo này: 32 cảnh báo LaTeX và 4 cảnh báo nhảy bậc, tất cả đều bắt nguồn từ dấu `$` của lệnh và comment `#` trong khối mã.

Hệ quả thật: gate chất lượng bàn giao có tỷ lệ nhiễu cao tới mức người viết bị khuyến khích **xóa bằng chứng lệnh khỏi báo cáo** để cho xanh. Điều đó xung đột trực tiếp với yêu cầu "bắt buộc kèm output thật" của cùng bộ quy tắc. Kiến nghị: parser của auditor cần cắt nội dung giữa ```` ``` ```` trước khi chạy cả hai kiểm tra `dollar_count` và heading-level; đồng thời `unescaped` hiện không loại được `\$` (probe dòng 7 vẫn bị gọi dù đã escape).

Ghi chú minh bạch: báo cáo này **giữ nguyên** các dòng lệnh bắt đầu bằng `$ ` vì đó là bằng chứng hành vi; vì vậy nó sẽ in ra ~36 WARNING giả dương từ lỗi vừa nêu, và tôi từ chối sửa bằng chứng để chiều theo công cụ.

---


### 5.1 [T9-dead-flag-và-nhánh-chết] | LOW | VERIFIED
- `self.is_surrogate`: 4 lần gán, 0 lần đọc (chứng minh ở 2.3). Cờ trung thực không có tác dụng.
- `src/models/sam_wsss_adapter.py:300-325`: `except Exception: pass` trong point-mining với nhánh fallback `argmax` suy biến - khi khai thác điểm thất bại, model nhận prompt suy biến mà không có tín hiệu.
- Ba bản sao `safe_torch_load` và `extract_patient_id` (xem 3.5, 3.7).

### 5.2 [T7-except-count] | LOW | VERIFIED
```text
src files with except Exception: 16 | total: 49
  8 src\colab_pkg\downloader.py
  8 src\utils\deliverable_auditor.py
  5 src\pipeline\wsi_lowmag_highmag_pipeline.py
  4 src\clinical\exhaustive_evaluator.py
  4 src\colab_pkg\evaluator.py
  3 src\clinical\survival_analysis.py
  3 src\colab_pkg\model_loader.py
  3 src\visualization\api_server.py
  2 src\colab_pkg\visualizer.py
  2 src\models\sam_wsss_adapter.py
```
49 khối `except Exception` trong 16 tệp `src/`. Phần lớn là "nuốt rồi đi tiếp" (`pass`, trả `None`, trả `{}`). Tiêu biểu nguy hiểm: `src/colab_pkg/evaluator.py:172-182` bọc `compute_fwiou` và `compute_boundary_f1` mỗi patch bằng `except Exception: pass` -> FwIoU/BF1 trung bình trên **số patch ít hơn** mẫu, không ghi chú. Các khối còn lại (downloader, deliverable_auditor) rủi ro thấp hơn vì chúng là code I/O phòng thủ.

### 5.3 [T5-root-notebook-hygiene] | LOW | VERIFIED
10 tệp `.ipynb` nằm ở gốc repo và 5 tệp `.ipynb` nằm trong `drive_upload_ready/SE374_WSSS_Drive_Code/` gốc, song song với bản trong `notebooks/`. Nhân ba bản thể của cùng một protocol chạy được, không có cơ chế chọn bản nào là chuẩn (liên quan trực tiếp tới drift ở 4.3).

### 5.4 [T1-warnings-swallowed] | LOW | VERIFIED
Hai warning còn lại của pytest:
```text
tests/test_clinical_service.py::test_exhaustive_benchmark_evaluator_dry_run_execution
  ...\scipy\stats\_axis_nan_policy.py:423: RuntimeWarning: Precision loss occurred in moment
  calculation due to catastrophic cancellation. This occurs when the data are nearly identical.

tests/test_colab_notebook.py::test_stage1_netcam_architecture_and_forward_cam
  ...\repos\WSSS-Tissue\tool\infer_utils.py:6: DeprecationWarning: scipy.misc is deprecated
```
Cảnh báo thứ nhất nói thẳng: kiểm định thống kê đang chạy trên dữ liệu "nearly identical" -> đúng bối cảnh bảng §6.1 có hai hàng giống hệt nhau (2.6). Không có `filterwarnings = error` trong config nên suite vẫn xanh. Cảnh báo thứ hai cho thấy test production **import từ `repos/`** (thư mục vendored của bên thứ ba), tức là một test của dự án phụ thuộc code không thuộc dự án.

### 5.5 [T4-empty-dataset-safety-net] | LOW | VERIFIED
`src/training/dataset_loader.py:570` có lưới an toàn đúng (`drop_last=True if len(dataset) > batch_size else False`), nhưng `src/datasets/bcss_dataset.py` (Stage-1 path được `train_proposed_sam.py` dùng) **không** có: `self.image_paths` rỗng -> `len(dataset) == 0` -> DataLoader nổ. Chứng minh hành vi (đã chạy):

```text
$ KMP_DUPLICATE_LIB_OK=TRUE python -c "BCSSDataset(root_dir='data/_missing', split='train') rồi iter DataLoader"
len(dataset) = 0
next(iter(loader)) -> StopIteration / ValueError: num_samples should be a positive integer value, but got num_samples=0
```
Lỗi này được báo cáo bên ngoài rất nhiều; xem mục 6.1.

---

## 6. BẰNG CHỨNG SẠCH (Chiều đạt - kèm lệnh và output)

Theo yêu cầu "Hạng mục SẠCH vẫn phải ghi: bạn đã chạy lệnh gì và output ra sao".

### 6.1 [T3] Chống rò rỉ bệnh nhân trên dữ liệu thật - ĐẠT
```text
$ KMP_DUPLICATE_LIB_OK=TRUE python -c "<load src/config/bcss_patient_splits.json + quét disk>"
manifests: ['src\\config\\bcss_patient_splits.json']
sizes 106 22 23
pairwise overlaps train-val 0 train-test 0 val-test 0
train manifest= 106 | disk TCGA ids = 106 | diff = []
val   manifest=  22 | disk TCGA ids = 22 | diff = []
test  manifest=  23 | disk TCGA ids = 23 | diff = []
verify_patient_leakage -> (không raise)
```
- 151 bệnh nhân = 106+22+23, khớp tuyệt đối giữa manifest và đĩa. Không một TCGA ID nào xuất hiện ở hai split. Tuyên bố "0.0% Strict Leakage (151 Cases)" của đợt trước là **ĐÚNG** trên cấu hình hiện hành.
- `src/datasets/patient_split.py:29-73` `verify_patient_leakage` thực sự `raise ValueError("Patient leakage detected!")` và `tests/test_patient_split.py:52-60` chứng minh bằng cách bơm overlap: `VERIFIED`.
- `src/datasets/patient_split.py:76-100` `split_patients_kfold` cho các fold rời nhau và tất định: `VERIFIED` qua test 162-175 (hypothesis fuzz).

### 6.2 [T-DONE] Toán metric đúng chuẩn - ĐẠT
Đọc toàn bộ `src/losses_metrics/metrics.py` (588 dòng):
- mIoU/mDice dùng `valid_classes = denominator > 0` và trung bình trên lớp có mặt -> định nghĩa chuẩn LowMagWSS/WSSS-Tissue.
- FwIoU (dòng 106-119) dùng trọng số tần suất trên `class_pixel_counts > 0`; Pixel Accuracy (121-127) và cả hai đều có guard `total_pixels == 0`.
- Boundary F1 (dòng 268+) hỗ trợ `ignore_index=255`.
- ICC(2,1)/CCC/Bland-Altman (dòng 356-399) được triển khai tường minh, không gọi thư viện mờ.
- Kiểm tra tiền lệ tử mà tôi nghi ngờ: dòng 444/456 `vtp_r, vtp_r_p = 1.0 if np.allclose(...) else 0.0, 1.0` - đã xác nhận bằng `ast` rằng đây là tuple assignment hợp lệ với precedence đúng, **không phải bug** (rút lại nghi vấn đầu phiên).
- NHƯNG: chính `valid_classes` là chỗ khiến 2.2 xảy ra. Toán đúng, giả định "cohort có đủ lớp" không được kiểm tra.

### 6.3 [T2] Từ chối bịa dữ liệu lâm sàng - ĐẠT
`src/clinical/exhaustive_evaluator.py:661, 699, 719, 723` có ghi chú và logic **từ chối** chèn survival months giả ("no dummy patients injected"). `VERIFIED` bằng đọc dòng + test `test_clinical_service.py::test_exhaustive_benchmark_evaluator_dry_run_execution` chạy thật (xanh).

### 6.4 [T2] Mock bị gating mặc định - ĐẠT
```text
src/colab_pkg/downloader.py:611   allow_mock: bool = False
src/colab_pkg/downloader.py:647-648  (nhánh mock nằm sau `if allow_mock:`)
                        else: raise RuntimeError(...)
```
`downloader.py` là module gương mẫu: mock phải bật tường minh, không có thì raise. Đối chiếu ngược lại với 2.1-2.3 cho thấy chuẩn mực đã tồn tại trong repo nhưng không được áp dụng đồng nhất.

### 6.5 [T8] Typography deliverable - ĐẠT
```text
$ python -c "<scan '**/*.md','**/*.py','**/*.ipynb' với IGNORED_DASH_PREFIXES của test>"
total hits after test ignore list 0
```
Toàn bộ `product/`, `src/`, `notebooks/`, `docs/`, `tests/` (các tệp test nhìn thấy) không có U+2014/U+2013/U+2012/U+2015/U+2212. Báo cáo 2026-11-02 đúng cho tập này. (Phần dot-dir ở 4.1 là lỗ hổng **của test**, không phải của deliverable.)
- Lệnh phụ: `python -m src.utils.deliverable_auditor product/proposal/SE374_MINI_PROPOSAL_WSSS_WSI.md` -> `[DELIVERABLE QUALITY AUDIT: PASSED]`, exit 0. `VERIFIED`.
- `src/utils/deliverable_auditor.py:452-462` có quét dummy-marker thật sự (không chỉ grep chuỗi), và `:630-647` xử lý `sys.argv` target đúng.

### 6.6 [T6] SAM prompt tensor device transfer - ĐẠT (tĩnh)
`src/models/sam_wsss_adapter.py:252-261` đặt `device=device` tường minh cho `pts` và `lbl` khi dựng point prompt; `torch.stack` phía sau thừa kế device. Đường prompt không phải nguồn lỗi device. `INFERENCE` vì không GPU.

### 6.7 [T1] src/ sạch ruff - ĐẠT
```text
$ python -m ruff check src/ --output-format concise
All checks passed!
```
0 lỗi trong `src/` (42 module). Khớp với tuyên bố cũ.

---

## 7. T10 - SĂN LỖI BÊN NGOÀI (3 chữ ký lỗi, fix được chấp nhận, kèm URL)

Yêu cầu: lỗi thực, đã có người xác nhận, đường dẫn và số dòng bị scrub, tên lớp giữ nguyên.

### 7.1 Chữ ký: `ValueError: num_samples should be a positive integer value, but got num_samples=0`

Nguồn tham chiếu:
- StackOverflow: `https://stackoverflow.com/questions/71615089/valueerror-num-samples-should-be-a-positive-integer-value-but-got-num-samples`
- StackOverflow: `https://stackoverflow.com/questions/70369070/can-a-pytorch-dataloader-start-with-an-empty-dataset` - câu được chấp nhận: "the `RandomSampler` class checks that `num_samples` is positive when it is initialised".
- GitHub issue (accepted): `https://github.com/sanghyun-son/EDSR-PyTorch/issues/185` - câu trả lời được chốt: "the meaning of the error message is that the dataset has a size of 0. You can check that with `print(len(dataset))`" kèm nguyên nhân subset lớn hơn dataset.
- Đồng mắc: `https://github.com/lucidrains/recurrent-interface-network-pytorch/issues/21`, `https://github.com/opendilab/InterFuser/issues/20`, `https://github.com/kohya-ss/musubi-tuner/issues/43`.

Fix được chấp nhận: **assert `len(dataset) > 0` trước khi dựng DataLoader** và để lỗi raise to rền ở bước khám phá dữ liệu, không phải ở vòng lặp.

Ánh xạ vào repo (INVARIANT-29 + T7): `src/datasets/bcss_dataset.py` để `self.image_paths = []` lọt qua (`:54-55`, `:71-72`), `src/training/dataset_loader.py:154-172` `find_image_files` trả `[]` khi thư mục không tồn tại mà không raise. Chỉ `create_stage2_dataloader` (`:570`) có lưới. Tôi đã tái hiện crash:

```text
$ KMP_DUPLICATE_LIB_OK=TRUE python -c "BCSSDataset(root_dir='data/_missing', split='train') + DataLoader(...) iter"
len(dataset) = 0
ValueError: num_samples should be a positive integer value, but got num_samples=0
```

Khuyến nghị (không sửa mã): đặt `assert len(self.image_paths) > 0, f"0 patches discovered under {root_dir}"` ngay cuối `_load_dataset`, và đồng nhất mọi đường khám phá qua một helper có allowlist subdirectory (`img`, `mask`, `training`) để chặn cả hai phía của lỗi (quá nông / quá sâu).

### 7.2 Chữ ký: `_pickle.UnpicklingError: Weights only load failed...` / `invalid load key, '\x1f'`

Nguồn tham chiếu:
- StackOverflow: `https://stackoverflow.com/questions/79584485/unable-to-torch-load-due-to-pickling-safety-error`
- StackOverflow: `https://stackoverflow.com/questions/79885063/gluonts-deeparestimator-fails-to-load-checkpoint-in-pytorch-2-6`
- StackOverflow: `https://stackoverflow.com/questions/79264683/error-loading-pytorch-model-checkpoint-pickle-unpicklingerror-invalid-load-ke` - câu được chấp nhận chỉ rõ `invalid load key, '\x1f'` = **tệp bị cắt/ tải hỏng**, không phải vấn đề weights_only.

Fix được chấp nhận (từ chính tài liệu PyTorch và các answer điểm cao): đặt `weights_only=True` tường minh, và nếu checkpoint có object tùy ý thì **khai báo `allowed_classes`** thay vì tắt cơ chế; đồng thời phân biệt `UnpicklingError` (do weights_only) với `FileNotFoundError`/EOF (do tệp hỏng).

Ánh xạ vào repo: 31 site `torch.load`, 9 site dùng mẫu blanket:
```python
try:
    obj = torch.load(path, map_location="cpu", weights_only=True)
except Exception:
    obj = torch.load(path, map_location="cpu", weights_only=False)
```
Vị trí: `src/colab_pkg/model_loader.py:567-576` (`safe_torch_load`, trùng bản sao tại `src/models/wsss_inference_methods.py:360-365` và `:838-846`), `src/colab_pkg/downloader.py:230,232`, `src/training/train_proposed_sam.py:402-404`, `src/training/train_stage2.py:298,300,309,311`.

Tác động: đúng theo answer ở URL thứ ba, `except Exception` gộp "tệp tải hỏng" vào "cần weights_only=False" -> hệ thống âm thầm unpickle toàn quyền một file gdown tải dở. Trên Colab (nơi mọi checkpoint đều đi qua gdown) xác suất tệp truncated là đáng kể. Đây là vừa rủi ro bảo mật vừa rủi ro khoa học (checkpoint nửa vời được nhận là hợp lệ).

### 7.3 Chữ ký: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`

Nguồn tham chiếu:
- StackOverflow: `https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least` - người hỏi: "I saved a checkpoint while training on gpu. After reloading the checkpoint and continue training I get the following error".
- StackOverflow: `https://stackoverflow.com/questions/77061667/resume-from-checkpoint-gives-device-error-in-huggingface-transformers-trainer`
- GitHub: `https://github.com/huggingface/peft/issues/2129`, `https://github.com/OpenGVLab/InternVL/issues/688`, `https://github.com/ultralytics/ultralytics/issues/16050`.

Fix được chấp nhận (lặp lại trong mọi thread): sau `optimizer.load_state_dict(...)` phải **duyệt `state` và `.to(param.device)`**, hoặc load trực tiếp `map_location=device`; hoặc gọi `model.to(device); optimizer = build_optimizer(...)` rồi mới load.

Ánh xạ vào repo: `src/training/train_proposed_sam.py:402` (`map_location="cpu"`) + `:408` (`optimizer.load_state_dict` không remap) và `:405` (`load_state_dict(..., strict=False)` che luôn mismatch khóa). Đường resume T4/L4 vì thế tiềm năng nổ ở epoch đầu sau resume, đúng kịch bản Colab bị ngắt giữa chừng. `INFERENCE` (tôi không có GPU để tái hiện).

Giải pháp thay thế đã tồn tại trong repo và nên thống nhất: `src/training/train_stage2.py:305-315` dựng lại optimizer thay vì load, tránh lỗi device nhưng đổi lại mất moment Adam - một đánh đổi phải ghi tường minh trong phương pháp luận.

---

## 8. T9 - TRÙNG LẶP VÀ DEAD CODE (đo đếm)

| Hạng | Số lượng | Bằng chứng |
|---|---|---|
| `def extract_patient_id` trong `src/` | 4 | grep ở 3.7 |
| `def safe_torch_load` (nguyên văn) | 3 site hành vi + 2 bản sao hàm | `model_loader.py:567-576`, `wsss_inference_methods.py:360-365`, `:838-846` |
| `except Exception` trong `src/` | 49 trên 16 tệp | đếm ở 5.2 |
| Site `torch.load` | 31 | grep ở 3.5 |
| Cờ ghi-không-đọc | `self.is_surrogate` (4 ghi, 0 đọc) | grep ở 2.3 |
| Manifest fallback song song | 2 (`dataset_loader.py:127-133` có fallback, `patient_split.py:103-154` không) | đọc trực tiếp |
| Shallow-glob tự chế | >= 10 vị trí | 4.2 |

Kết luận T9: dự án vi phạm INVARIANT-14/25 (chống trùng lặp) ở đúng lớp code **quyết định tính trung thực của kết quả** (nhận diện bệnh nhân, nạp trọng số, khám phá dữ liệu), chứ không phải ở code tiện ích. Trùng lặp ở đây nguy hiểm hơn bình thường: hai bản `extract_patient_id` **hành vi khác nhau** (mục 3.7), nên một thay đổi ở bản này không bảo vệ bản kia.

---

## 9. TÁI KIỂM ĐỊNH INVARIANT (phán quyết từng điều)

| INVARIANT | Phán quyết | Căn cứ |
|---|---|---|
| 07 Zero-Mockup | **VI PHẠM** | 2.1, 2.2, 2.3 (mặc định tạo dữ liệu/mô hình giả, không raise) |
| 08 Patient-level anti-leakage | **ĐẠT HÔM NAY, TRÚNG SỐ SẮP HỎNG** | 6.1 (106/22/23, overlap 0) + 2.4, 2.5, 3.7 |
| 09 Multi-seed reproducibility | **KHÔNG KIỂM CHỨNG ĐƯỢC** | mục 2.6: tệp seed/std được viện dẫn không tồn tại |
| 12 Typography ASCII hyphen | **ĐẠT cho deliverable, TEST MÙ dot-dir** | 6.5 + 4.1 |
| 14/25 Chống trùng lặp / Auditor gate | **VI PHẠM NHẸ** | mục 8 |
| 24 Scientific visuals | **ĐẠT (tĩnh)** | `deliverable_auditor.py:452-462` quét dummy marker thật; CLI exit 0 |
| 27/28 Colab lifecycle + notebook drift | **VI PHẠM** | 4.3 (5/5 DRIFT), 4.4 (hai protocol ngược nhau) |
| 29 Anti-shallow-glob | **VI PHẠM** | 4.2, và mặt trái 2.4 |
| 32 Output hygiene | **ĐẠT** | `deliverable_auditor` + test governance |

---

## 10. GIỚI HẠN CỦA KIỂM ĐỊNH NÀY

1. Không có GPU: toàn bộ mục 3.4, 6.6, 7.3 là `INFERENCE`. Mọi kết luận về AMP/GradScaler, autocast (`src/colab_pkg/model_loader.py:917-925`), `pin_memory` và device-mismatch chỉ dựa trên đọc mã.
2. Không chạy notebook: drift được đo bằng md5 và marker tĩnh (4.3, 4.4), không bằng execution.
3. Ràng buộc Builder-Auditor Separation được giữ tuyệt đối: tôi chỉ đọc file và chạy lệnh read-only; tệp duy nhất tôi ghi chính là báo cáo này. Vì thế các sửa chữa ở mục 7 chỉ là khuyến nghị, không phải patch.
4. Ba lệnh `WebSearch` đầu tiên bị từ chối vì truy vấn >100 ký tự; hai lệnh `WebFetch` lên StackOverflow trả về 403, tôi chuyển sang snippet tìm kiếm và `gh api` để đọc nội dung được chấp nhận (URL ở mục 7).
5. Flakiness (3.1) tôi chứng minh được **hiện tượng** (4 lần chạy, 3 kết quả) nhưng nguyên nhân chính xác là race nào thì chưa kết luận được; hai ứng viên có chỉ điểm đã nêu, kèm lý do loại trừ một phần (`test_thin_drive_runner.py` + `test_governance_invariants.py` chạy cặp thì xanh).
6. Trong quá trình audit, công cụ `Read`/`Grep` trả về văn bản bị biến dạng (đồng âm hóa tên định danh) ở một số lần. Tôi đã tái kiểm **từng dòng trích dẫn** bằng `sed -n` và `ast` qua Bash; nhờ vậy hai nghi vấn (`FORBIDDEN_DASHES` ở `tests/test_thin_drive_runner.py:55-60` và một dòng `os.name == "nt"` ở `dataset_loader.py`) được xác định là **artifact hiển thị, không phải lỗi mã** và đã rút lại trước khi viết báo cáo. Toàn bộ mã nguồn trích trong báo cáo này là output `sed -n` nguyên văn.

---

## 11. KHUYẾN NGHỊ THỨ TỰ ƯU TIÊN (không kèm patch, theo ràng buộc read-only)

1. Chặn ngay lớp "điểm tuyệt đối": mọi cohort đánh giá phải `assert` có ít nhất 2 lớp trong target và `assert mIoU < 100.0`; đổi `fallback_mask=True` thành `raise`; xóa nhánh `np.zeros` ở `dataset_loader.py:456-458`.
2. Đổi `use_surrogate_if_missing` mặc định thành `False`; bắt `is_surrogate` được đọc bởi evaluator và in thành cờ trong mọi bảng kết quả; fail-hard nếu `checkpoint_path is None`.
3. Sửa `evaluator.py:1085-1100`: pseudo-val bắt buộc lấy từ `val/`, không bao giờ từ `test/`; in WARNING và raise khi self-heal kích hoạt.
4. Dọn `data/BCSS-WSSS/test` khỏi `pred/`, `pred_palette/`, `miou_out/`, `1.png`; thay mọi khám phá dữ liệu bằng **một** helper có allowlist subdirectory, bỏ hết `glob.glob("*.png")` nông.
5. Bổ sung provenance thật cho §6.1 (tạo lại `experiments/benchmarks/benchmark_results.json` từ run thật, có `seed`, `mean`, `std`), hoặc gỡ bảng ra khỏi báo cáo.
6. CI: bỏ `--exit-zero`/`|| true`, cài `requirements.txt` (torch/numpy), thêm bước `ruff check .`, thêm bước kiểm tra leakage, thêm bước md5-diff `notebooks/` vs `drive_upload_ready/`, thay grep 5 chuỗi bằng scan AST cho các identifier mock thật.
7. Gom `extract_patient_id` và `safe_torch_load` về một implementation duy nhất; sửa `except Exception` -> `except (UnpicklingError, ValueError)` có chủ đích.
8. `test_governance_invariants.py`: chuyển từ `glob` sang `os.walk` (để thấy dot-dir) hoặc mở rộng patterns với `.*`; và `test_patient_split.py:94-145`: bỏ `in (...)`, chỉ chấp nhận manifest chân lý duy nhất.
9. Xóa `shutil.rmtree` ở `scripts/build_drive_package.py:1545` khỏi đường test, hoặc cho phép test chạy trên tmpdir; tách hẳn việc build bundle khỏi unit test.

---

*Hết báo cáo Tác tử 04. Mọi số liệu ở trên là output thật của lệnh đã chạy trên working tree tại thời điểm kiểm định; các mục suy đoán đã được đánh dấu INFERENCE rõ ràng.*
