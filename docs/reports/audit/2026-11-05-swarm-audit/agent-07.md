# AUDITOR 07 - Báo cáo Kiểm định Độc lập (Swarm 8 tác tử)

- **Ngày audit:** 2026-11-05
- **Vai trò:** Auditor 07, chạy TRỌN 10 chiều T1-T10 trên toàn repo (Zero-Decomposition).
- **Repo:** `SE374_WSSS_WORKSPACE`
- **Môi trường kiểm chứng:** Windows + Git Bash, Python 3.13.9, pytest 8.4.2, ruff 0.12.0, torch 2.10.0+CPU. KHÔNG có GPU -> mọi đường dẫn GPU chỉ audit tĩnh.
- **Nhiệm vụ:** Kiểm chứng lại tuyên bố của audit 2026-11-02 (`01_SWARM_RECHECK_VERIFICATION.md`): "348 passed / 0 failed, ruff 0 errors, 0.0% leakage, 0 non-ASCII hyphen".
- **Ràng buộc:** KHÔNG sửa source/test. Chỉ đọc + chạy lệnh read-only. Tách bạch VERIFIED (có lệnh + output thật) và INFERENCE (đọc mã, suy luận).

---

## 0. Phán quyết tổng quan (Executive)

Bản audit 2026-11-02 **PHÓNG ĐẠI có hệ thống**. Các con số then chốt không tái lập được:

| Tuyên bố cũ (2026-11-02) | Thực đo (2026-11-05) | Kết luận |
| :--- | :--- | :--- |
| 348 passed / 0 failed | **347 passed / 1 failed** | SAI (VERIFIED) |
| ruff 0 errors | src sạch, nhưng `src tests scripts` = **33**, cả repo (trừ repos) = **464** (gồm 6 F821) | SAI / phạm vi hẹp (VERIFIED) |
| 0 non-ASCII hyphen toàn workspace | Deliverable sạch, nhưng **38 file** có U+2013/2014/2015/2212 (`.agents/`, `graphify-out/`) | SAI về phạm vi "toàn workspace" (VERIFIED) |
| "Exactly 32 notebooks" | **60 file .ipynb** (trừ repos) | SAI (VERIFIED) |
| 0.0% patient leakage | Manifest + thư mục thật trên đĩa = **0 overlap** | ĐÚNG cho BCSS (VERIFIED), nhưng có lỗ hổng kiến trúc (xem T3) |

Nghiêm trọng nhất: test bảo vệ INV-12 **crash** (FileNotFoundError) thay khi chạy đủ bộ, và **CI không chạy được** vì thiếu dependency. Test xanh ≠ hành vi đúng (Anti-Mock-Testing Illusion).

---

## T1 - Tầng 1: pytest + ruff (VERIFIED)

**Lệnh:** `KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q -p no:cacheprovider`
**Output thật (507.55s):**
```
1 failed, 347 passed, 2 warnings in 507.55s (0:08:27)
```
**Thu thập:** `pytest --collect-only -q` -> `348 tests collected` (khớp "348 collected" nhưng KHÔNG khớp "348 passed").

**Chữ ký lỗi tệ nhất (nguyên văn):**
```
tests\test_governance_invariants.py:42:
> violations = _scan_files_for_bad_dashes(["**/*.py", "**/*.ipynb"])
...
tests\test_governance_invariants.py:26:
> with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
E   FileNotFoundError: [Errno 2] No such file or directory:
    'drive_upload_ready\\SE374_WSSS_Drive_Code\\notebooks\\SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb'
FAILED tests/test_governance_invariants.py::test_no_ai_em_dash_in_python_and_notebooks
```

**2 warnings (khớp mô tả cũ, vô hại):**
1. `RuntimeWarning: Precision loss ... catastrophic cancellation` tại `scipy/stats/_axis_nan_policy.py:423` (từ `test_clinical_service.py`, cohort mock phương sai ~0).
2. `DeprecationWarning: scipy.misc is deprecated` từ `repos/WSSS-Tissue/tool/infer_utils.py:6` (repo ngoài, `test_colab_notebook.py` import phải).

**Độ tin cậy:** VERIFIED. Lỗi này KHÔNG phải do bug logic dash mà do **test không cô lập** (xem T3/T10 [T1-flaky] và [T10-isolation]). Test pass khi chạy riêng lẻ (`4 passed`), fail khi chạy trong bộ đầy đủ / khi `drive_upload_ready/` (artifact sinh bởi test khác) đang bị `shutil.rmtree` dựng lại. Tôi đã tái lập bằng cách chạy 2 process pytest song song -> confirmed race trên artifact directory.

**Ruff:**
- `python -m ruff check src` -> `All checks passed!` (src/ SẠCH, khớp tuyên bố cho riêng src).
- `python -m ruff check tests` -> `All checks passed!`
- `python -m ruff check scripts` -> **Found 33 errors** (chủ yếu F401 unused-import, F811, E701, F841 trong `scripts/refactor_notebooks_pure_colab.py`, `scripts/update_agents_invariants.py`, ...).
- `python -m ruff check . --statistics --exclude repos` -> **Found 464 errors**:
  - 193 I001, 93 F811, 53 E701, 36 F541, 35 F401, 25 E401, 19 E741, **6 F821 (undefined-name)**, 4 F841.
- 6 lỗi **F821** nằm ở `notebooks/03_Train_Proposed_MedSAM*.ipynb:cell15` và `notebooks/04_Exhaustive_Benchmark_Survival*.ipynb:cell11` (biến `DRY_RUN`). Đây là **false-positive có chủ đích** do pattern `x = False if 'DRY_RUN' not in globals() else DRY_RUN` (tự guard globals) mà ruff AST không hiểu; VÔ HẠI về runtime nhưng chứng minh "0 errors toàn repo" là bịa.

**Kết:** src/ thật sự sạch; nhưng phát biểu "Ruff 0 errors" của audit cũ **sai khi mở rộng ra scripts/ + notebooks/** (33/464). INFERENCE rằng audit cũ chỉ chạy `ruff check src/` rồi suy diễn thành toàn repo.

---

## T2 - Zero-Mockup / INVARIANT-07

**Lệnh:** `grep -rniE "placeholder|synthetic|dummy|fake|mock|np\.random|random\.|torch\.rand|randint|hardcod|fabricat" --include=*.py src/` + mở file đọc ngữ cảnh từng nghi vấn.

**Kết quả: KHÔNG phát hiện số liệu bịa trình bày như kết quả thật trong src.** Các hit đều là:
- **Chống-mockup chủ động (tốt):** `src/clinical/exhaustive_evaluator.py:661,699,719` - khi thiếu dữ liệu sinh tồn, ghi rõ "Zero dummy patients injected" thay vì nhét bệnh nhân giả. `src/colab_pkg/downloader.py:119,150-169` có `_is_mock_path()` loại dataset mẫu 2-ảnh.
- **Seed hợp pháp:** `colab_runner.py:101-102`, `comparative_runner.py:41-42`, `train_stage1/2.py` -> `random.seed/np.random.seed(seed)` (INV-09).
- **Ngẫu nhiên tăng cường/persistence đúng nghĩa:** `dataset_loader.py:343,361,368` (D4 augment), `wsss_tissue.py:111-125` (mixup, đã guard `images.size(0)<2`), `evaluator.py:221,492` (bootstrap CI dùng `np.random.RandomState(seed)` - deterministic, không phải dự đoán giả).

**DEFECT [T2-synthetic-fallback] - Medium - INFERENCE:**
`src/training/train_proposed_sam.py:485-494`:
```python
# Dataloaders: use real data if exists, else synthetic fallback
if os.path.exists(args.data_dir) and not args.dry_run:
    ...
else:
    num_samples = 4 if args.dry_run else 32
    print(f"[INFO] Using SyntheticHistologyDataset with {num_samples} samples.")
    train_ds = SyntheticHistologyDataset(...)   # img=torch.rand, mask=torch.randint(0,5)
```
`SyntheticHistologyDataset.__getitem__` (206-217) sinh `img=torch.rand`, `mask=torch.randint(0,5)`, `weak_label=(torch.rand(4)>0.4)` - ảnh/mask **ngẫu nhiên thuần**. Vòng train (528-533) in `Val mIoU_fg=...` **định dạng y hệt** số liệu thật. Nếu người dùng chạy thiếu `data_dir` và bỏ qua dòng INFO, hệ thống sẽ **báo cáo mIoU/Dice trên mask random** như kết quả mô hình. Rủi ro Zero-Mockup thật (dù có print cảnh báo). Nên chặn (`raise`) khi `not dry_run` mà thiếu dữ liệu.

**Không tìm thấy:** `fake_metric`, `dummyData`, giá trị metric gán cứng trình bày là kết quả đo, mask giả đem báo cáo. **Deliverable sạch.**

---

## T3 - Patient-level Anti-Leakage / INVARIANT-08

### Bằng chứng hành vi thật (VERIFIED - chống Anti-Mock-Testing)
`KMP_DUPLICATE_LIB_OK=TRUE python -c "..."` trên **dữ liệu THẬT** (`data/BCSS-WSSS`, 50.219 PNG):
```
MANIFEST train/val/test patients: 106 22 23
MANIFEST pairwise overlap tr&va,tr&te,va&te: 0 0 0
DISK train/test/val patient count: 106 28 22
DISK train&test overlap: 0
DISK train&val overlap: 0
DISK val&test overlap: 0
```
=> Trên BOTH manifest và tên-file-thật-dưới-đĩa, **0 patient overlap**. "0.0% leakage" **ĐÚNG** cho cấu hình BCSS hiện tại.

### DEFECT [T3-proposed-loader-no-patient-filter] - High - INFERENCE
Repo có **2 loader** và chúng **không nhất quán** về enforcement:
- `src/training/dataset_loader.py` (Stage-1/Stage-2) -> `create_stage1_dataloaders` (465-542) **LỌC theo manifest**: `splits=load_bcss_patient_splits(); train_patients=set(...); ...; if pid not in allowed_set: continue` (501-508, 198-208). ĐÚNG INV-08.
- `src/datasets/bcss_dataset.py::BCSSDataset` -> **KHÔNG hề import/lọc manifest**. `_load_dataset` (48-79) chỉ `glob.glob(root/split/*.png)`. Nó tính `self.patient_ids` (58-59,75-76) **rồi bỏ đó (dead assignment)**, không đối chiếu, không gọi `verify_patient_leakage`.

`BCSSDataset` chính là loader mà **phương pháp đề xuất MedSAM** dùng: `train_proposed_sam.py:487-489` `from src.datasets.bcss_dataset import BCSSDataset; BCSSDataset(root_dir, split="train")`. Nghĩa là **con đường "proposed method" KHÔNG có lớp phòng rò rỉ theo Patient ID**, chỉ trông cậy cấu trúc thư mục BCSS (may mà train/val/test folder đang disjoint - 0 overlap như đã VERIFIED - nên chưa rò rỉ thực tế). Đây là **lỗ hổng kiến trúc tiềm ẩn (latent)**: nếu dataset có cùng bệnh nhân trải qua các folder split (rất phổ biến với patch/slice-level dataset), BCSSDataset sẽ rò rỉ mà không报警.

### Phán xét `tests/test_patient_split.py` (VERIFIED): KHÔNG tự tham chiếu
Tôi đọc full 176 dòng. Đây là test HÀNH VI thật sự, không circular:
- `test_verify_patient_leakage_catches_overlap` (52-60): cố tình tạo overlap `TCGA-A2-A0T2` giữa train và test, assert `ValueError` + message "Patient leakage detected" -> chứng minh hàm BẮT được rò rỉ.
- `test_151_patient_split_manifest_zero_leakage` (94-145): nạp manifest THẬT, kiểm 151 unique, pairwise disjoint bằng cả logic inline lẫn `verify_patient_leakage`.
- `test_verify_patient_leakage_on_sample_dataset` (32-49): glob file thật trong `data/sample_dataset` -> `verify_patient_leakage`.
- Hypothesis fuzz (162-174): property test regex TCGA + `st.text()` random.

**GAP kiểm thử (Medium):** các test chỉ kiểm **manifest + hàm verify rời rạc + sample_dataset tí hon (2 bệnh nhân)**. KHÔNG có test nào chứng minh `create_stage1_dataloaders`/`BCSSDataset` THỰC SỰ áp filter lên tập dữ liệu 151-bệnh-nhân/50k-patch thật, và test của `BCSSDataset` (`test_bcss_dataset_loader_dry_run`, 74-91) chỉ kiểm **shape tensor**, không kiểm rò rỉ. Nên đường rò rỉ-thực (loader path) không được test phủ -> tôi đã bù bằng lệnh VERIFIED ở trên.

### DEFECT [T3-manifest-vs-disk-count] - Low - VERIFIED
Thư mục `test` trên đĩa có **28** bệnh nhân nhưng manifest khai **23** (`exhaustive_evaluator.py:876` ghi "23 Active Test Patients"). Khi áp `allowed_set`, 5 bệnh nhân test ngoài manifest bị **lặng lẽ loại** -> số ca đánh giá lâm sàng thấp hơn dữ liệu có thật. Không phải rò rỉ nhưng là bất nhất dữ liệu/đếm.

---

## T4 - INVARIANT-29 (anti-shallow-glob + safety net)

- `src/training/dataset_loader.py:163` dùng `root_path.rglob("*")` -> **ĐÚNG** đa tầng cho loader chính.
- `src/datasets/bcss_dataset.py:55,72` dùng **`glob.glob(os.path.join(train_dir, "*.png"))`** -> **SUY VI PHẠM INV-29**: glob nông, không `rglob`, không `resolve_split_path`. Loader này chính là của proposed method (lihat T3).

**DEFECT [T4-no-safety-net] - High - VERIFIED+INFERENCE:**
INV-29 bắt buộc mỗi Dataset có `len(files)==0` placeholder để PyTorch không sập `num_samples should be a positive integer value, but got num_samples=0`. Kiểm:
- `grep -n "placeholder|smoke|len(self.files)==0|num_samples"` trong `dataset_loader.py` và `bcss_dataset.py` -> **KHÔNG có safety net**. `__len__` trả `len(self.files)` thẳng (232,310,438 và bcss 94-95). Nếu đường dẫn sai/rỗng -> `len==0` -> `DataLoader` crash đúng loại lỗi INV-29 cấm.
- Bằng chứng tái hiện độc lập (external): StackOverflow + nhiều GitHub issue cùng chữ ký lỗi này (xem T10). INFERENCE: khi `drive_upload_ready/`/`data/` chưa tải xong trên Colab, loader sẽ sập thay vì smoke-test placeholder.

**Không tìm thấy:** safety-net placeholder nào vi phạm T2 (vì chúng không tồn tại). Nghĩa là INV-29 "Emergency Safety Net" **chưa được hiện thực** ở lớp training loader.

---

## T5 - INVARIANT-28 + 27 (Colab lifecycle)

### drive.mount (`grep -rln "drive.mount|/content/drive" --include=*.ipynb` -> 23 file)
Mở cell thật của `notebooks/modular_drive/SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb`:
```python
if "google.colab" in sys.modules:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception: pass
MYDRIVE = Path("/content/drive/MyDrive")
...
WORK_ROOT = Path("/content" if "google.colab" in sys.modules else "./se374_wsss_work")
```
**Đánh giá:** drive.mount **được guard** (chỉ khi google.colab), wrapped try/except pass, và có **4 kịch bản fallback** (Scenario D tự `git clone https://github.com/medical-wsss-consortium/medical-wsss.git`), output lưu `/content` (ổ local VM). => **PHÙ HỢP tinh thần INV-28** (local disk, có GitHub fallback, không bắt buộc). **DEFECT [T5-drive-letter] - Low:** về chữ nghĩa, INV-28 cấm `drive.mount('/content/drive')`; nó vẫn được gọi và lần đầu trên Colab **vẫn bật pop-up OAuth** (trái "không dừng Run All đòi quyền Drive"), dù từ chối thì `except pass` đi tiếp.

### Vòng đời 7 bước (INV-27) - đo bằng python đếm regex trên 4 notebook staged T4:
```
01_Stage1 : gpu=True install=False realdata=True weights=True metric=False viz=True zipdownload=True
02_Stage2 : gpu=True install=False realdata=True weights=True metric=True  viz=True zipdownload=True
03_MedSAM : gpu=True install=False realdata=True weights=True metric=True  viz=True zipdownload=True
04_Bench  : gpu=True install=False realdata=True weights=True metric=True  viz=True zipdownload=True
```
- **Bước 2 (cài thư viện) thiếu** ở cả 4 notebook staged (`install=False`): chúng giả định `src/` + torch/torchvision đã có, không có cell `pip install` cho monai/segmentation-models/opencv -> trên Colab runtime sạch sẽ `ModuleNotFoundError`. INFERENCE (03 cần `segment_anything`).
- `metric=False` cho 01 là **bỏ nhầm do regex** (Stage-1 là classifier, đo accuracy/AUROC/F1, không phải mIoU/Dice) - KHÔNG phải defect.
- `zipdownload=True` (files.download/make_archive/ZipFile) có đủ -> **INV-27 bước 7 (đóng gói ZIP + tải 1-click) ĐẠT**.

### Drift GPU variants & copies: xem T9.

---

## T6 - Device & tài nguyên (đa phần SẠCH)

**VERIFIED qua audit tĩnh toàn `src/`:**
- **`torch.load` + weights_only (torch>=2.6 default True):** TẤT CẢ call site dùng đúng pattern try `weights_only=True` / except `weights_only=False` + `map_location="cpu"`. Liệt kê 22 call site (downloader 230/232, model_loader 574/576, wsss_inference 363/365, generate_pseudo_masks 321/323, train_proposed_sam 402/404, train_stage1 390/392, train_stage2 298/300/309/311) + `safe_torch_load` (3 định nghĩa). => Repo **đã phòng bộ risk torch 2.6 migration** (đối chiếu issue nnUNet #2681 ở T10). KHÔNG có `torch.load(...)` trần thiếu weights_only.
- **map_location:** đầy đủ mọi chỗ load.
- **AMP/autocast:** `train_proposed_sam.py:274` dùng API mới `torch.amp.autocast(device_type=device.type, enabled=use_cuda_amp)` với `use_cuda_amp=(device.type=="cuda")`; `GradScaler("cuda")` chỉ tạo khi cuda (473). CPU-safe.
- **SAM prompt tensors device:** `sam_wsss_adapter.py:252-261` dựng `point_coords/point_labels/active_mask` với `device=device` lấy từ `cams.device`; `706` `.to(coords.device)`. => prompt chuyển đúng device.
- **Seed (INV-09):** `random.seed/np.random.seed` + `torch.manual_seed` đặt tại 5 entrypoint train; notebook staged `np.random.seed(SEED)`.
- **VRAM T4 16GB:** notebook đặt `WORK_ROOT`, batch/accum `accum_steps=2`, `PROGRESS_INTERVAL=250`, TF32 config - sizing có chủ đích cho T4/L4/A100 (audit tĩnh, không có GPU nên không đo peak VRAM thật).

**Liệt kê nơi thiếu (INFERENCE):** vì không có GPU, không verify được `scaler` thực chạy; không đo được OOM peak. Đường GPU chỉ audit tĩnh. **KHÔNG phát hiện `.to("cuda")` cứng** bỏ rơi CPU path.

---

## T7 - Silent failure

`grep -rn "except Exception" src/` -> **49**; bare `except:` -> **0**; `contextlib.suppress` -> **0** (tốt). Hầu hết `except Exception as e:` có log/tái-nạp hợp lệ. Nhưng có nhóm **biến lỗi thật thành kết quả vô nghĩa**:

**DEFECT [T7-stats-silent-zero] - High - VERIFIED (đọc mã):**
`src/colab_pkg/evaluator.py:410-422`:
```python
try:
    t_res = stats.ttest_rel(arr_method[:n], arr_base[:n])
    t_stat = float(0.0 if np.isnan(t_res.statistic) else t_res.statistic)
    t_pval = float(1.0 if np.isnan(t_res.pvalue) else t_res.pvalue)
except Exception:
    t_stat, t_pval = 0.0, 1.0
try:
    w_res = stats.wilcoxon(diffs, ...)
    ...
except Exception:
    w_stat, w_pval = 0.0, 1.0
```
Nếu phép kiểmpaired t-test/Wilcoxon **sập** (scipy lỗi, NaN, mảng rỗng) hoặc cho `pvalue=NaN`, nó **âm thầm trả p=1.0** = "không khác biệt có ý nghĩa". Trong bảng benchmark so sánh MedSAM vs baseline, một crash thống kê sẽ bị **báo thành "không cải thiện"** thay vì报错 -> kết luận lâm sàng/benchmark sai hướng mà không ai biết. Nan-to-0 cũng che cảnh báo dữ liệu suy biến.

**DEFECT [T7-load-mask-corruption] - Medium - INFERENCE:**
Cùng pattern try/except ở mọi `torch.load`: nếu checkpoint **thật sự hỏng/bị giả mạo** (không chỉ chứa object non-tensor), `except Exception` sẽ **lặng lẽ reload với `weights_only=False`** = thực thi pickle tùy ý (rủi ro code-exec, bandit đã đánh dấu nosec B614). Lỗi hỏng file bị che thành "load thành công nhưng không an toàn".

**DEFECT [T7-missing-mask-zero] - Medium - INFERENCE:**
`bcss_dataset.py:132-133` và `dataset_loader.py:456-458`: khi file mask thiếu, fallback `mask_tensor = torch.zeros(...)` (toàn nền). Trong **đánh giá**, mẫu mất mask được tính như một ca "không có tổn thương" -> IoU/Dice bị kéo lệch **không cảnh báo** -> metric vô nghĩa một phần.

---

## T8 - INVARIANT-12 + CI

### Dash scan (VERIFIED)
`grep -rnP "[\x{2013}\x{2014}\x{2015}\x{2212}]" --include=*.py --include=*.md --include=*.ipynb .` (trừ repos) -> **38 file vi phạm**, NHƯNG giới hạn ở:
- `.agents/rules/*`, `.agents/teamwork/**` (docs nội bộ swarm; Python `glob` **bỏ qua thư mục chấm `.`, nên test INV-12 cũ không quét tới**), `graphify-out/**` (nằm trong `IGNORED_DASH_PREFIXES`).
Quét RIÊNG thư mục deliverable/source:
`grep ... src tests scripts notebooks product docs drive_upload_ready experiments *.ipynb` -> **0 vi phạm**.
=> **INV-12 COMPLIANT cho mọi sản phẩm nộp + mã nguồn.** Tuyên bố "0 non-ASCII toàn workspace" là **sai về mặt chữ** (38 file), dù phạm vi ý-nghĩa (deliverable) đúng.

### `.github/workflows/ci.yml` đối chiếu từng lệnh (VERIFIED)
1. **`pip install flake8 pytest Pillow` (27)** nhưng test import **torch(40 file), monai, cv2, scipy, fastapi, hypothesis** (đo bằng grep import) và **không có bước `pip install -r requirements.txt`**. Bước 4 `pytest tests/ -v` (53) sẽ **crash collection `ModuleNotFoundError: torch`** -> **CI KHÔNG BAO GIỜ ĐỎ nhưng cũng KHÔNG BAO GIỜ XANH THẬT**; pipeline dựng trên repo này **bị hỏng**. **[T8-ci-broken] CRITICAL.**
2. **Lint không gate:** bước 1 (32-33) dùng `flake8 ... || true` và `--exit-zero` -> **mọi lỗi lint bị bỏ qua**, build không bao giờ fail vì style. Dùng **flake8 thay vì ruff** (repo chuẩn là ruff.toml) -> sai ngăn xếp, vi phạm tinh thần INV-22/CI hiện đại.
3. **Zero-mockup grep quá hẹp** (38): chỉ quét 5 token tự điển `'TODO: mock','fakeData','dummyData','fake_metric','simulateTx'` - **không bắt được** synthetic fallback (T2), random-as-prediction, hay metric gán cứng. Gate chống-mockup minh họa (cosmetic).
4. **Bước 3 deliverable_auditor chạy ĐƯỢC và gate THẬT** (VERIFIED): `python -m src.utils.deliverable_auditor product/proposal/SE374_MINI_PROPOSAL_WSSS_WSI.md` -> `[PASSED] ... EXIT=0`. Feed file có em-dash -> `[CRITICAL] ... INVARIANT-12` và **exit code 1** -> auditor **thực sự chặn**, không phải rubber-stamp. (File proposal tồn tại, đã `ls`.)
5. CI `python-version: '3.10'` nhưng dev/test chạy **3.13.9** -> lệch phiên bản (torch 2.10 vs 2.2).

### `tests/test_governance_invariants.py` thực thi enforce gì
Đọc full 63 dòng. 4 test: 2 scan dash (`**/*.md`, `**/*.py`+`**/*.ipynb`), 1 check INV-24/25 có trong AGENTS.md (substring), 1 check `product/` không có ảnh chụp. **Enforce THẬT** nhưng (a) **crash** vì quét artifact sinh-run (T1), (b) **bỏ sót `.agents/` do dot-dir glob**, (c) INV-24/25 chỉ kiểm sự tồn tại chuỗi, không kiểm hình ảnh thực. INFERENCE.

---

## T9 - Dead code & duplication

### `src/config/` (VERIFIED)
`ls src/config` -> chỉ `.gitkeep` + `bcss_patient_splits.json`, **KHÔNG `__init__.py`, 0 file `.py`**. `grep "import src.config|from src.config"` -> **0** -> không ai import nó như Python package (JSON được đọc qua `Path(...)/"bcss_patient_splits.json"`). => KHÔNG phải dead module, **NHƯNG** AGENTS.md mục layout hứa `src/config/` = "siêu tham số, random seeds, paths, device GPU", thực tế **siêu tham số nằm rải trong notebook/script** (seed/batch/path lặp ở 4 bản sao notebook) -> **không có central config module**, trái INV-30/INV-09 về quản lý seed tập trung. Medium.

### Entrypoint INV-02 (VERIFIED)
`ls src/pipeline/run_pipeline.py` -> **No such file**. AGENTS.md:190 khai `python -m src.pipeline.run_pipeline` là Pipeline Orchestrator (INV-02 bắt buộc điều phối qua `src/pipeline/`). **Lệnh tài liệu hóa không tồn tại** -> gọi sẽ `No module named src.pipeline.run_pipeline`. Medium-High (tài liệu vs code lệch).

### Notebook duplication/drift (VERIFIED - md5)
So `md5sum` cặp trùng tên root vs `notebooks/`:
```
[SAME]  01_Train_Stage1_Classifier_T4.ipynb  vs  notebooks/...
[SAME]  02_Pseudo_Masks_Train_Stage2_T4.ipynb vs notebooks/...
[SAME]  03_Train_Proposed_MedSAM_T4.ipynb    vs  notebooks/...
[DRIFT] notebooks/SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2  vs drive_upload_ready/...
[DRIFT] notebooks/SE374_Colab_WSI_LowMag_HighMag_BonusDemo  vs  ./SE374_Colab_WSI_LowMag_HighMag_BonusDemo
```
- **DEFECT [T9-demoverse-drift] - Medium (VERIFIED):** `SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb` tồn tại ở **4 bản phân kỳ**: root 40.318B, `notebooks/` 29.395B, root `_v2` **686.888B**, `drive_upload_ready/` 28.888B -> **nhiều "source of truth" không đồng bộ**, khó biết bản nào là chuẩn (đối ngược với "exactly 32 notebooks" của audit cũ; thực = 60).
- Drift `drive_upload_ready/` là **hệ quả build artifact** (packager rewrite notebook), chấp nhận được, nhưng nó chính là nguyên nhân làm governance test crash (T1/T10).

### Module chết (VERIFIED - grep tham chiếu)
- `draw_all_proposal_figures.py`, `draw_pipeline_diagram.py` -> **0** lượt được import bởi src/tests/scripts/notebooks (standalone CLI figure-gen, "dead" theo nghĩa import). Low.
- **Trùng tên:** `src/pipeline/colab_runner.py` VÀ `src/colab_pkg/colab_runner.py` - 2 runner cùng tên, nguy cơ trôi lệch. Low/Medium.
- `bcss_dataset.py::BCSSDataset.patient_ids` -> **dead assignment** (tính ra không dùng, xem T3).

---

## T10 - Chất lượng test + bug-hunt đối chiếu ngoài

### Đánh giá `tests/` theo testing-best-practices (VERIFIED đọc mã)
- **Test HÀNH VI thật (tốt):** `test_patient_split.py` (xét T3), `test_deeplabv3plus.py` (forward + penalty -1e4), `test_benchmark_components.py` (IoU/Dice accumulator), `test_clinical_service.py` (Kaplan-Meier/log-rank). Đây là behavioral AAA, ít mock, có đường lỗi (overlap->ValueError, invalid path->FileNotFoundError).
- **DEFECT [T10-weak-notebook-tests] - Medium (VERIFIED):** `tests/test_colab_pro_notebooks.py` là module LỚN NHẤT (71 test) nhưng chủ yếu **AST-compile + assert chuỗi con** trên text notebook, kiểu `assert "tqdm" in content`, `assert "stage1_latest.pth" in content`, `assert "google.colab" in content` (dòng 118-131...). **KHÔNG thực thi logic notebook, không đo metric** -> cảm giác an toàn giả (Anti-Mock-Testing Illusion) cho chính artifact Colab. Tương tự `test_colab_notebook.py` (30 test) thiên về cấu trúc JSON/AST.
- **DEFECT [T1-flaky]/[T10-isolation] - Critical (VERIFIED):** `test_governance_invariants.py::test_no_ai_em_dash_in_python_and_notebooks` **phụ thuộc artifact sinh-run**: nó `glob("**/*.ipynb")` từ CWD **bao gồm `drive_upload_ready/` (gitignored, `.gitignore:32`)** rồi `open()` từng file; trong khi `test_thin_drive_runner.py` (test_2/3/7) gọi `packager.build_drive_package()` vốn `shutil.rmtree(bundle_dir)` (build_drive_package.py:1545) dựng/xóa đúng thư mục đó. => **order- & state-dependent, KHÔNG cô lập**: fail trong full run (FileNotFoundError), pass đơn lẻ (`4 passed`), pass ở 1 run nền (exit 0) -> non-deterministic. Vi phạm nguyên tắc test isolation. Ignore-list (`IGNORED_DASH_PREFIXES`) **quên loại `drive_upload_ready/`**.
- **Nondeterminism/thời gian:** không thấy `time.sleep` trong test; có `coarse_to_fine_router.py:238` đo wall-clock latency (runtime, không phải test) -> nếu test bất kỳ assert latency cố định sẽ fragile; ghi nhận INFERENCE.
- **Phủ đường lỗi/biên:** tốt cho split/leakage/invalid path; **YẾU** cho rỗng: không test `len(dataset)==0`/mask-thiếu/None -> đúng chỗ T4/T7 sụp.

### Bug-hunt đối chiếu ngoài (site:stackoverflow / github issues, chữ ký thật)
**(1) `ValueError: num_samples should be a positive integer value, but got num_samples=0`**
- Mã repo CÓ NGUY CƠ cao: `BCSSDataset.__len__`/`dataset_loader` trả 0 khi glob nông/trạng thái rỗng, KHÔNG có placeholder (T4).
- Nguồn đối chiếu: StackOverflow accepted Q71615089 + facebookresearch/detr#196. Bản vá chuẩn (trích thông lệ accepted): kiểm `if len(dataset)==0: raise` tường minh HOẶC chèn 1 sample placeholder (chính là INV-29 "Emergency Safety Net" repo cam kết nhưng chưa làm).
  - https://stackoverflow.com/questions/71615089/valueerror-num-samples-should-be-a-positive-integer-value-but-got-num-samples
  - https://github.com/facebookresearch/detr/issues/196

**(2) `torch.load` weights_only mặc định đổi ở PyTorch 2.6**
- Chữ ký: `UnpicklingError: Weights only load failed ... PyTorch 2.6 changed default of weights_only ... it can result in arbitrary code execution`.
- Repo **đã xử lý đúng** bằng try `weights_only=True`/except `weights_only=False` + map_location (T6). Đối chiếu accepted: nnUNet#2681, Lightning#20643, axondeepseg#994 khuyến nghị cân nhắc `weights_only=False` chỉ cho checkpoint tin cậy + validate -> repo dùng `# nosec B614` nhưng **except quá rộng** (nuốt cả lỗi hỏng file -> reload pickle-unsafe, T7).
  - https://github.com/MIC-DKFZ/nnUNet/issues/2681
  - https://github.com/Lightning-AI/pytorch-lightning/issues/20643

**(3) `RuntimeWarning: Precision loss ... catastrophic cancellation` (scipy) - đã hiện diện trong suite**
- Chữ ký khớp audit cũ (mock cohort phương sai ~0). Đối chiếu scipy `_axis_nan_policy`: p-value khi dữ liệu gần-trùng-lặp không tin cậy. Repo che bằng con số -> liên quan T7 (báo p mặc định). Ghi nhận để không dùng cohort 5-ảnh-trùng cho kết luận lâm sàng.

---

## Ma trận nghiệm thu (Verification Matrix)

| Chiều | Kết quả | DEFECT |
| :--- | :--- | :--- |
| T1 pytest/ruff | **347 passed / 1 FAILED** (không phải 348/0); ruff scripts=33, repo=464 (6 F821) | [T1-flaky] Critical, [T3-claim-inflation] |
| T2 Zero-Mockup | src/deliverable SẠCH; synthetic fallback hazard | [T2-synthetic-fallback] Medium |
| T3 Leakage | Manifest+disk thật 0 overlap (VERIFIED); test không tự tham chiếu | [T3-proposed-loader] High, [T3-count] Low |
| T4 glob/safety net | dataset_loader rglob OK; BCSSDataset shallow glob, KHÔNG safety net | [T4-no-safety-net] High |
| T5 Colab lifecycle | guard drive.mount OK, ZIP+files.download đủ; thiếu pip install cell | [T5-drive-letter] Low, thiếu bước cài lib |
| T6 Device | torch.load weights_only hygiene ĐÚNG; SAM prompt device-correct | SẠCH (chỉ INFERENCE GPU) |
| T7 Silent fail | 49 except Exception, 0 bare/suppress; stats->p=1.0, mask->0 | [T7-stats-silent-zero] High, [T7-load] Medium, [T7-mask] Medium |
| T8 INV-12 + CI | deliverable 0 dash (VERIFIED); auditor gate THẬT (exit 1) | [T8-ci-broken] Critical, lint `||true`, mockup grep hẹp |
| T9 Dead/dup | run_pipeline không tồn tại; config 0 .py; 60 notebook; demo drift | [T9-demoverse-drift] Medium, [INV-02 entry] Med-High, dead figs Low |
| T10 Test quality | behavioral tests tốt; colab_pro 71 test chỉ AST/substring; isolation fail | [T10-weak-notebook-tests] Medium |

## Tổng đếm defect (Auditor 07)
- **Critical: 3** - [T1-flaky-governance-test], [T8-ci-broken], [T3-claim-inflation-phóng-dại-audit-cũ].
- **High: 3** - [T3-proposed-loader-no-patient-filter], [T4-no-safety-net], [T7-stats-silent-zero].
- **Medium: 6** - [T2-synthetic-fallback], [T7-load-mask-corruption], [T7-missing-mask-zero], [T9-demoverse-drift], [T9-config/INV-02-entry], [T10-weak-notebook-tests].
- **Low: 4** - [T3-manifest-vs-disk-count], [T5-drive-letter], [T9-dead-modules], [T9-dup-colab-runner].
- **SẠCH có bằng chứng:** src/ ruff, INV-12 deliverable, torch.load weights_only hygiene, SAM prompt device, 0.0% leakage BCSS (manifest + 50.219 ảnh thật), drive.mount guard + local disk + ZIP export, auditor là gate thật (exit 1).

## Ghi chú phương pháp
- Đã chạy pytest thật (507s), load 1 batch THẬT từ `data/BCSS-WSSS` (shape 4,3,224,224) và `checkpoints/stage1_checkpoint_trained_on_bcss.pth` -> đạt Anti-Mock-Testing Gate (không chỉ tin pytest xanh).
- Mọi lệnh prefix `KMP_DUPLICATE_LIB_OK=TRUE`. GPU audit tĩnh (không có T4/L4 trên máy).
- Không sửa source/test; không tạo script mới trong src/tests/scripts (chỉ `python -c` inline read-only).
