# SWARM AUDIT MASTER REPORT - 2026-11-05

- **Chế độ:** Kiểm định đối kháng đa tác tử theo `AGENTS.md` [INVARIANT-26] + [INVARIANT-31] (3 tầng).
- **Cơ cấu:** 8 tác tử độc lập (N chẵn), cùng một bộ prompt đồng nhất, mỗi tác tử tự chạy trọn 10 chiều T1-T10 trên toàn repo. Lead chỉ điều phối và tái đo, không sửa mã.
- **Báo cáo từng tác tử:** `agent-01.md` (807 dòng) ... `agent-08.md` (706 dòng). Hai tác tử 04 và 05 chạm trần 150 lượt nhưng vẫn ghi ra báo cáo đầy đủ (944 và 591 dòng).
- **Ngưỡng True Defect:** >= 3 tác tử đồng thuận độc lập trên cùng file/tính năng, kèm số dòng minh chứng, và Lead tái đo bằng lệnh.

---

## 0. SỰ CỐ PHÁ HỦY TÀI SẢN TRONG LÚC AUDIT (đọc trước tiên)

Chính bộ lệnh audit tôi giao (`python -m pytest tests/`) đã gây ghi đè tài sản thật. 4/8 tác tử tự khai báo (01, 03, 06, 08). Lead xác minh bằng mtime.

| Tài sản | Trạng thái | Khôi phục được không |
| :--- | :--- | :--- |
| `checkpoints/medsam_wsss_adapter_best.pth` (86 MB) | bị ghi đè 03:40 hôm nay | **KHÔNG**. Gitignored, không bản sao nào trong 12 tệp `.zip`, không có trong `drive_upload_ready/` |
| `checkpoints/medsam_wsss_adapter_last.pth` (86 MB) | bị ghi đè 03:40 hôm nay | **KHÔNG** (như trên) |
| `notebooks/SE374_Colab_WSSS_{T4,L4,A100,Pipeline,Sanity150}.ipynb`, `SE374_Colab_Scientific_Research_Runner.ipynb` | bị build script dựng lại | **CÓ** (được git theo dõi, diff chưa commit) |
| `product/proposal/assets/preview/pages/page_1..10.png` | bị ghi đè lại | **CÓ** (được git theo dõi) |
| `drive_upload_ready/**` | bị `shutil.rmtree` + dựng lại | Một phần; là build artifact, dựng lại bằng `scripts/build_drive_package.py` |
| `src/**`, `tests/**` | **KHÔNG bị chạm** (mtime 2026-11-02; 72 diff trong working tree là công việc chưa commit từ trước audit) | Không cần |

**Cơ chế gốc (True Defect D5, 8/8 đồng thuận):** `tests/test_thin_drive_runner.py:96,175,493` và `tests/test_colab_notebook.py:921-924` bỏ qua fixture `tmp_path`, gọi thẳng `build_drive_package()` / `build_variants()`, và `scripts/build_drive_package.py:1525-1545` gọi `shutil.rmtree()` trên `drive_upload_ready/` thật. Suite **không hermetic** và **tự phá** khi chạy.

**Hệ quả dây chuyền:** vì 8 tác tử chạy `pytest` song chồng lên nhau, mỗi lần chạy làm biến đổi trạng thái đĩa, nên con số pytest đo được **phân kỳ giữa các tác tử**: `348/0`, `347/1`, `346/2`, `345/3`, `2 failed/346 passed`, `5 failed/343 passed`. Bản thân hiện tượng này là bằng chứng cho D5, nhưng đồng nghĩa **không con số pytest đơn lẻ nào là đáng tin** nếu chưa sửa test isolation.

**Hành động Lead đã làm (thuận nghịch, không phá):** cô lập hiện trường sang `scratch/quarantine-2026-11-05/` (2 checkpoint + 2 notebook). **Chưa** chạy `git restore`/`git checkout` cho bất kỳ tệp nào.

---

## 1. PHÁN QUYẾT VỀ BẢN AUDIT KỲ TRƯỚC (2026-11-02)

`docs/reports/audit/2026-11-02-swarm-audit/01_SWARM_RECHECK_VERIFICATION.md` **phóng đại có hệ thống**. 5/7 tuyên bố của nó bị bác bằng lệnh thật:

| Tuyên bố cũ | Thực đo 2026-11-05 | Kết luận |
| :--- | :--- | :--- |
| "348 Passed / 0 Failed" | Phụ thuộc state đĩa: `345 passed / 3 failed`, `346/2`, `347/1` | **SAI** |
| "Ruff: 0 errors (all 42 modules)" | `ruff check src tests` = All checks passed; `ruff check scripts` = **33 errors**; `notebooks/` = **6x F821 undefined-name** | **SAI phạm vi** (đúng riêng `src/`+`tests/`) |
| "0 non-ASCII hyphen across entire workspace" | 230 dòng / 38 tệp chứa U+2013/U+2014 ngoài `repos/`, gồm **2 ký tự U+2013 ngay trong tệp PPTX nộp** `product/proposal/SE374_PROPOSAL_SLIDES_WSSS_WSI.pptx` | **SAI** (test enforce có blind spot thư mục bắt đầu bằng dấu chấm) |
| "0.0% Patient Leakage" | Đúng trên dữ liệu sản xuất (106/22/23 patient disjoint, 0 overlap, đối chiếu 31.826 - 50.207 ảnh thật) | **ĐÚNG** - nhưng phương pháp **không được test chứng minh** (xem D10) |
| "100% Zero-Mockup" | Artifact đã commit chứa số liệu vô nghĩa (xem D8) | **SAI** |
| "CI/CD Gate" | Workflow **chưa từng chạy** (`total_count = 0`) | **SAI** |

---

## 2. MA TRẬN ĐỒNG THUẬN (Peer-Swap Consensus)

Cột "Votes" = số tác tử độc lập báo cáo cùng defect. Ngưỡng True Defect: >= 3.

| ID | Defect | Votes | Severity | Lead tái đo | Điểm dẫn chứng |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **D1** | CI gate vô hiệu hoàn toàn | 7/8 | **Critical** | VERIFIED | `.github/workflows/ci.yml:5-9` trigger `main`/`dev`, branch thật là `master`; `:27` chỉ cài `flake8 pytest Pillow` trong khi 19/22 test cần torch; `:32-33` `flake8 ... \|\| true` + `--exit-zero` |
| **D2** | Fallback âm thầm sang mô hình rác rồi báo cáo thành tích | 6/8 | **Critical** | VERIFIED | `src/clinical/exhaustive_evaluator.py:338-347` load fail -> `DeepLabV3PlusWithCGM` **trọng số ngẫu nhiên**, trong khi `:316` in ra "All 3 benchmark models successfully initialized"; `src/models/sam_wsss_adapter.py:759-765` nuốt `ModuleNotFoundError` -> `SurrogateImageEncoder`, và `grep is_surrogate` ngoài file adapter = **0 consumer** |
| **D3** | Ground truth bị thay bằng hằng số | 3/8 | **Critical** | VERIFIED | `src/clinical/exhaustive_evaluator.py:439` `np.full(shape_hw, fill_value=4)` khi thiếu mask -> metric tự sinh |
| **D4** | Mask thiếu -> toàn 0 (label TƯỜNG), và `np.clip` biến nhãn ignore 255 thành 4 | 5/8 | **Critical** | VERIFIED | `src/training/dataset_loader.py:322,324-326` và lặp lại `:455,456-458`; `fallback_mask=True` mặc định BẬT |
| **D5** | Test không hermetic, phá repo khi chạy | 8/8 | **Critical** | VERIFIED | `tests/test_thin_drive_runner.py:96,175,493`, `tests/test_colab_notebook.py:921-924`, `scripts/build_drive_package.py:1525-1545` (`shutil.rmtree`) - gây ra mục 0 |
| **D6** | Artifact đã nộp chứa số liệu bịa/vô nghĩa | 6/8 | **Critical** | VERIFIED | `product/reports/clinical/exhaustive_3way_benchmark_results.json` (commit `9e3596a`, 2026-10-24): `total_patches_evaluated: 2` (x3 block), `patient_cohort_count: 1`, `t_statistic: Infinity`, `t_test_p_value: 0.0`, `cohen_d: 0.0`, `overall_survival_months: 15.0`; tệp `.md` anh em lại ghi "23 Active Test Patients" |
| **D7** | Bảng 5-seed công bố mà không có log; `[INVARIANT-09]` chưa từng được thực thi | 5/8 | **High** | VERIFIED | `product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md:190-197` std = 0.00; `src/pipeline/comparative_runner.py:35,111` `DEFAULT_SEEDS` không được tham chiếu |
| **D8** | Dataset glob nông, không có lưới an toàn `len == 0` | 8/8 | **High** | VERIFIED | `src/datasets/bcss_dataset.py:55,72` `glob.glob(os.path.join(dir,"*.png"))`; không `rglob`, không `resolve_split_path`, không placeholder -> đúng chữ ký `ValueError: num_samples should be a positive integer` |
| **D9** | `drive.mount("/content/drive")` vi phạm `[INVARIANT-28]` | 7/8 | **High** | VERIFIED | đếm **23 notebook** ngoài `repos/`; `tests/test_thin_drive_runner.py::test_10` còn **assert sự tồn tại** của nó = pháp chế hóa vi phạm |
| **D10** | Test chống rò rỉ tự tham chiếu, không chứng minh gì | 7/8 | **High** | VERIFIED | `tests/test_patient_split.py:29` bau `extract_patient_id("GDPH_1234_patch_05.png") == "GDPH"` trong khi docstring `:14` ghi `-> 'GDPH_1234'`; `extract_patient_id` bị **nhân bản 4 bản** khác nhau |
| **D11** | Metric bịdrop âm thầm, thống kê sập thì báo `p = 1.0` | 5/8 | **High** | VERIFIED | `src/colab_pkg/evaluator.py:172-182` (FWIoU/BF1 bị loại không báo), `:414-421` (nuốt crash thống kê -> `p=1.0` "không khác biệt") |
| **D12** | `ruff check` bị công bố ở phạm vi sai | 8/8 | **High** | VERIFIED | `scripts/` = 33 errors; `notebooks/` = 6 lỗi `F821 undefined-name` (`DRY_RUN`) -> `NameError` khi Run All trên Colab |
| **D13** | `torch.load` / `load_state_dict` che mất mismatch | 7/8 | **Medium** | VERIFIED | `exhaustive_evaluator.py:331,363` `strict=False`; `src/colab_pkg/model_loader.py:582,615-622` in "[OK] Nap trong so thanh cong (Missing: 567, Unexpected: 1)" |
| **D14** | Notebook drift giữa 3 vị trí lưu bản sao | 7/8 | **Medium** | VERIFIED | md5 khác nhau giữa `gốc` vs `notebooks/` vs `drive_upload_ready/` ở cả 4 cấu hình |
| **D15** | Blind spot trong bộ quét chính tả `[INVARIANT-12]` | 4/8 | **Medium** | VERIFIED | `src/utils/deliverable_auditor.py` + `tests/test_governance_invariants.py` dùng `glob.glob` dot-suppressed, `IGNORED_DASH_PREFIXES` thiếu `.agents` -> 230 dòng vi phạm lọt lưới |
| **D16** | Fallback synthetic dataset lưu checkpoint thật | 4/8 | **Critical** | VERIFIED | `src/training/train_proposed_sam.py:485-500` `SyntheticHistologyDataset` (`torch.rand`) in `Val mIoU_fg=0.0861` không nhãn synthetic và **ghi `.pth`** -> đây chính là con đường đã xóa checkpoint mục 0 |

**Không phải defect (xác nhận SẠCH, có lệnh chứng minh):** đường truyền patient-level split trên dữ liệu thật 0.0% overlap; prompt tensor của SAM chuyển device đúng; `pin_memory` gated by `cuda.is_available()`; bootstrap có seed; `ruff src/ + tests/` = 0; thí nghiệm liên canh `experiments/colab_run_1_stage2_baseline/` là 4986 patch T4 thật (mIoU 68.706).

---

## 3. XẾP HẠNG RỦI RO THEO HỆ QUẢ LÂM SÀNG / HỌC THUẬT

1. **Mất irretrievable 1 cặp checkpoint MedSAM** + nguy cơ tái diễn mỗi lần `pytest` (D5, D16).
2. **Số liệu nộp có thể bị kết luận là bịa** -> chế tài 0 điểm theo [INVARIANT-07] (D2, D3, D4, D6, D7).
3. **Không có CI gate thật**: mọi "nghiệm thu tự động" là danh nghĩa (D1, D12, D15).
4. **Pipeline dễ sập trên Colab**: `F821 DRY_RUN`, `num_samples = 0`, `drive.mount` chặn `Run All` (D8, D9, D12).
5. **Kết luận thống kê không đáng tin** (D10, D11).

---

## 4. ĐỀ XUẤT THỨ TỰ XỬ LÝ (chưa thực thi - chờ Người dùng phê chuẩn)

- **B1 (cấp bách, chặn tái phá):** cô lập test phá hủy - buộc `test_thin_drive_runner.py` / `test_colab_notebook.py` dùng `tmp_path`, chặn `shutil.rmtree` ngoài `tmp_path`.
- **B2:** khôi phục tệp tracked bị ghi đè (10 PNG + 6 notebook) và **train lại** cặp MedSAM checkpoint (không có nguồn khôi phục nào khác).
- **B3:** siết "không bao giờ chạy inference bằng mô hình random-init/surrogate mà không raise" cho D2, D3, D4, D13, D16.
- **B4:** vô hiệu hóa và thay bằng số liệu thật có log cho D6, D7; xóa artifact commit `9e3596a`.
- **B5:** sửa CI (D1), ruff scope + `DRY_RUN` (D12), `drive.mount` (D9), shallow glob + safety net (D8).
- **B6:** khử 4 bản `extract_patient_id` về 1, sửa test tự tham chiếu (D10, D15, D14).

Theo [INVARIANT-26], Fixer Swarm chỉ được giao mã sau khi Người dùng duyệt phạm vi, và mỗi mã sửa phải chuyển sang tác tử kiểm toán độc lập khác re-audit (Peer-Swap Fix Handoff), Fixer không tự viết test nghiệm thu.
