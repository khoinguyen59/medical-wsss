# BAO CAO KIEM TOAN TOAN DIEN DU AN SE374.R11 (ZERO-TRUST AUDIT REPORT)

- **Ngay lap bao cao:** 2026-11-15
- **Pham vi kiem toan:** Toan bo ma nguon `src/`, bo kiem thu `tests/`, kich ban tu dong `scripts/`, toan bo `60` file `.ipynb` ngoai `repos/`, tai lieu/san pham trong `product/`, cau hinh `.github/workflows/ci.yml`, va doi chieu repo tac gia trong `repos/WSSS-Tissue/`.
- **Phuong phap:** Kiem toan zero-trust tu nguyen ly goc (doc truc tiep ma nguon, notebook, bao cao, va chay doc lap cac cong cu kiem chung), khong tin mac dinh vao trang thai PASS cua bo `tests/` hien co.
- **Trang thai thay doi ma nguon:** Khong sua bat ky file code, test, script, notebook, hay bao cao nao trong dot kiem toan nay. Day la ban bao cao doc lap cho phe duyet tu nguoi dung truoc khi sua loi.

---

## 1. Tom tat Dieu hanh & Ket qua Kiem chung Doc lap

He thong hien tai dat trang thai PASS tren bo `316` unit/integration tests khi chay rieng thu muc `tests/`, nhung kiem toan doc lap phat hien **hang loat loi nghiem trong thuoc 4 nhom (A, B, C, D)** lam pipeline huan luyen/danh gia tren Google Colab bi vo hoac tao ra ket qua sai lech ve mat khoa hoc, trong khi tai lieu bao cao trong `product/` con ton tai so lieu va duong dan hong.

### 1.1. Bang Ket qua Chay Lenh Kiem chung Doc lap

| Cong cu / Lenh kiem chung | Exit Code | Ket qua Quan sat Truc tiep | Y nghia Kiem toan |
| :--- | :---: | :--- | :--- |
| `python -m pytest tests/ --collect-only -q` | `0` | `316 tests collected in 5.76s` | Chi quet thu muc `tests/`, ne duoc loi o root repo |
| `python -m pytest --collect-only -q` (tai root) | `2` / Timeout | Timeout sau `80.51s`, `1779 tests collected, 728 errors` | Pytest quet xuyen `repos/` va gap loi `AttributeError: module 'triton' has no attribute ...` trong 10 test cua du an |
| `python -m ruff check src tests scripts --output-format=concise` | `1` | `Found 39 errors` (`src/`: 0, `scripts/`: 33, `tests/`: 6) | CI chi chay tren `src/` nen bo sot toan bo `39` loi lint |
| Quet `60` file `.ipynb` ngoai `.git` va `repos/` | `0` | Tat ca notebook chinh va Colab runner deu dinh URL clone sai | Chi `5` notebook ban sao ThinDrive/BonusDemo dung `drive.mount` |
| `git remote -v` | `0` | `origin https://github.com/khoinguyen59/medical-wsss.git` | Khop voi repo that nhung bi ma hoa sai trong Colab runners |

Chi tiet phan bo `39` loi `ruff`: `23` loi trong `scripts/archive/`, `3` loi trong [build_drive_package.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/scripts/build_drive_package.py), `3` loi trong [build_t4_l4_notebook_suites.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/scripts/build_t4_l4_notebook_suites.py), `3` loi trong `scripts/download_papers.py`, `1` loi trong `scripts/generate_thin_colab_notebooks.py`, `1` loi trong `tests/test_clinical_data_qa.py`, `1` loi trong [test_governance_invariants.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/tests/test_governance_invariants.py), `3` loi trong [test_scientific_upgrades.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/tests/test_scientific_upgrades.py), `1` loi trong `tests/test_wsi_pipeline_extension.py`.

---

## 2. Giai ma Hien tuong "316 Tests Xanh Nhung Du An Hong Nang" (7 Diem Mu cua Bo Test)

Ly do cot loi khien `316` bai test trong `tests/` van PASS du ma nguon va notebook co loi nghiem trong nam o 7 diem mu duoi day:

1. **Test ma hoa nguoc lai chinh bug vao dieu kien PASS:** Trong [test_training_service.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/tests/test_training_service.py#L89-L112), line `112` khang dinh hanh vi sai `extract_patient_id("GDPH_1234_patch_01.png") == "GDPH"` la dung, con lines `89-90` tao file gia lap `val` co nhan trong ngoac vuong `"TCGA-AN-A03X_1024_512_[1 0 0 0].png"` khac voi anh BCSS `val/img` thuc te.
2. **Test runner Colab o che do cuc bo (`IN_COLAB=False`) va `monkeypatch` Engine gia:** [test_scientific_upgrades.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/tests/test_scientific_upgrades.py) chay [SE374_Colab_Scientific_Research_Runner.ipynb](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/SE374_Colab_Scientific_Research_Runner.ipynb) voi `IN_COLAB=False` (ne khoi lenh `git clone` hong o lines `48-54`) va thay model that bang `DummySegmentationModel`, `DummyWSSSEngine`, cung `monkeypatch` tren `dl_mod`/`ml_mod`.
3. **Bo quen cac ham loi trong `evaluator.py`:** [test_scientific_upgrades.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/tests/test_scientific_upgrades.py) khong bao gio goi `run_multiscale_baseline_comparison` hay `run_pda_ablation_study` trong [evaluator.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/colab_pkg/evaluator.py#L1981-L2083), nen loi `AttributeError` tren `patch_mious` chua tung bi kich hoat.
4. **Chi kiem thu MedSAM tren `SurrogateImageEncoder` (`checkpoint_path=None`):** Ca [test_sam_wsss_adapter.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/tests/test_sam_wsss_adapter.py) lan [test_colab_pro_notebooks.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/tests/test_colab_pro_notebooks.py) deu khoi tao `MedSAM_WSSS_Adapter(checkpoint_path=None)` hoac `use_surrogate_if_missing=True`, khong bao gio kiem tra duong nap `sam_model_registry["vit_b"]` voi anh `224x224`.
5. **Kiem tra notebook bang `ast.parse` va so chuoi don gian:** [test_colab_pro_notebooks.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/tests/test_colab_pro_notebooks.py#L25-L39) chi phan tich cu phap AST va kiem tra chuoi nhu `"DeepLabV3Plus" in content`, `"PM_bn7" in content`, `"drive.mount" not in content` tren 13 notebook (`notebooks/01_*` den `04_*` va file root `02_Pseudo_Masks_Train_Stage2_T4.ipynb`), bo sot cac file root `01/03/04_T4` cung nhu xung dot kien truc khi import `DeepLabV3Plus`.
6. **Chi xac minh tensor shape va dictionary keys:**
   - [test_scientific_upgrades.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/tests/test_scientific_upgrades.py) truyen `dummy_deep_feats` khop san `(32, 32)` cho `StainGuidedAffinityRefiner` (che lap loi mismatch `(224, 224)` vs `(28, 28)`), va chi check dict key cua `ComparativeAblationEvaluator`.
   - [test_clinical_service.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/tests/test_clinical_service.py#L442-L444) chi kiem tra su ton tai cua key `"mIoU_4cls"`.
   - [test_api_server.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/tests/test_api_server.py) chi kiem tra HTTP 200, kich thuoc `[224, 224]`, Otsu blank glass gating, va truong `"loaded"` bi hardcode `True`.
7. **Cong CI va Audit bao cao qua mong:** [ci.yml](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/.github/workflows/ci.yml#L1-L57) chi chay `ruff check src/`, quet 5 chuoi don gian bang `grep`, va chi chay `deliverable_auditor` tren duy nhat [SE374_MINI_PROPOSAL_WSSS_WSI.md](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/product/proposal/SE374_MINI_PROPOSAL_WSSS_WSI.md), bo qua toan bo file trong `product/reports/`.

---

## 3. Nhom A: Loi Gay Crash Tuc Thi Hoac Hong Pipeline Tren Colab (CRITICAL)

### A1. Hardcode URL `git clone` Khong Ton Tai Trong Toan Bo Colab Notebooks & Builders

- **Vi tri:**
  - [SE374_Colab_Scientific_Research_Runner.ipynb](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/SE374_Colab_Scientific_Research_Runner.ipynb#L48-L54) (`check=True`, crash ngay khi `IN_COLAB=True`).
  - [build_t4_l4_notebook_suites.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/scripts/build_t4_l4_notebook_suites.py#L470-L471) (va lines `2100-2101`, `2186-2187`, `2290-2291`, `2376-2377`, `check=False`).
  - [build_drive_package.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/scripts/build_drive_package.py#L193).
  - Qua quet toan bo `60` file `.ipynb` ngoai `.git`/`repos/`:
    * `5` file ban sao `SE374_Colab_WSI_LowMag_HighMag_BonusDemo.ipynb` va `SE374_Colab_WSSS_{A100_Full4986,L4_Full4986,T4_Full4986,T4_Sanity150}_ThinDriveRunner_v2.ipynb` (o `notebooks/`, `notebooks/modular_drive/`, `drive_upload_ready/`) vua dung `drive.mount` vua chua URL `medical-ai-consortium/bcss-wsss.git`.
    * `SE374_Colab_Scientific_Research_Runner.ipynb` (root & `notebooks/`), `notebooks/SE374_Colab_Scientific_Research_Runner_FIXED.ipynb`, 12 notebook `notebooks/01_*` den `04_*` (cac bien the base, `_L4`, `_T4`), va 4 notebook root `01_*_T4.ipynb` den `04_*_T4.ipynb` deu chua URL `medical-ai-consortium/bcss-wsss.git`.
- **Nguyen nhan & Tac dong:** Git remote thuc te cua du an la `https://github.com/khoinguyen59/medical-wsss.git`. Khi chay tren Colab ma chua co san thu muc code tren Drive, lenh clone repo gia `https://github.com/medical-ai-consortium/bcss-wsss.git` bi fail (`check=True` gay dung chuong trinh ngay lap tuc; `check=False` im lang roi chay tiep voi `Path.cwd()` khong co `src/`).

### A2. Nham Lan Hai Kien Truc `DeepLabV3Plus` Khac Nhau Giua Stage 2 va Stage 4 + Tai Checkpoint Im Lang

- **Vi tri:**
  - [deeplabv3plus.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/models/deeplabv3plus.py#L509-L550) va [line 723](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/models/deeplabv3plus.py#L723): `DeepLabV3Plus = DeepLabV3PlusWithCGM` dung `ResNetBackbone` (`512` channels), `DeepLabV3PlusDecoder` (`project_low`, `refinement`, `classifier`).
  - [wsss_inference_methods.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/models/wsss_inference_methods.py#L301-L357) (va [model_loader.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/colab_pkg/model_loader.py#L595-L625)): `DeepLabV3Plus` kieu `WSSS-Tissue` dung `ResNet101` + `ASPP` + `Decoder` (`decoder.last_conv.8.weight`, `59,339,940` tham so).
  - [build_t4_l4_notebook_suites.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/scripts/build_t4_l4_notebook_suites.py#L807-L814) (Stage 2) vs [lines 1749-1800](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/scripts/build_t4_l4_notebook_suites.py#L1749-L1800) (Stage 4), cung toan bo `8` notebook `02_*` va `04_*` (root `T4` va `notebooks/`).
- **Nguyen nhan & Tac dong:** Stage 2 uu tien `from src.models.wsss_inference_methods import DeepLabV3Plus` (`ResNet101`) de huan luyen va luu `stage2_retrained_best.pth`. Nhung den Stage 4, cell nap model lai uu tien `from src.models.deeplabv3plus import DeepLabV3Plus` (`DeepLabV3PlusWithCGM` / `ResNetBackbone`), khoi tao ca Model A (tac gia), Model B (retrained), va fallback Model C bang `DeepLabV3PlusWithCGM`! Dong thoi `line 1758` chi doc `ckpt_a.get('model', ckpt_a)` (bo sot key `'state_dict'` ma `build_and_load_deeplabv3plus` o `wsss_inference_methods.py:386` dung) va nap bang `strict=False`. Hau qua: decoder va cac tang khong khop bi bo qua im lang, Stage 4 danh gia trong so ngau nhien tren Model A va Model B.

### A3. Crash `AttributeError` (`patch_mious`) va Phong Dai Ket Qua 100 Lan Trong `evaluator.py`

- **Vi tri:** [evaluator.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/colab_pkg/evaluator.py#L434), [line 467](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/colab_pkg/evaluator.py#L467), [lines 1981-1998](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/colab_pkg/evaluator.py#L1981-L1998), va [lines 2064-2083](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/colab_pkg/evaluator.py#L2064-L2083).
- **Nguyen nhan & Tac dong:** `ColabSegmentationEvaluator` chi khai bao thuoc tinh `self.patch_miou_4cls`, khong co `self.patch_mious`. Hai ham `run_multiscale_baseline_comparison` (`lines 1981-1983`) va `run_pda_ablation_study` (`lines 2064-2066`) goi `eval_*.patch_mious` gay `AttributeError` ngay lap tuc. Ngoai ra, `patch_miou_4cls` da nhan `100.0` tai `line 467` (da la `%`), nhung `lines 1992, 1998, 2082, 2083` lai nhan them `* 100.0`, dan den chenh lech phan tram bi phong dai 100 lan neu doi ten thuoc tinh ma khong bo `* 100.0`.

### A4. Sai Ten File Checkpoint & State-Dict Key cua MedSAM Giua Train va Eval

- **Vi tri:** [train_proposed_sam.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/training/train_proposed_sam.py#L397) va [line 424](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/training/train_proposed_sam.py#L424) so voi [exhaustive_evaluator.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/clinical/exhaustive_evaluator.py#L566) va [line 690](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/clinical/exhaustive_evaluator.py#L690).
- **Nguyen nhan & Tac dong:** `train_proposed_sam.py` luu checkpoint tai `"medsam_wsss_adapter_best.pth"` voi dict key `"model_state_dict"`. Nguoc lai, `exhaustive_evaluator.py` mac dinh tim `"checkpoints/medsam_adapter_best.pth"` va doc `state.get("model", state.get("state_dict", state))`. Khi dung `exhaustive_evaluator.py`, checkpoint MedSAM bi bo qua hoac tra ve wrapper dict lam sai lech nap trong so.

### A5. Crash Shape Khi Nap Backbone SAM/MedSAM `vit_b` That Tren Anh `224x224`

- **Vi tri:** [sam_wsss_adapter.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/models/sam_wsss_adapter.py#L752) va [lines 814-816](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/models/sam_wsss_adapter.py#L814-L816).
- **Nguyen nhan & Tac dong:** Khi truyen `checkpoint_path` that vao `MedSAM_WSSS_Adapter`, he thong nap `sam_model_registry["vit_b"](checkpoint=checkpoint_path)` (yeu cau dau vao `1024x1024` voi `pos_embed` luoi `64x64`), nhung `forward()` day thang `images` `(B, 3, 224, 224)` vao `self.image_encoder(images)` ma khong resize anh len `1024x1024` hay noi suy `pos_embed`. Chay thuc te voi weights SAM/MedSAM se crash ngay lap tuc.

### A6. `api_server.py` Luon Khong Nap Trong So Nhung Tra Ve `"loaded": True`

- **Vi tri:** [api_server.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/visualization/api_server.py#L81), [line 96](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/visualization/api_server.py#L96), va [line 271](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/visualization/api_server.py#L271) so voi [wsss_tissue.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/models/wsss_tissue.py#L160).
- **Nguyen nhan & Tac dong:** `api_server.py` khoi tao `WSSSBenchmarkingEngine` (chi co thuoc tinh `self.classifier`, `self.pda`, `self.relocalizer`) roi tim `getattr(engine, "stage1_model", None) or getattr(engine, "model", None)`. Gia tri luon la `None`, khong tai bat ky checkpoint nao, nhung endpoint health van tra ve `"loaded": True`.

### A7. Crash Unfold Khi `deep_feats` va `init_cam` Khac Kich Thuoc Khong Gian Trong `sg_dap.py`

- **Vi tri:** [sg_dap.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/models/sg_dap.py#L75-L86).
- **Nguyen nhan & Tac dong:** `StainGuidedAffinityRefiner.forward()` chi resize `rgb_img` ve `(h, w)` cua `init_cam`, nhung khong resize `deep_feats` truoc khi chay `F.unfold(norm_deep_feats, ...).view(b, -1, k2, h * w)`. Neu `init_cam` o `(224, 224)` va `deep_feats` tu Stage 1 o `(28, 28)`, phep `.view(..., h * w)` bi `RuntimeError`.

### A8. `generate_pseudo_masks.py` Dung Anh RGB 3 Kenh Gia Lam `deep_feats` & Ghi De `255` Len Lop Background `4`

- **Vi tri:** [generate_pseudo_masks.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/training/generate_pseudo_masks.py#L360-L374).
- **Nguyen nhan & Tac dong:** Script noi suy `raw_img_tensors` thanh `feats_proxy` (chi 3 kenh RGB thay vi deep feature map tu backbone) de dua vao `sg_dap_refiner`, roi gan `gated_np[sg_np == 255] = 255` sau buoc Otsu gating, xoa luon ca nhan nen kinh lop `4` o nhung vung Otsu da xac dinh la background.

---

## 4. Nhom B: Loi Sai Khoa Hoc Nghiem Trong (Lam Meo So Lieu Va Mat Gia Tri Doi Chung)

### B1. Nham Lan Chuan Hoa Dau Vao (`ImageNet Mean/Std` vs `[0, 1]`) Cho Checkpoint Tac Gia Chu Han (2022)

- **Vi tri:** Repo tac gia [1_train_stage1.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/repos/WSSS-Tissue/1_train_stage1.py#L30-L32) va [infer_fun.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/repos/WSSS-Tissue/tool/infer_fun.py#L99); engine SOTA [wsss_inference_methods.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/models/wsss_inference_methods.py#L1494-L1495); doi lap voi [generate_pseudo_masks.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/training/generate_pseudo_masks.py#L324), [exhaustive_evaluator.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/clinical/exhaustive_evaluator.py#L755), va [build_t4_l4_notebook_suites.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/scripts/build_t4_l4_notebook_suites.py#L149) (va lines `740, 1577`).
- **Nguyen nhan & Tac dong:** Code goc `WSSS-Tissue` chi dung `transforms.ToTensor()` (`[0, 1]`, khong tru ImageNet mean/std). `wsss_inference_methods.py` cung chuan hoa ve `[0, 1]`. Nhung `generate_pseudo_masks.py`, `exhaustive_evaluator.py`, va cac notebook Stage 1/2/4 lai tru `IMAGENET_MEAN`/`IMAGENET_STD`, lam sai lech hoan toan phan phoi dau vao cua checkpoint tac gia Chu Han (2022) va khong nhat quan voi `MultiMethodWSSSEngine`.

### B2. Hardcode 4 Kenh Trong `MultiMethodWSSSEngine` Lam Bo Sot Kenh 4 Khi `num_classes=5`

- **Vi tri:** [wsss_inference_methods.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/models/wsss_inference_methods.py#L1016) (cung lines `1205, 1263, 1367-1368, 1405-1406`) va ban sao trong [model_loader.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/colab_pkg/model_loader.py#L1080) (cung lines `1255, 1308, 1397-1398, 1425-1426`).
- **Nguyen nhan & Tac dong:** Cac ham ensemble/CGM/refinement hardcode kich thuoc `4` (`np.zeros((4, H, W))`, `cgm_gate.reshape(4, 1, 1)`, `calibrated = probs_4cls * w`, `refined_probs = np.empty_like(probs_4cls)` chi dien `for c in range(4):`). Khi `probs_4cls` co 5 kenh (`num_classes=5`), kenh thu 5 (`Background`) hoac gay loi broadcast shape hoac giu gia tri rac chua khoi tao trong `np.empty_like`.

### B3. Bat Nhat Quan Prompt Giua Huan Luyen va Danh Gia Tren MedSAM

- **Vi tri:** [train_proposed_sam.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/training/train_proposed_sam.py#L285-L294) vs [line 357](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/training/train_proposed_sam.py#L357) va [exhaustive_evaluator.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/clinical/exhaustive_evaluator.py#L778); cung voi [build_t4_l4_notebook_suites.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/scripts/build_t4_l4_notebook_suites.py#L1462-L1481) (Stage 3 train) vs [line 1643](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/scripts/build_t4_l4_notebook_suites.py#L1643) (Stage 4 eval).
- **Nguyen nhan & Tac dong:** Trong `train_proposed_sam.py`, luc train ghep prompt cua ca 4 lop `(B, 24, 2)` dua vao `model(...)`, nhung luc `evaluate()` va trong `exhaustive_evaluator.py` lai goi `model(images)` voi `point_coords=None, point_labels=None`. Trong notebook Stage 3, `build_t4_l4_notebook_suites.py` chi trich 1 lop `primary_c` de huan luyen multi-class `num_classes=5`, nhung Stage 4 cung goi `model_medsam(img_t)` khong co bat ky prompt nao.

### B4. Fallback Im Lang Sang Du Lieu Gia, Mask Toan Lop `4`, Hoac Lop `0` (Tumor)

- **Vi tri:**
  - [train_proposed_sam.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/training/train_proposed_sam.py#L540) va [lines 586-595](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/training/train_proposed_sam.py#L586-L595): fallback ve `SyntheticHistologyDataset` ke ca khi `dry_run=False`.
  - [exhaustive_evaluator.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/clinical/exhaustive_evaluator.py#L718-L720) va [line 767](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/clinical/exhaustive_evaluator.py#L767): fallback `img_dir = mask_dir = self.dataset_dir / "test"`, va neu thieu mask thi tra ve ma tran toan so `4` (Background).
  - [bcss_dataset.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/datasets/bcss_dataset.py#L50-L162): chi `glob("*.png")` nong, khong kiem tra dataset rong, va neu thieu mask `val`/`test` thi tra ve `torch.zeros((h_t, w_t))` (tuc gan toan bo anh thanh lop `0` - Tumor).

### B5. Bat Dong Nhat `fn_to_bg_4cls` Giua Global mIoU va Per-Patch / Per-Patient mIoU

- **Vi tri:** [evaluator.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/colab_pkg/evaluator.py#L510-L527) vs [line 623](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/colab_pkg/evaluator.py#L623).
- **Nguyen nhan & Tac dong:** Khi tinh chi so toan cuc trong `compute_metrics()` (`line 623`), `fn4` cong them `self.fn_to_bg_4cls` (cac pixel foreground bi du doan nham sang lop 4 Background). Tuy nhien khi luu `patch_miou_4cls` (`lines 510-511`) va `patient_cm_4cls` (`line 527`), `_compute_4cls_miou_mdice_from_cm(counts_4)` lai khong cong `fn_to_bg_4cls`, khien ket qua Wilcoxon theo patch va mIoU theo benh nhan cao hon thuc te.

### B6. Cat Lat Trong So `fc_weights[:, :feat.shape[1]]` Khi Trich Xuat CAM Da Tang & Gia Mao 3 Tang Trong Notebook

- **Vi tri:** [backbone.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/models/backbone.py#L211-L214), [pda_generator.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/models/pda_generator.py#L106-L107), [generate_pseudo_masks.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/training/generate_pseudo_masks.py#L196-L201), va [build_t4_l4_notebook_suites.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/scripts/build_t4_l4_notebook_suites.py#L757-L791).
- **Nguyen nhan & Tac dong:** Trong `backbone.py`, `pda_generator.py`, va `generate_pseudo_masks.py`, chieu kenh cua tang nong (`b5_2`, `b4_5`, vd `512`/`1024`) khong khop voi chieu vao cua lop linear cuoi (`4096`), nhung code lay truc tiep `weight[:, :feat_channels, :, :]` de nhan chap. Dieu nay vo nghia ve mat bieu dien dac trung. Rieng trong Stage 2 notebook (`build_t4_l4_notebook_suites.py:757-791`), notebook chi goi `model_stage1.forward_cam(img_tensor)` mot lan duy nhat roi luu cung `refined_cam` vao ca 3 thu muc `PM_bn7`, `PM_b5_2`, `PM_b4_5` chi bang cach doi nguong (`0.20`, `0.25`, `0.15`).

### B7. Cat Sai `Patient ID`, Gan Nhan Mac Dinh Cho Anh Khong Ngoac Vuong, va Quet De Quy Ro Ri Du Lieu

- **Vi tri:** [patient_split.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/datasets/patient_split.py#L14-L45) va [dataset_loader.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/training/dataset_loader.py#L55-L215).
- **Nguyen nhan & Tac dong:**
  - `re.split(r"[_+]", fname)[0]` cat ten file `"GDPH_1234_patch_01.png"` thanh `"GDPH"`, gop chung tat ca benh nhan cua benh vien GDPH thanh 1 benh nhan duy nhat (vi pham nguyen tac tach theo benh nhan - Invariant 08); con anh WSSS4LUAD dung dau `-` bi giu ca chuoi dai lam ID, phan manh 1 benh nhan thanh nhieu patch doc lap.
  - Anh khong co nhan ngoac vuong `[a b c d]` bi `dataset_loader.py` gan mac dinh `multi_label[0] = 1.0`, va ham `find_image_files(self.root_dir)` mac dinh `recursive=True` co the quet xuyen thu muc con `train`/`val`/`test`.

### B8. `ComparativeAblationEvaluator` Chay Ablation Tren Lop Ngau Nhien Chua Huan Luyen, Goi Lap Module & Thieu Phep Lat D4

- **Vi tri:** [comparative_ablation_evaluator.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/clinical/comparative_ablation_evaluator.py#L53-L70), [lines 89-92](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/clinical/comparative_ablation_evaluator.py#L89-L92), [lines 131-135](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/clinical/comparative_ablation_evaluator.py#L131-L135), [lines 184-185](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/clinical/comparative_ablation_evaluator.py#L184-L185), va [lines 238-243](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/clinical/comparative_ablation_evaluator.py#L238-L243).
- **Nguyen nhan & Tac dong:**
  1. Khoi tao moi hoan toan `StainGuidedAffinityRefiner`, `MultiPrototypePPCLoss`, `IntraPatchDualBranchAttention`, va `self.feat_cam_head = nn.Conv2d(feat_dim, 4, 1)` ma khong co bat ky co che tai checkpoint da huan luyen nao.
  2. Trong `_apply_intra_patch_attention_refinement` (`lines 131-135`), ngay ca khi `gamma == 0`, ham van cong `+ 0.25 * self.feat_cam_head(enhanced_feats)` tu lop `Conv2d` khoi tao ngau nhien, lam nhieu ket qua M9 ngay o trang thai identity.
  3. O che do `m9`/`m_combined` (`lines 184-185`), `self.intra_attn` bi goi lap 2 lan lien tiep.
  4. Phuong thuc xoay chi dung `torch.rot90` (4 goc xoay nhom C4, thieu 4 phep lat doi xung cua D4).
  5. Fallback `rgb.repeat(...)` gia lam `backbone_feats` 64 kenh neu thieu feature map that.

### B9. Cac Loi Toan Hoc & Thiet Ke Khoa Hoc Khac Trong Huan Luyen, Otsu, Routing va Notebook

- **Otsu Filter tren tensor da chuan hoa:** [otsu_filter.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/datasets/otsu_filter.py#L35-L70) kiem tra `if image.max() <= 1.0:` roi fallback sang `np.clip(image, 0, 255).astype(np.uint8)`. Tensor sau ImageNet normalization co `max` khoang `2.64 > 1.0`, bi ep kieu `uint8` thanh gia tri `{0, 1, 2}`, lam nguong Otsu hoan toan sai.
- **Min-Max Stretching pha huy gia tri tuyet doi cua CAM:** [wsss_tissue.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/models/wsss_tissue.py#L160-L265) bo qua `logits` tu `self.classifier` va keo dan min-max ve `[0, 1]` tren moi kenh lop, khuech dai nhieu o lop vang mat trong patch.
- **AsymmetricLoss & chon checkpoint Stage 1 sai:** [train_stage1.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/training/train_stage1.py#L85-L135) bo qua tham so `disable_torch_grad_focal_loss`, dung `probs.pow(gamma_neg)` thay vi `asym_probs.pow(gamma_neg)`, va chon checkpoint tot nhat dua tren `val_macro_f1` gia.
- **CoarseToFineAdaptiveRouter du thua forward va lech latency:** [coarse_to_fine_router.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/models/coarse_to_fine_router.py#L259-L265) va [line 341](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/models/coarse_to_fine_router.py#L341) chay `engine.predict_m6` xong lai goi `logits_10 = model(t10)` them lan nua, va cong thuc `effective_latency = coarse_lat + (beta * fine_lat)` van tinh phi `coarse_lat` ngay ca khi `beta = 1.0`.
- **Sai lech trong kich ban sinh 4 Notebook (`build_t4_l4_notebook_suites.py`):**
  - Stage 1 (`lines 217-218`): PDA dropout chi nhan `torch.bernoulli(keep_prob)` khong bu he so `1 / keep_prob`; `lines 236-349` chon `STAGE1_BEST_CKPT` theo do chinh xac tren tap **train**, khong co tap validation.
  - Stage 2 (`line 1018`): ghi de `STAGE2_BEST_CKPT` vo dieu kien sau moi epoch, khong he cap nhat `best_miou_s2`.
  - Stage 3 (`line 1514`): chon `MEDSAM_BEST_CKPT` dua tren train `avg_loss` thap nhat.
  - Stage 4 (`lines 1571, 1665-1671, 1896-1897`): `apply_otsu_gating_eval` chi dung nguong cung `sat < 0.08` (khong phai Otsu), `patch_ious_a`/`patch_ious_c` tinh IoU gop toan foreground (`gt_np < 4`) thay vi 4-class macro mIoU va bo qua Model B / A2, va phai gan tay `km_engine.log_rank_test = km_engine.logrank_test`.

---

## 5. Nhom C: Loi Ky Thuat Ha Tang, Kien Truc & Chat Luong Ma Nguon (HIGH / MEDIUM)

1. **Trung lap hon 1.800 dong code giua 2 module loader:** [wsss_inference_methods.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/models/wsss_inference_methods.py) (`1.824` dong) va [model_loader.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/colab_pkg/model_loader.py) (`1.831` dong) sao chep y nguyen ca logic lan bug 4-kenh, gay lech dong bo khi sua loi mot noi.
2. **Pytest root khong chay duoc (`728` errors, timeout `80.51s`):** Thieu cau hinh loai tru `repos/` (trong `pytest.ini` hoac `pyproject.toml`), khien `python -m pytest` thu thap nham `MONAI`, `OpenMidnight`, `PistoSeg`, `WSInfer`, `meta_tools` va dinh loi `triton` trong 10 test cua du an.
3. **39 loi Lint (`ruff`) trong `scripts/` va `tests/`:** Da liet ke day du o Muc 1.1; `.github/workflows/ci.yml` chi kiem tra `src/` nen khong chan duoc cac loi nay.
4. **Nham lan tin hieu `fano_factor` va `stain_grad` trong Stage 2:** [stain_deconv.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/models/stain_deconv.py#L114-L116) tra ve `(he_channels, fano_factor, stain_features)` (`fano_factor` la chi so phan tan chromatin nhan D_H), nhung [train_stage2.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/training/train_stage2.py#L140-L175) giai nen thanh `_, stain_grad, _ = self.stain_deconv(img_01)`.
5. **Phan manh 60 file notebook:** Ton tai dong thoi 4 ban root `01_*_T4.ipynb` den `04_*_T4.ipynb`, 12 ban trong `notebooks/`, cac ban `ThinDriveRunner_v2`/`BonusDemo` trong `notebooks/modular_drive/` va `drive_upload_ready/`, trong khi `test_colab_pro_notebooks.py` chi kiem tra `13` file.

---

## 6. Nhom D: Loi Sai Tai Lieu & San Pham Bao Cao (Deliverables & Compliance)

1. **Bao cao cuoi ky [SE374_FINAL_TECHNICAL_REPORT_WSSS.md](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md):**
   - Lines `150-156` nhung anh bang duong dan tinh tu root `product/proposal/assets/fig*.png`, khien bi vo link khi mo tu `product/reports/` (dung phai dung duong dan tuong doi sang thu muc `proposal/assets/` thay vi bat dau bang `product/`).
   - Lines `189-194` bao cao bang sanity voi `mIoU 4.32%-6.32%` tu trong so chua huan luyen.
   - Lines `290-306` dung `mermaid` kieu `gantt` (khong nam trong cac loai bieu do Mermaid duoc ho tro).
   - Mau thuan noi bo so luong test: line `34` ghi `316 passing tests`, nhung lines `253, 277, 337` ghi `311 passing tests`.
2. **Bao cao lam sang [EXHAUSTIVE_3WAY_BENCHMARK_REPORT.md](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/product/reports/clinical/EXHAUSTIVE_3WAY_BENCHMARK_REPORT.md):**
   - Ghi ket qua hong: `Model A: 0.00%`, `t = inf, p = 0.0000e+00`, `Cohort Size: 1 patients (1 Stroma-Rich, 2 Stroma-Poor)` (lines `21, 46, 52, 66`).
   - Line `70` tham chieu `bland_altman_tsr_vtp.png`, nhung file nay khong ton tai trong `product/reports/clinical/`.
3. **So do quy trinh & De cuong Proposal:**
   - [draw_pipeline_diagram.py](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/src/visualization/draw_pipeline_diagram.py#L45-L208) su dung nhieu mau sac so (`#D53F8C`, `#805AD5`, `#E53E3E`, `#F0FFF4`, `#38A169`, `#2B6CB0`, `#9F7AEA`, `#DD6B20`), vi pham chuan don sac / thang xam hoc thuat (Invariant 24).
   - [SE374_MINI_PROPOSAL_WSSS_WSI.md](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/product/proposal/SE374_MINI_PROPOSAL_WSSS_WSI.md#L4-L20) con placeholder `[Họ và tên Sinh viên 1]` (can hoi xac nhan thong tin hanh chinh truoc khi nop theo Invariant 20).

---

## 7. Lo Trinh Khac Phuc De Xuat Theo 4 Giai Doan (Cho Phe Duyet Truoc Khi Sua)

1. **Giai doan 1 - Khoi phuc Kien truc & Chay Duoc Tren Colab (Nhóm A):**
   - Dong nhat kien truc `DeepLabV3Plus` (`WSSS-Tissue` vs `DeepLabV3PlusWithCGM`), sua thu tu import va key checkpoint trong `build_t4_l4_notebook_suites.py` va sinh lai toan bo notebook dong bo.
   - Sua toan bo URL `git clone` sang `https://github.com/khoinguyen59/medical-wsss.git` va bo fallback nguy hiem.
   - Sua `patch_mious` -> `patch_miou_4cls` (bo `* 100.0` thua), dong bo ten file & key `medsam_wsss_adapter_best.pth` (`model_state_dict`), them resize `1024x1024` / noi suy `pos_embed` cho `sam_wsss_adapter.py`, sua `sg_dap.py` resize `deep_feats`, va sua logic nap model that trong `api_server.py`.
2. **Giai doan 2 - Chuan hoa Khoa hoc & Loai bo Nhieu So Lieu (Nhom B):**
   - Chuan hoa dung phan phoi dau vao: `[0, 1]` cho model goc `WSSS-Tissue` Stage 1 & Stage 2, tach bach transform voi model dung ImageNet; sua `otsu_filter.py`, `wsss_tissue.py`, va `train_stage1.py`.
   - Loai bo hardcode `4` kenh trong `wsss_inference_methods.py` va `model_loader.py` (hop nhat thanh 1 nguon duy nhat).
   - Dong bo prompt MedSAM giua train va eval, loai bo fallback sang du lieu gia / mask toan 0 hoac 4 khi `dry_run=False`.
   - Them tap `val` tach theo Patient ID chuan (sua `extract_patient_id` cho `GDPH` va `WSSS4LUAD`) de chon `BEST_CKPT` trong Stage 1/2/3 thay vi chon tren train metric hoac ghi de vo dieu kien.
   - Sua `ComparativeAblationEvaluator` (bo cong logits ngau nhien khi `gamma == 0`, xoa goi lap `intra_attn`, bo sung du 8 phep D4, ho tro nap weights that, va dung feature map that thay cho `rgb.repeat`) va dong nhat `fn_to_bg_4cls` trong `evaluator.py`.
3. **Giai doan 3 - Ha tang Kiem thu & CI/CD (Nhom C + 7 Diem Mu):**
   - Cau hinh `norecursedirs = repos ...` cho `pytest`, sua 39 loi `ruff`, mo rong `ci.yml` kiem tra ca `src/`, `tests/`, `scripts/` va chay `deliverable_auditor` tren toan bo file `.md` trong `product/`.
   - Viet lai bo test that (khong mock bug, kiem tra `strict=True` khi load state dict, kiem tra shape khop that cho MedSAM & SG-DAP, kiem tra toan bo cac ham trong `evaluator.py`).
4. **Giai doan 4 - Lam sach Tai lieu & San pham Nop (Nhom D):**
   - Chuyen `draw_pipeline_diagram.py` sang chuan don sac/thang xam hoc thuat (300 DPI / SVG), sua duong dan anh va loai bo bang sanity chua huan luyen/Gantt Mermaid khoi `SE374_FINAL_TECHNICAL_REPORT_WSSS.md`, xoa/tai tao `EXHAUSTIVE_3WAY_BENCHMARK_REPORT.md` tu ket qua chay that sau khi Colab hoan tat.
