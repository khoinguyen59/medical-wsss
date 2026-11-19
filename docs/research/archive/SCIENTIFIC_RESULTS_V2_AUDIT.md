# Audit ket qua `SE374_Scientific_Research_Results_v2.zip`

**Ngay kiem toan:** 2026-10-24
**Nguon:** `SE374_Scientific_Research_Results_v2.zip` sinh boi `SE374_Colab_Scientific_Research_Runner.ipynb`
**Trang thai artifact:** `sanity_dry_run = true`, `device = cuda`
**Ket luan tong the:** KHONG DU DIEU KIEN BAO CAO. Day la dry-run de kiem tra pipeline, khong phai ket qua nghien cuu.

---

## 0. Vi sao khong dung duoc

`SE374_Colab_Scientific_Research_Runner.ipynb` dat `SANITY_DRY_RUN = True` va `DRY_RUN_SAMPLES = 50`.
Hau qua truc tiep len tung thi nghiem:

| Tham so | Gia tri dry-run | Gia tri can cho bao cao |
| :--- | :--- | :--- |
| Exp 1 test patch | 46 patch / 23 benh nhan | 4.986 patch / 106 benh nhan |
| Exp 2 mau | 46 patch | 200+ patch |
| Exp 3 benh nhan | 23 benh nhan, 2 patch/BN | 150 patch, nhieu patch/BN |
| Exp 4 LUAD | 19-20 patch | toan bo test split |
| Exp 5 Whole-ROI | 1 ROI, cap 50 patch | toan bo ROI, khong cap |

Voi 46 patch tren 23 benh nhan, moi benh nhan chi dong gop 2 lat cat. Do lech chuan mIoU theo benh nhan do duoc la **20,63 diem** (xem `test_metrics.patient_miou_std`). Sai so chuan cua trung binh la `20,63 / sqrt(23) = 4,30` diem. Khoang tin cay 95% la `65,17 +- 8,4` diem. Khong the ket luan bat cu dieu gi tu con so nay.

---

## 1. Exp 1 - Validation-to-Test Protocol

**File:** `val_to_test/val_to_test_report.json`

| Chi so | Val (44 patch) | Test (46 patch) |
| :--- | ---: | ---: |
| mIoU (4 lop) | 66,18 | 65,17 |
| mDice (4 lop) | 79,54 | 76,99 |
| FwIoU | 61,00 | 72,24 |
| Pixel Accuracy | 80,45 | 86,96 |
| Boundary F1 | 41,50 | 44,73 |

**Per-class IoU tren test:** TUM 81,65 - STR 76,79 - LYM 69,82 - NEC 32,43 - BACK 1,85

**Generalization gap:** `+1,01` -> `protocol_passed = true`

### Danh gia: TOT VE PHUONG PHAP, KHONG DUNG DUOC VE SO LIEU

Diem manh that su cua thi nghiem nay la **giao thuc**, khong phai con so:

- Tham so `min_island_area` duoc quet tren **val** (16 / 32 / 64) roi dong bang truoc khi cham **test**. Day la quy trinh dung chuan, khong ro ri tham so vao test.
- Gap val-test chi 1,01 diem -> khong overfit vao val.
- Khoang tin cay rong (xem muc 0) khien 65,17% khong mang y nghia thong ke.

**Van de phu:** `val_tuning_history` cho ca 3 gia tri 16/32/64 deu ra **dung mot con so 66,1808...**. Dieu nay chung to nguong `min_island_area` khong he anh huong den ket qua tren mau val nay - hoac phep loc dao khong duoc kich hoat, hoac khong co dao nao nho hon 64 pixel trong mau val. Can kiem tra lai; neu filter khong chay thi buoc "tuning" la hinh thuc.

**So sanh voi ket qua that:** Run 3 full 4.986 patch dat **70,53% mIoU 4 lop** (`experiments/colab_run_3_multiscale_sota_v2/COLAB_EVALUATION_REPORT.json`). Dry-run thap hon 5,36 diem, nam gon trong khoang tin cay `65,17 +- 8,4`. Khong co bang chung nao cho thay mo hinh kem di.

**Hanh dong:** Chay lai voi `SANITY_DRY_RUN = False` tren toan bo test split. Sau do moi trich xuat con so cho bao cao.

---

## 2. Exp 2 - Multi-Magnification & Coarse-to-Fine Pareto Router

**File:** `pareto/pareto_results.json`, `pareto/coarse_to_fine_pareto_curve.png`

### 2a. Phan Multi-Magnification: HOP LY

| Do phong dai | mIoU |
| :--- | ---: |
| 5x | 44,48 |
| 10x | 54,51 |
| 20x | 64,34 |
| 40x | 65,17 |

Xu huong don dieu tang theo do phong dai, phu hop voi ban chat vat ly cua bai toan (do phan giai cang cao, chi tiet mo hoc cang ro). Day la phan **dung** cua thi nghiem va co the dung lam hinh minh hoa.

### 2b. Phan Adaptive Router: THAT BAI

| Budget | Adaptive mIoU | Random mIoU | Ket luan |
| ---: | ---: | ---: | :--- |
| 0% | 47,23 | - | - |
| 25% | 45,93 | 47,11 | Random THANG |
| 50% | 45,46 | 48,35 | Random THANG |
| 75% | 49,40 | - | - |
| 100% | 49,37 | - | - |

- `auc_gain = 0,0` - router khong tao ra bat ky loi ich nao so voi duong co so.
- Duong adaptive **khong don dieu**: 47,23 -> 45,93 -> 45,46 -> 49,40 -> 49,37. Tang ngan sach tu 0% len 50% lam mIoU **GIAM** 1,77 diem. Mot router thong minh khong the hanh xu nhu vay.
- Tai ca 2 diem do duoc, **random routing cao hon adaptive routing**.

### 2c. Loi trong bieu do Pareto

Hinh `coarse_to_fine_pareto_curve.png` gan nhan **"Sweet Spot (Budget 25% -> mIoU 45.9%)"**. Nhung 45,93% la diem **thap nhat trong toan bo duong cong** va thap hon ca random (47,11%). Nhan nay sai ve mat khoa hoc va neu dua vao slide se bi phan bien ngay.

### 2d. Loi code trong `src/models/coarse_to_fine_router.py`

1. `auc_gain = max(0.0, ...)` - ham `max` nay che mat moi gia tri am. Khong the phan biet "router te hon random 5 diem" voi "router bang random". Day la loi nghiem trong ve tinh trung thuc cua bao cao.
2. `np.trapezoid` duoc goi tren truc budget dang **phan so 0..1** thay vi phan tram 0..100, nen don vi cua AUC bi lech 100 lan.
3. Khong co phep do do phuc tap nao thuc su duoc dung de quyet dinh patch nao can 40x. Thieu co che nay thi "adaptive" chi la mot phep chon ngau nhien co trong so.

**Hanh dong:** Sua `auc_gain` de tra ve gia tri am that, sua truc tich phan, bo nhan "Sweet Spot", va hoac (a) loai Exp 2b khoi bao cao, hoac (b) hien thuc mot bo do do phuc tap that su (entropy CAM, disagreement multi-scale) truoc khi chay lai.

---

## 3. Exp 3 - Patient-Level Biomarker Correlation

**File:** `biomarkers/biomarker_correlation.json`

| Chi so | Gia tri |
| :--- | ---: |
| So benh nhan | 23 |
| VTP Pearson r | 0,9668 (p = 6,45e-14) |
| VTP Spearman rho | 0,9592 |
| VTP MAE | 6,38 |
| VTP RMSE | 10,35 |
| TSR Pearson r | 0,9697 (p = 2,51e-14) |
| TSR MAE | 6,46 |
| CAP concordance | 100% |

### Danh gia: TIN HIEU TOT, NHUNG CHUA DU CO SO DE BAO CAO

Day la thi nghiem co **tin hieu manh nhat** trong ca 5. Tuong quan r > 0,96 voi p < 1e-13 la con so rat an tuong. Nhung co 3 van de phai xu ly truoc khi dua vao bao cao:

1. **n = 23 benh nhan, 2 patch/benh nhan.** Voi bai toan tuong quan, n = 23 la nho. Mot vong chay lai voi cach chon patch khac co the lam r tut xuong dang ke. Can tang len toi thieu 150 patch trai deu tren toan bo benh nhan test.

2. **Nhieu ca co VTP = TSR.** Dieu nay xay ra khi mo hinh khong du doan duoc lop STR trong vung u. He qua la hai bien gan nhu trung nhau, va viec ca hai cung cho r ~0,97 khong phai la hai bang chung doc lap ma la **mot** bang chung duoc dem hai lan. Can bao cao them so ca co `VTP == TSR`.

3. **CAP concordance 100%** voi n = 23 va chi 3 muc nguy co la con so qua dep de tin. Can bang phan bo nhan CAP thuc te de kiem tra xem co bi mat can bang nghiem trong khong.

**Hanh dong:** Giu thi nghiem nay lam trong tam, nhung phai chay full (150 patch) va bao cao kem phan bo nhan CAP, so ca VTP=TSR, va bieu do Bland-Altman o do phan giai cao.

---

## 4. Exp 4 - "Cross-Dataset LUAD-HistoSeg Generalization"

**File:** `luad/luad_benchmark_metrics.json`, `luad/luad_cross_dataset_report.json`

| Chi so | Gia tri |
| :--- | ---: |
| So patch | 19 patch / 19 benh nhan |
| mIoU (4 lop) | 78,76 |
| mDice (4 lop) | 87,84 |
| Boundary F1 | 27,85 |
| BACK IoU | **0,00** |

**Per-class IoU:** TE 78,30 - NEC 89,23 - LYM 64,97 - TAS 82,56 - BACK 0,00

### Danh gia: SAI BAN CHAT - DAY KHONG PHAI ZERO-SHOT

Day la van de nghiem trong nhat trong toan bo artifact, va la van de **liem chinh hoc thuat**.

Notebook tai:

```python
luad_ckpts = ensure_luad_checkpoints(checkpoint_dir=WORK_ROOT / "checkpoints", ...)
model_luad_s2, model_luad_s1, _ = load_luad_models(
    stage2_ckpt=luad_ckpts.get("stage2"),
    stage1_ckpt=luad_ckpts.get("stage1"),
    ...
)
```

`ensure_luad_checkpoints` trong `src/colab_pkg/downloader.py` (dong 544-613) tai ve dung 2 file:

- `stage1_checkpoint_trained_on_luad.pth`
- `stage2_checkpoint_trained_on_luad.pth`

Sau do chinh 2 checkpoint **da huan luyen tren LUAD** nay duoc dung de danh gia tren tap **test cua LUAD**. Day la **in-domain evaluation**, khong phai chuyen giao cheo co quan (BCSS -> LUAD).

Mo ta trong cell markdown cua notebook ghi "Cross-Dataset LUAD-HistoSeg Generalization" va "zero-shot". Neu nop kem con so 78,76% voi nhan "zero-shot BCSS -> LUAD", do la **bao cao sai su that**. Hau qua theo INVARIANT-07 la 0 diem.

**Van de phu:** `BACK IoU = 0,00` tren ca 4 lop. Mo hinh khong du doan duoc mot pixel background nao. Voi bai toan phan vung mo hoc, day la dau hieu pipeline danh gia dang map nhan sai (co the LUAD dung 4 lop khong co background, hoac thu tu lop khac BCSS).

**Hanh dong (chon 1):**
- **(a)** Doi ten muc thanh "In-domain LUAD-HistoSeg benchmark" va bao cao trung thuc rang day la danh gia tren mien du lieu cua chinh no. Khong con la dong gop ve generalization, nhung van la mot ket qua hop le.
- **(b)** Chay dung zero-shot: nap checkpoint BCSS, danh gia tren test LUAD, bao cao ket qua that (du thap). Day moi la dong gop khoa hoc co gia tri.

Khuyen nghi chon (b) vi do moi la cau hoi nghien cuu that su, va mot con so zero-shot thap nhung trung thuc co gia tri hon mot con so in-domain cao nhung bi dan nhan sai.

---

## 5. Exp 5 - Whole-ROI Stitching

**File:** `whole_roi/whole_roi_metrics.json`, `whole_roi/Whole_ROI_Sample_whole_roi_comparison.png`

| Chi so | Gia tri |
| :--- | ---: |
| So ROI | **1** |
| mIoU (4 lop) | 4,65 |
| mDice (4 lop) | 7,98 |
| Pixel Accuracy | 10,57 |
| Boundary F1 | 2,39 |
| TUM IoU | 0,00 |
| STR IoU | 17,64 |
| LYM IoU | 0,00 |
| NEC IoU | 0,97 |

**Confusion matrix 4 lop:**

```
            pred_TUM  pred_STR  pred_LYM  pred_NEC
gt_TUM             0         0         0         0
gt_STR       2675049    879442         0    344041
gt_LYM        597108   1085775         0    217879
gt_NEC            72        41         0      5528
```

### Danh gia: KHONG HOP LE - LOI PIPELINE, KHONG PHAI LOI MO HINH

Doc confusion matrix: cot `pred_TUM` va `pred_LYM` **toan so 0**. Mo hinh khong dua ra mot du doan TUM hay LYM nao tren toan bo ROI. Gan nhu toan bo pixel duoc gan nhan STR.

Nguyen nhan nam o `src/pipeline/wsi_lowmag_highmag_pipeline.py`:

- Dong 574-576: canvas duoc **khoi tao bang coarse Stage-1 prediction** (`stitched_mask[tissue_mask] = coarse_preds[tissue_mask]`), roi moi ve de cac patch Stage-2 len tren.
- Notebook goi voi `max_patches_to_infer=50 if SANITY_DRY_RUN else None`.

Voi 50 patch tren mot ROI lon, phan lon canvas **khong bao gio duoc Stage-2 ve de**, va giu nguyen du doan tho cua Stage-1. Stage-1 la CAM classifier, khong phai segmentor, nen chat luong cua no rat kem. Ket qua 4,65% la **do Stage-1 chiem phan lon dien tich**, khong phai do Stage-2.

Them nua, chi co **1 ROI** duy nhat - khong co y nghia thong ke du chi de goi la mot thi nghiem.

**Hanh dong:**
1. Bo cap `max_patches_to_infer` (hoac dat du lon de phu 100% tissue).
2. Doi logic khoi tao canvas: dung nhan `ignore/background` cho vung chua duoc Stage-2 phu, va **chi tinh metric tren vung da duoc phu**.
3. Chay tren toi thieu 10-20 ROI khac nhau, bao cao trung binh +- do lech chuan.
4. Kiem tra lai scale/stitching (`scale_x`, `scale_y`) truoc khi tin vao bat ky con so nao.

---

## 6. Bang tong hop

| # | Thi nghiem | Chat luong | Dung duoc bao cao? | Ly do chinh |
| :-: | :--- | :--- | :--- | :--- |
| 1 | Val-to-Test Protocol | Phuong phap tot, so lieu yeu | Sau khi chay full | CI 95% rong +-8,4 diem; tuning khong co tac dung |
| 2a | Multi-Magnification | Tot | **Co** (sau full run) | Xu huong don dieu hop ly |
| 2b | Adaptive Router | That bai | **Khong** | Thua random, `auc_gain` bi clamp ve 0, nhan "Sweet Spot" sai |
| 3 | Biomarker Correlation | Tin hieu manh | Sau khi chay full | n = 23 qua nho, VTP=TSR lap lai |
| 4 | LUAD "zero-shot" | Sai ban chat | **Khong** (voi nhan hien tai) | Dung checkpoint LUAD de test LUAD |
| 5 | Whole-ROI | Loi pipeline | **Khong** | Cap 50 patch, canvas lap bang Stage-1, 1 ROI |

---

## 7. Con so duy nhat dung duoc ngay bay gio

**Run 3, M6, 70,53% mIoU 4 lop tren 4.986 patch test**

Nguon: `experiments/colab_run_3_multiscale_sota_v2/COLAB_EVALUATION_REPORT.json`

Day la ket qua that, chay tren toan bo test split, khong phai dry-run. Bao cao hien tai `product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md` dang dung so sanity 4-6% o muc 6 - can cap nhat ngay bang con so 70,53% nay.

---

## 8. Thu tu khac phuc de xuat

| Uu tien | Viec can lam | File | Anh huong |
| :-: | :--- | :--- | :--- |
| P0 | Sua nhan Exp 4 hoac chay dung BCSS -> LUAD | `SE374_Colab_Scientific_Research_Runner.ipynb` cell 15 | Liem chinh hoc thuat |
| P0 | Cap nhat bao cao cuoi ky bang 70,53% | `product/reports/SE374_FINAL_TECHNICAL_REPORT_WSSS.md` muc 6 | Liem chinh hoc thuat |
| P1 | Bo cap 50 patch + sua khoi tao canvas Whole-ROI | `src/pipeline/wsi_lowmag_highmag_pipeline.py` dong 574 | Exp 5 hop le tro lai |
| P1 | Sua `auc_gain` (bo `max(0.0, ...)`), sua truc tich phan | `src/models/coarse_to_fine_router.py` | Exp 2b trung thuc |
| P2 | Bo nhan "Sweet Spot" tren hinh Pareto | `src/colab_pkg/visualizer.py` | Tranh bi phan bien |
| P2 | Tang mau biomarker len 150 patch | notebook cell 13 | Exp 3 du tin cay |
| P3 | Dat `SANITY_DRY_RUN = False` va chay full Colab | notebook cell 2 | Toan bo artifact |

**Chi phi uoc tinh cho full run tren Colab T4 Free:** khoang 1,5 - 2,5 gio cho ca 5 thi nghiem (Exp 1 chiem phan lon do phai chay 4.986 patch). Mien phi, 0 compute units. Can xin phe duyet truoc khi chay theo INVARIANT-01.

---

## 9. Ket luan

Artifact v2 la mot **ban kiem tra suc khoe pipeline**, va no da lam tot viec do: pipeline chay duoc tu dau den cuoi tren GPU, xuat du JSON/PNG, khong crash. Nhung no **khong phai** ket qua nghien cuu.

Trong 5 thi nghiem, chi **Exp 2a (multi-magnification)** la dung duoc ngay. **Exp 1** va **Exp 3** co phuong phap dung nhung so lieu chua du tin cay. **Exp 2b**, **Exp 4** va **Exp 5** phai sua truoc khi co the dua vao bat ky van ban nao.

Hai van de can xu ly ngay lap tuc vi lien quan den liem chinh hoc thuat, khong lien quan den chat luong mo hinh:

1. Exp 4 dang bi dan nhan "zero-shot cross-dataset" trong khi thuc te la in-domain.
2. Bao cao cuoi ky dang bao cao 4-6% thay vi 70,53%.

Hai van de nay phai sua truoc khi nop bat cu thu gi, ke ca ban nhap.
