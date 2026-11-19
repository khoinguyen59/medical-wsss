# AUDIT SO 02 - Kiem Dich Doc Lap (Swarm 2026-11-05)

- **Goc nhin kiem dich:** Nghi ngo gia dinh sai/qua bao. Nhiem vu: doi chieu tuyen bo cua bao cao cu (2026-11-02: "348 passed / 0 failed, ruff 0 errors, 0.0% patient leakage, 0 non-ASCII hyphen").
- **Pham vi:** Toan bo repo `SE374.R11 Weakly-Supervised Semantic Segmentation`, chay day du 10 chieu T1-T10, khong phan rap task.
- **Moi truong:** Windows + Git Bash, Python 3.13.9, pytest 8.4.2, ruff 0.12.0, torch 2.10.0+CPU, khong co GPU. Moi lenh Python duoc tien toi `KMP_DUPLICATE_LIB_OK=TRUE` (tranh OMP Error #15).
- **Che do:** Builder-Auditor tach biet. Kiem dich chi DOC + chay read-only, KHONG sua source/test/script. Chi ghi dung mot tep bao cao nay.

## Ket luan tong (verdict)

Tuyen bo "348/0 + ruff 0 errors" **KHONG the tai lap present va bi sai pham vi lenh**:

- `pytest tests/` cung mot lenh cho ba ket qua khac nhau tuy thuoc trang thai dia: **348 passed / 0 failed** (b0775vod6), **346 passed / 2 failed** (bby2s1w0x), **345 passed / 3 failed** (luong tach rieng). Nguyen nhan: bo test KHONG ke kin (non-hermetic) va ph thuoc thu tu/cong thoi (order/concurrency dependent).
- Lenh chi dinh `ruff check src tests scripts` tra ve **33 loi, exit 1** (tuyen bo "0 errors" SAI). Toan repo (tru `repos/`) = **464 loi**, gom 6 loi `F821` Undefined name `DRY_RUN` trong notebook 03/04.
- `src/` va `tests/` sach ruff rieng le (exit 0); toan bo 33 loi nam trong `scripts/`.

**Dem loi theo muc do:** Critical = 1, High = 6, Medium = 5, Low = 4. Cac kham pha sach cung duoc ghi nhan (T6 weights_only, T6 seed, T6 autocast, T7 bare-except = 0, T8 dau gach ASCII trong deliverable = 0).

---

## T1 - Chay that pytest + ruff

**Lenh + ket qua that:**

```
$ KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q -p no:cacheprovider
... (luong b0775vod6) 348 passed, 2 warnings in 253.79s
... (luong bby2s1w0x) 2 failed, 346 passed, 2 warnings in 209.09s
$ KMP_DUPLICATE_LIB_OK=TRUE python -m ruff check src tests scripts
Found 33 errors.   (RUFF_EXIT=1)
$ KMP_DUPLICATE_LIB_OK=TRUE python -m ruff check src        -> src_exit=0
$ KMP_DUPLICATE_LIB_OK=TRUE python -m ruff check tests      -> tests_exit=0
$ KMP_DUPLICATE_LIB_OK=TRUE python -m ruff check scripts    -> scripts_exit=1
$ KMP_DUPLICATE_LIB_OK=TRUE python -m ruff check . --statistics --exclude repos
464 errors (193 I001, 93 F811, 53 E701, 36 F541, 35 F401, 25 E401, 19 E741, 6 F821, 4 F841)
```

### [T1-non-hermetic] | High | confidence High
`tests/test_thin_drive_runner.py::test_2`/`test_3` goi `packager.build_drive_package()` thao tac tren cay thu muc THAT `drive_upload_ready/` thay vi `tmp_path` (tham so nhan vao nhung bi bo qua). Khi mot tien trinh khong song/cua san pham cu nam keep tay, `shutil.rmtree` fail. Day la ly do mot lenh pytest cho 3 ket qua khac nhau.

```
tests\test_thin_drive_runner.py:98:
    summary = packager.build_drive_package()
scripts\build_drive_package.py:1545: in build_drive_package
    shutil.rmtree(bundle_dir)
path = WindowsPath('.../drive_upload_ready/SE374_WSSS_Drive_Code')
FAILED tests/test_thin_drive_runner.py::test_2_drive_bundle_packager_creates_complete_folder_and_zip
FAILED tests/test_thin_drive_runner.py::test_3_new_thin_runner_notebooks_valid_concise_and_modular
```
- **Hau qua:** "348/0" la con so mong manh, khong phai bat bien cua mot checkout sach. Trong CI (fresh checkout, chua co artifact) cac test nay co the fail flip-flop. Khong the dung lam chung chi "0 failed".

### [T1-ruff-scripts] | Medium | confidence High
33 loi ruff trong `scripts/` (fix_bfloat16_and_dataset_count.py, fix_dataset_path_resolution.py: 8 loi each; download_papers.py, build_drive_package.py, build_t4_l4_notebook_suites.py, refactor_notebooks_pure_colab.py...), chu yeu E741 (ten bien khan nghi `l`/`I`/`O`), I001, F401, F841.
- **Hau qua:** Tuyen bo "ruff 0 errors" SAI theo dung lenh duoc yeu cau; chi dung neu thu hep pham vi con `src tests`.

### [T1-f821-dryrun] | High | confidence High
`DRY_RUN` duoc tham chieu nhung chua bao gio dinh nghia:

```
notebooks\03_Train_Proposed_MedSAM.ipynb:cell 15: F821 Undefined name `DRY_RUN`
notebooks\03_Train_Proposed_MedSAM_L4.ipynb / _T4.ipynb: F821 `DRY_RUN`
notebooks\04_Exhaustive_Benchmark_Survival.ipynb / _L4: F821 `DRY_RUN`
```
- **Hau qua:** Chay tren Colab, o code nao cham `DRY_RUN` se `NameError` -> notebook gay giua chung. 5+ notebook bi anh huong.

---

## T2 - Quet Zero-Mockup (giao duc/sao chep so lieu)

**Lenh:** `grep -rn -e "np.random" -e "torch.rand" -e "fake" -e "mock" -e "surrogate" ... src/` va mo tung file boi canh.

### [T2-surrogate-silent] | Critical | confidence High
MedSAM adapter tu dong roi ve `SurrogateImageEncoder` (mang kich thuong duoc gan `~86M params` gia theo hinh dang ViT-B, khoi tao NGAU NHIN) khi thieu checkpoint/segment_anything:

```
src/models/sam_wsss_adapter.py:610   class SurrogateImageEncoder(nn.Module):
src/models/sam_wsss_adapter.py:745   self.is_surrogate = True      # default
src/models/sam_wsss_adapter.py:759-765  ... SurrogateImageEncoder(out_dim=256); self.is_surrogate = True
src/clinical/exhaustive_evaluator.py:350 "fallback to surrogate image encoder"
src/clinical/exhaustive_evaluator.py:355 use_surrogate_if_missing=True
```
`is_surrogate` chi duoc gan trong class; `grep -rn "is_surrogate" src/` cho thay KHONG cho nao doc no ra ngoai -> khong co canh bao in ra, khong co co che danh dau ket qua la gia trong bao cao.
- **Hau qua lam sang:** Neu MedSAM trong so vang mat, pipeline van chay va xuat phan doan cua mot mang CHUA HUAN LUYEN, nhung duoc ghi nhan nhu la phuong phap MedSAM. Day la che do mockup lang vi pham INV-07 nghiem trong nhat.

### [T2-notebook-output-fabrication] | High | confidence Medium
`scripts/build_drive_package.py` ham `sync_historical_and_standalone_outputs()` (~428-1045) **che tao** stdout va base64 PNG cho cell notebook: hardcode "PyTorch Version: 2.10.0+cu128", vong tien do `for step_i in range(250,4986,250)`, va ty le lop tung patch cung gia tri cung (1031-1039). README (322-332) hardcode metric Run1-6 gom "70.53%".
- Trang thai hien tai: cac duong dan `COLAB_EVALUATION_REPORT.json` ma no doc (`experiments/colab_run_1_heuristic`, `_3_fixed`, `_4_full`, `_6_v2`) **khong ton tai** trong `experiments/` (chi co `_stage2_baseline`, `_dual_stage_ensemble`, `_multiscale_sota_v2`, `results_dryrun_v1`), nen khoi tao bi guard `is_file()` va no-op lang. May moc che tao van con trong source va co the bung khi co artifact.
- **Hau qua:** Co che "dien dan ket qua" vao notebook khong do from a real run la rui ro fabricat; tuy duoc guard, ban chat no vi pham tinh minhan ky thuat.

### [T2-synthetic-dataset] | Low | confidence High
`SyntheticHistologyDataset` (`src/training/train_proposed_sam.py:195-217`, dung `torch.rand`/`torch.randint`) chi duoc dung khi `--dry_run` hoac thieu du lieu:
```
train_proposed_sam.py:486  if os.path.exists(args.data_dir) and not args.dry_run: -> BCSSDataset that
train_proposed_sam.py:490  else: -> SyntheticHistologyDataset (num_samples=4/32)
```
Co nhan ro "dry-run and local smoke testing". Duoc dung trong `tests/test_sam_wsss_adapter.py:307` nhu fixture hop le. KHONG roi vao duong do BCSS bao cao (Colab dung `colab_runner` tai BCSS that). Ghi nhan la chap nhan duoc nhung can doi ten de tranh dau doc INV-07.

### [T2-cach-nhan] | Clean | confidence High
Duong bao cao co guard chong mockup chu dong: `exhaustive_evaluator.py:661,699,719,723` ("do not fabricate dummy patients", "Zero dummy patients injected"); `metrics.py:3` header "zero-mockup implementations". `colab_runner.py:597,617,656` dung `torch.rand`/`np.random` voi tensor nho 32x32 cho smoke/inspect, khong phai tinh metric. `evaluator.py:221,492` dung `np.random.RandomState(seed)` de lay mau determinstic, khong che tao du lieu.

---

## T3 - Chong roi ro cap benh nhan (patient-level anti-leakage)

**Lenh:** `grep -rn -e "train_test_split" -e "GroupKFold" -e "Stratified" -e "shuffle=True" src/`

- Chi co 3 cho `shuffle=True`, tat ca o `DataLoader` da dao thuat tap train (`dataset_loader.py:519,569`, `train_proposed_sam.py:496`) - khong phai tran split. **Khong phat hien** `train_test_split` sai pham hoac GroupKFold bi sua dung.
- Split benh nhan dieu khien bang manifest tinh nhan `src/config/bcss_patient_splits.json` (151 benh nhan), doc boi `patient_split.py` va `dataset_loader.py:116`.

**Phan dinh `test_patient_split.py`:** Test nay KHONG ty quy (self-referential) o phan quan trong: `test_verify_patient_leakage_catches_overlap` dung du lieu doi dau co tinh overlap va assert `ValueError` -> chung minh ham `verify_patient_leakage` that bat roi ro. Tuy nhien:

### [T3-manifest-config-only] | Low | confidence Medium
`test_151_patient_split_manifest_zero_leakage` chi kiem tra **tinh roi nhau noi bo cua JSON manifest** (train/val/test disjoint, 151 unique), CHUNG MINH manifest dung, nhưng KHONG kiem duong that pipeline loc patch theo manifest luc chay. "0.0% leakage" duoc chung o cap manifest, con o cap runtime la duoc GIA DINH (qua `select_patient_files` nhom theo TCGA ID). Can mot test noi-chain (tich hop) de khang dinh tuyet doi.

### [T3-hardcoded-fallback] | Medium | confidence High
`dataset_loader.py:127-133` neu manifest vang mat **lang ly tra ve split hardcode 7 benh nhan**:
```
if not config_path.exists():
    return {"train": [3 IDs], "val": [2 IDs], "test": [2 IDs]}
```
Ba danh sach roi nhau nen khong gay leakage, nhưng co ghep im lang tu 151 con 7 benh nhan -> ket qua/reproducibility bi sai lech nghiem trong ma khong bao loi. Nen raise thay vi fallback lang.

---

## T4 - INVARIANT-29 chong tim kiem nong (anti-shallow-glob)

**Lenh:** `grep -rn -e ".glob(" -e ".rglob(" -e "resolve_split_path" src/`

### [T4-shallow-glob] | High | confidence High
`BCSSDataset` tai `src/datasets/bcss_dataset.py` dung **tim kiem nong** va KHONG co `rglob`/`resolve_split_path`:
```
bcss_dataset.py:55   self.image_paths = sorted(glob.glob(pattern))   # pattern = .../*.png
bcss_dataset.py:72   self.image_paths = sorted(glob.glob(pattern))
```
`exhaustive_evaluator.py:394` cung dung nong `img_dir.glob("*.png")`. Trong khi do `dataset_loader.py:163` dung `rglob("*")` va `downloader.py:146` dung `base.glob(f"**/{sp}/img")` (multi-tier, dat chuan). INV-29 yeu cau MOI Dataset phai co `resolve_split_path` + `rglob` -> `BCSSDataset` vi pham truc tiep.

### [T4-no-safety-net] | Medium | confidence High
INV-29 bat buoc luoi an toan `len(files)==0` placeholder (`len(dataset)>=1`). `grep -rn "len(.*)==0\|placeholder\|num_samples" src/datasets/ src/training/dataset_loader.py` tra ve **RONG**. `BCSSDataset.__len__` chi la `return len(self.image_paths)` (line 95) -> neu thu muc rong/doan duong sai, `len=0` -> DataLoader vang `ValueError: num_samples should be a positive integer`. Khop chinh xac voi loi duoc ghi nhan boi Stack Overflow (xem T10).

### [T4-silent-zero-mask] | Medium | confidence High
Khi file mask thieu, bo nap im lang thay bang 0 chu khong bao loi: `bcss_dataset.py:133` `mask_tensor = torch.zeros(...)`, `dataset_loader.py:326` va `458` (BCSSEvalDataset) cung zero-mask fallback. Anh huong truc tiep den Dice/mIoU (lop bi tinh la toan nen).overlap voi T7.

---

## T5 - INVARIANT-28/27 vong doi Colab + drift notebook

**Lenh:** `grep -rn -e "drive.mount" -e "/content/drive" notebooks/ *.ipynb` (loai mirror `drive_upload_ready`).

### [T5-inv28-drive-mount] | High | confidence High
INV-28: "TUYET DOI NGHIEM CAM viec tu y dua lenh `drive.mount("/content/drive")`... notebook phai chay mot mach tu dau den cuoi, khong dung lai doi cap quyen OAuth". Nhung TOAN BO 8 notebook ThinDriveRunner_v2 (T4/L4/A100/Sanity150, o ca `notebooks/` va `notebooks/modular_drive/`) cung BonusDemo deu co:
```
SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb  cell-02:
    if "google.colab" in sys.modules:
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
        except Exception: pass
```
Da duoc guard `try/except: pass` + co fallback clone GitHub (kich ban D), nhưng LENH CAM VAN HIEN DIEN va se pop OAuth khi Run All -> vi pham cong thuc cua INV-28.

### [T5-test-enshrines-violation] | High | confidence High
`tests/test_thin_drive_runner.py` test_10 assert `"drive.mount" in c2_src` - tuc la BO TEST ENSHRINE (hop phap hoa) chinh vi pham INV-28. Test xanh dang bao ve loi quy che.

**7-buoc vong doi:** Notebook Thin Runner co day du Buoc 1-7 (mount/load -> config -> tai du lieu + ckpt -> khoi tao model + engine -> danh gia 7 phuong phap -> truc quan 300-DPI -> xuat JSON/CSV/MD + zip), dong kin that, khong buoc metric. Ghi nhan dat ve mat vong doi.

### [T5-config-drift] | Medium | confidence High
4 cau hinh GPU (T4/L4/A100/Sanity) ton tai o `notebooks/` va `notebooks/modular_drive/` trung khop md5, nhưng ban `drive_upload_ready/` **drift** (xem T9). Nen tai build lai bundle truoc khi dong goi.

---

## T6 - Thiet bi / tai nguyen / weights_only / seed

### [T6-weights_only] | Clean | confidence High
MOI cho `torch.load` trong `src/` deu duoc phoi `try weights_only=True / except weights_only=False  # nosec B614`: `downloader.py:230/232`, `model_loader.py:574/576`, `wsss_inference_methods.py:363/365`, `generate_pseudo_masks.py:321/323`, `train_proposed_sam.py:402/404`, `train_stage1.py:390/392`, `train_stage2.py:298/300/309`. Xu ly DUNG cho rui ro doi mac dinh torch>=2.6 (khop loi PyTorch 2.6 trong T10).

### [T6-seed] | Clean (note) | confidence High
`set_reproducible_seed` dat `random.seed` + `np.random.seed` + `torch.manual_seed` + `cuda.manual_seed_all` + `backends.cudnn.deterministic=True` (`colab_runner.py:100-110`, `train_stage1.py:372-376`, `train_stage2.py:250-252`). INV-09 dat. Note nho: thieu `torch.use_deterministic_algorithms(True)` nen determinism khong tuyet doi mot so toan tu; chap nhan duoc.

### [T6-autocast] | Clean | confidence High
`model_loader.py:_get_autocast_context` (916-926) tra ve `nullcontext` tren CPU -> khong crash khi chay T4/L4/A100/CPU. `.to(dev)` va `next(model.parameters()).device` dung nhat quan; SAM prompt tensor duoc xép tren `cams.device` (CAMPromptGenerator) -> thiet bi nhat quan, khong co rui ro transfer sai thiet bi.

---

## T7 - Loi im lang (silent failure)

**Lenh:** `grep -rn "except\s*:" --include=*.py src/ scripts/ tests/`

- **bare `except:` = 0** (tuyen bo sach phan nay DUNG).
- **`except Exception` trong src/ = 49** - nhieu cho nuot loi. Ket hop voi `drive.mount ... except Exception: pass` (notebook) va zero-mask fallback (T4) -> kha nang an loi. Khong co `contextlib.suppress` (0) va `warnings.filterwarnings` (0 trong src/) - sach ve hai mat nay.
- Xep chung voi [T3-hardcoded-fallback], [T4-silent-zero-mask] thanh mot cum rui ro "im lang khi thieu du lieu".

---

## T8 - INVARIANT-12 typo + CI gate

**Lenh:** `grep -rnP "[\x{2012}\x{2013}\x{2014}\x{2015}\x{2212}]" --include=*.py --include=*.md --include=*.ipynb . | grep -v /repos/` -> 230 hit, nhưng **scoped toi deliverable (src/, scripts/, notebooks/, root *.ipynb, AGENTS.md, README.md, docs/, product/) = 0**.

### [T8-deliverable-clean] | Clean | confidence High
"0 non-ASCII hyphen" **DUNG** cho toan bo deliverable code + notebook + tai lieu. 230 hit con lai nam trong `.agents/**` va `graphify-out/**` (san pham phu cua agent, khong phai giao pham).

### [T8-governance-blindspot] | Low | confidence High
`tests/test_governance_invariants.py` dung `BAD_DASH_CHARS=['\u2014','\u2013','\u2012','\u2015','\u2212']` (bao phu tot, them U+2012) nhưng `glob.glob("**/*.md", recursive=True)` **khong lien ket thu muc an (dot-directory)**. Chung minh:
```
python -c "import glob; print(len(glob.glob('**/*.md',recursive=True)), any('.agents' in p for p in glob.glob('**/*.md',recursive=True)))"
-> 755 False
```
-> 755 file md duoc quet, 0 tu `.agents`. Test co DIEM MUT cau truc: vi pham trong `.agents/**` khong the bi phat hien (may man la deliverable sach nen ket luan van dung pham vi can kiem).

### [T8-ci-not-ruff] | Medium | confidence High
`.github/workflows/ci.yml` **KHONG chay ruff**. Buoc 1 dung `flake8 src/ --select=E9,F63,F7,F82 ... || true` (hai lenh deu co `|| true` -> **khong bao gio fail build**). Buoc 2 mockup grep chi doi chieu 5 token cu the (`TODO: mock`, `fakeData`, `dummyData`, `fake_metric`, `simulateTx`) - khong cai nao xuat hien -> luon PASS, khong quet `np.random`/`torch.rand`/`synthetic` that. Buoc 4 `pytest tests/ -v` tren ubuntu fresh checkout -> de gap flip-flop cua [T1-non-hermetic]. Ket luan: CI khong enforce duoc cac tuyen bo cua bao cao cu.

---

## T9 - Code chet & trung lap (dead code / duplication)

### [T9-notebook-drift] | Medium | confidence High
`md5sum` 3 ban cua moi notebook v2 cho thay **ban `drive_upload_ready/` khac ban goc**:
```
T4_Full4986 : notebooks=35b09c4b... modular=35b09c4b... drive=f39ae743...   (KHAC)
L4_Full4986 : notebooks=70533ff0... modular=70533ff0... drive=b16c971b...   (KHAC)
A100_Full4986: notebooks=c127184e... modular=c127184e... drive=ab370de9...   (KHAC)
T4_Sanity150: notebooks=2250fea4... modular=2250fea4... drive=a23f17c7...   (KHAC)
```
Ban nguyen (`notebooks/`) va `modular_drive/` khop nhau, nhưng bundle upload-ready da drift -> nguoi dung chay ban khac ban goc trong repo. Do `build_drive_package.py` sync/doi notebook khi build; bundle la stale cho den khi rebuild.

### [T9-src-config] | Clean (giai thich) | confidence High
`src/config/` co 0 file `.py` (chi `.gitkeep` + `bcss_patient_splits.json`). Day la THU MUC MANIFEST duoc doc theo duong dan boi `patient_split.py:112` va `dataset_loader.py:116` -> KHONG phai dead code, dung dung muc dich JSON store.

### [T9-generated-source-changes] | Note | confidence High
`git status` cho thay `src/*.py` bi ' M' nhung `git diff` chi la **xu ly whitespace duoi** (xoa trailing space), ngay 2026-11-02 (builder sua, KHONG phai do test chay). Kiem dich KHONG quy cho test gay bien doi source. Bien doi that cua test la vao `drive_upload_ready/` (artifact sinh so).

---

## T10 - Chat luong test + doi sou loi ben ngoai

**Quality theo testing-best-practices:**
- **Test order dependence:** [T1-non-hermetic] - `test_thin_drive_runner.py::test_2/3/7` goi `build_drive_package()` ghi de `drive_upload_ready/` that, bo qua `tmp_path` -> thu tu/chung thoi anh huong ket qua. Day la loi chat luong test nghiem trong nhat.
- **Test enshrine vi pham:** test_10 assert `drive.mount` (T5) - test bao ve loi quy che thay vì bat.
- **Assert qua chat len noi dung cu the (brittle):** test_9/test_1 va README assert doi tung chuoi (vi du "70.53%", "M6_..._Ensemble", cell count bang `expected_cells`) -> de gay khi refact.
- **Khong phat hien `time.sleep` nondeterminism** trong src/; `np.random.RandomState(seed)` dung determinstic -> OK.
- **Boundary/error coverage kha:** `test_patient_split.py` co Hypothesis fuzz (`test_hypothesis_extract_patient_id_never_crashes` tren `st.text()`), `test_load_patient_splits_invalid_path` kiem FileNotFoundError -> tot ve bien.
- **Over-mocking:** thap; fixture `SyntheticHistologyDataset` dung co che do cho smoke.

**Doi sou loi that tu PyTorch/Stack Overflow (URL cau tra loi duoc chap nhan):**

1. `ValueError: num_samples should be a positive integer value, but got num_samples=0` -> "The problem is that the dataset is empty. The datapath may be wrong... ending up with no object." Truong hop khop chinh xac [T4-no-safety-net]/[T4-shallow-glob] cua `BCSSDataset`. https://stackoverflow.com/questions/71615089/
2. PyTorch 2.6 `torch.load` default `weights_only` False->True: "Re-running torch.load with weights_only=False..." -> dung nhu [T6-weights_only] xu ly; src da phoi try/except dung cach. https://stackoverflow.com/questions/79584485/ , https://stackoverflow.com/questions/79715567/
3. Windows duong dan backslash trong dataset (`'list index out of range'`): "occurs when you run your code on windows because windows paths use backslash" - lien quan [T4] khi `glob.glob(pattern)` cung duong dan tren `bcss_dataset.py`. https://stackoverflow.com/questions/61522539/

---

## Phụ luc: Lệnh đã chạy (repro)

```
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q -p no:cacheprovider
KMP_DUPLICATE_LIB_OK=TRUE python -m ruff check src tests scripts
KMP_DUPLICATE_LIB_OK=TRUE python -m ruff check . --statistics --exclude repos
grep -rnP "[\x{2012}-\x{2015}\x{2212}]" --include=*.py --include=*.md --include=*.ipynb . | grep -v /repos/
grep -rn "drive.mount" notebooks/ *.ipynb | grep -v drive_upload_ready
md5sum notebooks/*.ipynb notebooks/modular_drive/*.ipynb drive_upload_ready/.../notebooks/*.ipynb
grep -rn "is_surrogate" src/    # -> chi gan, khong doc ngoai
python -c "import glob; print(len(glob.glob('**/*.md',recursive=True)))"   # -> 755, dot-dir blind spot
```

## Ket luan cho Builder

- Cac tuyen bo cu "348/0" va "ruff 0 errors" can bo lai: dung lenh `src tests scripts` -> 33 loi; pytest non-hermetic -> ket qua flip-flop. Uu tien lam hermetic `test_thin_drive_runner.py` (dùng `tmp_path` that su, tranh `shutil.rmtree` tren repo that).
- Xuri `DRY_RUN` undefined (F821) trong notebook 03/04.
- Cong khai ro `is_surrogate` ra bao cao + in WARNING khi roi ve SurrogateImageEncoder (INV-07).
- Thay `glob.glob("*.png")` nong bang `rglob`/`resolve_split_path` + them luoi an toan `len==0` placeholder cho `BCSSDataset` (INV-29).
- Xoay `drive.mount` khoi notebook Thin Runner hoac sua INV-28/test_10 cho phep co dieu kien (quyet quan quy che).
- Rebuild `drive_upload_ready/` de dong bo md5 voi `notebooks/`.
- `dataset_loader.py` fallback hardcode 7 benh nhan: thay bang `raise`.
