# Huong Dan Luu Tru Du Lieu Y Khoa (`data/`)

Thu muc nay danh rieng de chua du lieu anh y te (BCSS-WSSS, LUAD-HistoSeg, Camelyon16). Toan bo tep tin anh va mat na trong thu muc nay deu duoc loai tru khoi he thong Git qua `.gitignore` de tranh lam phinh to dung luong repository.

---

## 1. Cau Truc Thu Muc Du Lieu Chuan

```
data/
├── BCSS-WSSS/                             # Tap du lieu chinh ung thu vu (Han et al., MedIA 2022)
│   ├── train/                             # Anh huan luyen kem nhan yeu trong ten tep
│   │   └── TCGA-XX-XXXX_X_Y_[a b c d].png
│   ├── valid/                             # Anh kiem dinh kem ground truth mask
│   │   ├── img/
│   │   └── mask/
│   └── test/                              # Anh kiem thu doc lap kem ground truth mask
│       ├── img/
│       └── mask/
├── LUAD-HistoSeg/                         # Tap du lieu ung thu phoi (Han et al., MedIA 2022)
│   ├── train/
│   ├── valid/
│   └── test/
└── sample_dataset/                        # Tap du lieu mau cuc nho (Mini Sample) dung cho Dry-Run
    ├── train/
    ├── valid/
    └── test/
```

---

## 2. Nguon Tai Chinh Thuc (Official Download Links)
- **BCSS-WSSS:** Google Drive: `https://drive.google.com/drive/folders/1iS2Z0DsbACqGp7m6VDJbAcgzeXNEFr77?usp=sharing`
- **LUAD-HistoSeg:** Google Drive: `https://drive.google.com/drive/folders/1E3Yei3Or3xJXukHIybZAgochxfn6FJpr?usp=sharing`
- **Pretrained Checkpoints:** Google Drive: `https://drive.google.com/drive/folders/1pRsf-TulOARYN13J-_iJNw_VUa2d1NnZ?usp=sharing`

---

## 3. Quy Tac Bat Bien Chon Tap Benh Nhan (`[INVARIANT-08]`)
- Khi giai nen va phan chia, bat buoc tach theo `Patient ID` (vi du: `TCGA-A2-A0T2`).
- Tuyet doi khong tron lan cac patches cua cung mot benh nhan giua tap train va tap test.
