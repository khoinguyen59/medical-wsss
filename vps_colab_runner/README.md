# Temporary VPS Colab Runner (Oracle 2-4 OCPU / Linux VPS)

> **Luu y quan trong:** Thu muc `vps_colab_runner/` hoan toan doc lap. Sau khi huan luyen xong, chay `git rm -r vps_colab_runner` de xoa sach ma khong anh huong du an chinh.

---

## 1. Cai dat 1 lan tren VPS

1. Chep tep xac thuc ADC len VPS (ngoai thu muc git, quyen `chmod 600`):
   ```bash
   ssh <user>@<vps-ip> "mkdir -p ~/.config/gcloud && chmod 700 ~/.config/gcloud"
   scp "<duong-dan-toi-acc11.json>" <user>@<vps-ip>:~/.config/gcloud/application_default_credentials.json
   ssh <user>@<vps-ip> "chmod 600 ~/.config/gcloud/application_default_credentials.json"
   ```
2. Clone repo va chay setup:
   ```bash
   git clone https://github.com/khoinguyen59/medical-wsss.git
   cd medical-wsss && bash vps_colab_runner/setup_vps_colab.sh
   ```
3. Kiem tra thu khong ton GPU (`--dry-run`):
   ```bash
   bash vps_colab_runner/run_colab_vps.sh --notebook nb1 --gpu A100 --high-mem --preset arm_a_polysgd --session se374_arm_a --dry-run
   ```

---

## 2. Chay da phien va da GPU (`T4|L4|G4|A100|H100`) trong `tmux`

Moi phien phai dung ten hop le `^se374_[a-zA-Z0-9_-]+$` (hoac `nb1`/`nb2` legacy). Runner tu dong co lap `--config vps_colab_runner/runs/<session>/.state/colab_sessions.json` va ban sao notebook `.state/run_notebook.ipynb` de tranh ghi de `notebooks/*_output.ipynb`.

```bash
tmux new -s colab_se374_arm_a
cd ~/medical-wsss
bash vps_colab_runner/run_colab_vps.sh \
  --notebook nb1 --gpu A100 --high-mem --gpu-fallback A100,L4 \
  --preset arm_a_polysgd --session se374_arm_a \
  --env STAGE1_SEED=42 --max-minutes 240 --sync-interval 300
```

- `--high-mem`: Tu dong bat cho `H100|G4` (mac dinh `0` tren `A100|L4|T4`, co the bat tren `A100` bang `--high-mem` hoac tat bang `--no-high-mem`; bo qua tren `L4` theo `colab_cli`).
- `--preset <name>`: Truyen `--env STAGE1_EXP_PRESET=<name>` vao Colab notebook.
- `--env KEY=VALUE`: Lap lai duoc nhieu lan.
- `--credentials <path>`: Chi dinh tep ADC rieng neu can.

---

## 3. Bao ve P0 chong tat nham phien ngoai he thong (`session_guard.py`)

1. **Snapshot `protected_endpoints` bat bien (fail-closed):** Truoc moi lan `colab new`, `session_guard.py snapshot-protected` doc `colab sessions` (read-only) va ghi cac endpoint da ton tai (vi du ComfyUI A100 dang chay tren `acc11`) vao `vps_colab_runner/.state/<acct>_protected_endpoints.json`. Neu snapshot loi, runner TU CHOI tao VM.
2. **Allow-list `owned_sessions`:** Chi cac phien do he thong nay tao moi duoc ghi vao `vps_colab_runner/.state/<acct>_owned_sessions.json`.
3. **Tat phien khop kep (dual-match):** `--stop-session <name>`, `--stop-owned`, `--stop-all` (alias an toan cua `--stop-owned`) va `cleanup()` chi phep chay `colab --config <store> stop -s <name>` khi CA `session_name` VA `endpoint` khop `owned_sessions` VA `endpoint` KHONG nam trong `protected_endpoints`. Tuyet doi khong dung `colab stop --all`.
4. **Dong bo an toan & phan tach canh bao:**
   - `[EARLY STOP]` / `[EARLY STOP PATIENCE]`: Cho phep Cell 8 xuat du lieu va chay `download_final_artifacts` (thoat sach 0).
   - `[CALIBRATION COLLAPSE ALARM]` / `NaN`: Ngat nhanh fail-fast.
   - Luon chay 1 lan `sync_all_checkpoints` dong bo sau khi dung watcher (ke ca `stage1_best_cls.pth` va `STAGE1_TRAINING_METRICS.json`).

---

## 4. Web dashboard & Telegram bot (`SELL_BOT`)

```bash
cd ~/medical-wsss && git pull --ff-only && bash vps_colab_runner/start_dashboard_public.sh
```

- Mac dinh `--colab-refresh 60` giay de han che phinh `colab.log`.
- Giao dien hien thi bang tong hop **Owned vs Foreign**; `POST /api/start` kiem soat dong thoi bang `COLAB_MAX_OWNED_CONCURRENT` (mac dinh `1` de bao ve quota cho phien A100 ngoai he thong, khong bi chan boi `usage.active` cua phien ngoai).
- `POST /api/stop` ho tro `{"session": "<name>"}` hoac `{"scope": "owned_all"}` (mac dinh tuong thich `SELL_BOT` khi body rong).
- Lenh dung khan cap:
  - Dung 1 phien owned: `bash vps_colab_runner/run_colab_vps.sh --stop-session se374_arm_a`
  - Dung tat ca phien owned: `bash vps_colab_runner/run_colab_vps.sh --stop-owned`
