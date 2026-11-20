# Temporary VPS Colab Runner (Oracle 2-4 OCPU / Linux VPS)

> **Lưu ý quan trọng:** Thư mục `vps_colab_runner/` này hoàn toàn độc lập, chỉ dùng tạm thời để điều khiển chạy Google Colab từ VPS. Sau khi huấn luyện xong toàn bộ mô hình, chỉ cần chạy `git rm -r vps_colab_runner` để xóa sạch khỏi GitHub mà không ảnh hưởng bất kỳ tệp nào của dự án chính.

---

## 1. Cài đặt 1 lần trên VPS (Oracle 2-4 OCPU, ARM64/x86_64, Ubuntu hoặc Oracle Linux)

### Bước 1: Chép tệp xác thực ADC (`acc9.json`) lên VPS (ngoài thư mục git)
```bash
ssh <user>@<vps-ip> "mkdir -p ~/.config/gcloud && chmod 700 ~/.config/gcloud"
scp "<duong-dan-toi-acc9.json>" <user>@<vps-ip>:~/.config/gcloud/application_default_credentials.json
ssh <user>@<vps-ip> "chmod 600 ~/.config/gcloud/application_default_credentials.json"
```

### Bước 2: Clone repo và chạy setup 1 lần
```bash
git clone https://github.com/khoinguyen59/medical-wsss.git
cd medical-wsss
bash vps_colab_runner/setup_vps_colab.sh
```

### Bước 3: Kiểm tra thử không tốn GPU (`--dry-run`)
```bash
bash vps_colab_runner/run_colab_vps.sh --notebook nb1 --gpu L4 --dry-run
```

---

## 2. Chạy huấn luyện trong `tmux` (An toàn khi tắt máy local)

### Chạy Notebook 1 (Stage 1, giới hạn cứng 210 phút, sync mỗi 5 phút)
```bash
tmux new -s colab_nb1
cd ~/medical-wsss
bash vps_colab_runner/run_colab_vps.sh --notebook nb1 --gpu L4 --max-minutes 210 --max-retries 3 --sync-interval 300
```

### Chạy Notebook 2 (Stage 2, giới hạn cứng 240 phút, sync mỗi 5 phút)
```bash
tmux new -s colab_nb2
cd ~/medical-wsss
bash vps_colab_runner/run_colab_vps.sh --notebook nb2 --gpu L4 --max-minutes 240 --max-retries 3 --sync-interval 300
```

Nhấn `Ctrl + B` rồi `D` để thoát `tmux` (máy local tắt đi ngủ bình thường).

---

## 3. Cơ chế dừng chống tốn GPU & tự cứu checkpoint
1. **`trap cleanup EXIT INT TERM`:** Mọi tình huống thoát (chạy xong, lỗi, `Ctrl+C`, `kill`) đều gọi `colab --auth adc stop` và xác nhận `Active assignments: 0`.
2. **Đồng hồ cắt cứng (`--max-minutes`):** Quá giới hạn thời gian sẽ tự kéo checkpoint về VPS rồi tắt ngay VM.
3. **Ngắt sớm khi bất thường (`--stop-on-alarm`):** Phát hiện `Loss: nan`, `Non-finite epoch loss` hoặc `[EARLY REGRESSION ALARM]` sẽ lưu best checkpoint và tắt ngay VM.
4. **Đồng bộ checkpoint nguyên tử (`SHA256`) & Auto-Resume:**
   - Mỗi `300` giây đồng bộ `*_retrained_best.pth` và `*_latest.pth` vào `vps_colab_runner/runs/<session>/checkpoints/` qua tệp `.tmp` + kiểm `SHA256` + `mv -f`.
   - Nếu rớt mạng hoặc Colab thu hồi VM, tự mở VM mới và upload ngược checkpoint lên `/content/checkpoints/` (khớp `SHA256`) để chạy tiếp từ `epoch + 1` (tối đa `--max-retries 3`).

---

## 4. Theo dõi, Dừng khẩn cấp, Kéo kết quả & Xóa sạch sau khi dùng xong

- **Xem log trực tiếp:** `tail -f ~/medical-wsss/vps_colab_runner/runs/nb1/run.log`
- **Kiểm tra trạng thái & số dư:** `bash vps_colab_runner/run_colab_vps.sh --notebook nb1 --status`
- **Dừng khẩn cấp toàn bộ Colab VM:** `bash vps_colab_runner/run_colab_vps.sh --stop-all`
- **Kéo kết quả về máy Windows:**
  ```bash
  scp -r <user>@<vps-ip>:~/medical-wsss/vps_colab_runner/runs/nb1 ./experiments/vps_stage1_results
  ```
- **Xóa sạch thư mục tạm khỏi GitHub sau khi xong dự án:**
  ```bash
  git rm -r vps_colab_runner && git commit -m "chore: remove temporary vps_colab_runner"
  ```
