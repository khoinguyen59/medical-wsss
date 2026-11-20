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

## 4. Web dashboard theo dõi (giao diện sáng, trắng xanh dương, mở được trên điện thoại)

Dashboard chỉ dùng thư viện chuẩn Python. Một lệnh trên VPS sẽ bật dashboard cùng Cloudflare quick tunnel và in ra link `https://...trycloudflare.com/?key=...` bấm được trên điện thoại:

```bash
cd ~/medical-wsss && git pull --ff-only && bash vps_colab_runner/start_dashboard_public.sh
```

- Lần chạy đầu script tạo tài khoản `colab` và in **một lần** mật khẩu ngẫu nhiên cùng `DASH_API_KEY`. Mật khẩu chỉ lưu dạng hash PBKDF2 trong `~/.colab_dash_auth.json` (quyền 600, không nằm trong git). Đổi mật khẩu: `python3 vps_colab_runner/dash_auth.py --init --reset-password`.
- Mở link, đăng nhập bằng tài khoản + mật khẩu, trình duyệt nhớ phiên 30 ngày. Sai 5 lần thì khóa IP 10 phút.
- Link quick tunnel đổi mỗi lần chạy lại script. Cần link cố định thì phải dùng Cloudflare named tunnel (cần tên miền).
- Chưa đăng nhập thì mọi trang và API đều bị chặn (`302` về trang đăng nhập hoặc `401`).
- `DASH_API_KEY` dành cho bot Telegram (header `X-API-Key`), xem mục 6.

Hiển thị: ô thông số chính (epoch hiện tại/tổng, chỉ số val tốt nhất, loss mới nhất, thời gian đã chạy, ước tính còn lại, GPU bật/tắt), cấu hình lần chạy (loss, LR, batch, pretrained, thiết bị), số dư Colab, đếm ngược watchdog, tải VPS, bảng checkpoint đã tải về (kích thước, thời điểm, SHA256), biểu đồ theo epoch, sự kiện chính, log đã lọc nhiễu. Nút **TẮT GPU NGAY** dừng runner và chạy `--stop-all`. Mục **Khởi chạy phiên mới** chạy NB1/NB2 trong tmux (chỉ khi không có GPU nào đang bật, giới hạn 10-300 phút).

Dùng cục bộ không cần Cloudflare: `python3 vps_colab_runner/dashboard.py --port 8765` rồi `ssh -N -L 8765:127.0.0.1:8765 ...` (nếu chưa tạo tài khoản thì chỉ nhận truy cập loopback, không cần đăng nhập).

---

## 5. Theo dõi, Dừng khẩn cấp, Kéo kết quả & Xóa sạch sau khi dùng xong

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

---

## 6. Xem trên bot Telegram (SELL_BOT)

Bot kéo dữ liệu qua `GET /api/status` và điều khiển qua `POST /api/stop`, `POST /api/start` với header `X-API-Key: <DASH_API_KEY>`. Cấu hình trong `.env` của SELL_BOT:

```
COLABVPS_URL=https://<link-cloudflare-hien-tai>
COLABVPS_API_KEY=<DASH_API_KEY>
COLABVPS_ALERTS=true
```

Lệnh admin: `/train` (trạng thái, nút Làm mới, Log, Checkpoint, TẮT GPU, Chạy NB1/NB2, bật/tắt cảnh báo), `/trainlog`, `/trainurl <link>` (đổi link khi Cloudflare đổi, không cần sửa `.env`). Cảnh báo tự động gửi tới `ADMIN_ID`: kết thúc epoch, lập kỷ lục, runner dừng (NaN, hồi quy sớm, hết watchdog), GPU bật nhưng runner không chạy hơn 10 phút, watchdog còn dưới 10 phút, mất kết nối dashboard.