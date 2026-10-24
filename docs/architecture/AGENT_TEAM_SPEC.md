# ĐẶC TẢ ĐỘI AGENT - SE374.R11

> Trạng thái: đã tạo file cấu hình, chưa chạy thử một chu kỳ thật nào.
> Ngày tạo: 2026-10-24.

---

## 1. Mô hình tổng thể

Một chu kỳ làm việc gồm **1 Leader + 3 pool động**:

| Thành phần | Số lượng | Vai trò | Nguyên tắc |
| :--- | :---: | :--- | :--- |
| Leader | 1 | Đóng gói task, gọi pool, đếm phiếu, ra Fix Directive | Không sửa code |
| Research pool | 4 đến 10 | Trả lời cùng một câu hỏi từ nhiều nguồn | Dư thừa có chủ đích |
| Impl pool | 4 đến 10 | Sửa code theo Fix Directive | Chia scope, không trùng file |
| Check pool | 4 đến 10 | Kiểm cùng một artifact từ nhiều lập trường | Dư thừa có chủ đích |

Tổng tối thiểu: 1 + 4 + 4 + 4 = **13 agent**. Tối đa theo thiết kế hiện tại: **31 agent**.

---

## 2. Vì sao research và check chạy dư thừa, còn impl thì không

Đây là điểm khác biệt quan trọng nhất so với mô hình chia để trị.

**Research và check** làm việc trên dữ liệu đọc, không ghi. Nhiều agent cùng đọc một
artifact không phá nhau, nên chạy dư thừa được. Giá trị nằm ở chỗ: khi bốn agent
độc lập cùng kết luận một điều, độ tin cậy cao hơn hẳn một agent. Khi họ lệch nhau,
chỗ lệch chính là tín hiệu cần điều tra.

**Impl** làm việc trên file. Bốn agent cùng sửa một file sẽ ghi đè lên nhau, tạo ra
kết quả không agent nào chịu trách nhiệm. Vì vậy impl chia theo scope thư mục, mỗi
file có đúng một chủ. Nếu hai mục của Fix Directive cùng chạm một file, Leader phải
giao cả hai cho cùng một agent.

Nói cách khác: research và check dùng **N-version programming**, impl dùng
**partition by write scope**.

---

## 3. Danh mục file agent

Tất cả nằm trong `.opencode/agents/`. Đường dẫn lồng tạo thành agent ID.

| ID | File | Mode | Được ghi | Được đọc chính |
| :--- | :--- | :--- | :--- | :--- |
| `leader` | `leader.md` | primary | `docs/diary/**` | toàn repo |
| `research/paper` | `research/paper.md` | subagent | `docs/diary/audit/**` | `docs/papers/**` |
| `research/repo` | `research/repo.md` | subagent | `docs/diary/audit/**` | `repos/**` |
| `research/docs` | `research/docs.md` | subagent | `docs/diary/audit/**` | docs chính thức qua webfetch |
| `research/logs` | `research/logs.md` | subagent | `docs/diary/audit/**` | `experiments/**` |
| `research/spec` | `research/spec.md` | subagent | `docs/diary/audit/**` | `AGENTS.md`, `product/**` |
| `impl/data` | `impl/data.md` | subagent | `src/datasets`, `src/config`, `downloader.py` | toàn repo |
| `impl/model` | `impl/model.md` | subagent | `src/models`, `src/losses_metrics` | toàn repo |
| `impl/pipeline` | `impl/pipeline.md` | subagent | `src/pipeline`, `src/colab_pkg`, `scripts` | toàn repo |
| `impl/eval` | `impl/eval.md` | subagent | `tests`, `src/utils`, `src/visualization`, `experiments` | toàn repo |
| `check/prover` | `check/prover.md` | subagent | `docs/diary/audit/**` | toàn repo |
| `check/falsifier` | `check/falsifier.md` | subagent | `docs/diary/audit/**` | toàn repo |
| `check/breaker` | `check/breaker.md` | subagent | `docs/diary/audit/**` | toàn repo |
| `check/clinician` | `check/clinician.md` | subagent | `docs/diary/audit/**` | `product/**`, `experiments/**` |

Ghi chú: 14 định nghĩa, nhưng workflow tối thiểu chỉ dùng 13 agent. `research/spec`
là slot dự trữ, chỉ dùng khi pool research từ 6 trở lên hoặc khi cần kiểm governance.

### Ranh giới quyền

- Leader: `edit` deny toàn bộ, chỉ allow `docs/diary/**`. `subagent` deny `*`, allow
  `research/*`, `impl/*`, `check/*`. `shell` ask, trừ `git status`, `git diff`,
  `git rev-parse`, `pytest`.
- Research và check: `edit` deny toàn bộ, chỉ allow `docs/diary/audit/**`. Không có
  quyền tạo subagent.
- Impl: `edit` deny toàn bộ, allow đúng scope của mình. `shell` ask, riêng `pytest`
  và `git status`/`git diff` thì allow. Mọi lệnh git làm thay đổi trạng thái đều deny.

Quy tắc khớp cuối cùng thắng, nên mọi rule rộng được đặt trước, ngoại lệ đặt sau.

---

## 4. Cách chọn cỡ pool

| Cỡ | Điều kiện | Research slots | Check slots | Impl | Tổng |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 4 | Sửa ≤ 2 file, không đụng số liệu sẽ công bố | paper, repo, docs, logs | prover, falsifier, breaker, clinician | 4 scope cố định | 13 |
| 6 | Sửa 3-10 file, hoặc đụng losses/metrics/split | thêm paper-challenge, repo-challenge | thêm prover(logs), falsifier(docs) | chia nhỏ scope | 19 |
| 8 | Đụng dữ liệu bệnh nhân, hoặc số liệu vào báo cáo/slide | thêm docs-challenge, logs-challenge | thêm breaker(logs), clinician(code) | chia nhỏ scope | 25 |
| 10 | Kết quả đem nộp/công bố, > 10 file, hoặc đang tranh chấp | thêm spec, spec-challenge | thêm 2 slot đổi model | chia nhỏ scope | 31 |

Quy tắc này là bảng cố định, không để Leader tự chọn. Nếu để tự chọn, nó sẽ có xu
hướng chọn nhỏ cho nhanh.

---

## 5. Trần hữu ích của từng pool

Tăng số agent chỉ có giá trị nếu mỗi agent khác nhau thật. Trần hữu ích bằng số tổ
hợp (nguồn bằng chứng × lập trường).

| Pool | Trục | Số tổ hợp | Ý nghĩa |
| :--- | :--- | :---: | :--- |
| Research | 5 nguồn × 2 lập trường | 10 | Vượt 10 là bản sao |
| Check | 4 lập trường × 3 nguồn | 12 | Vượt 12 là bản sao |
| Impl | Số thư mục độc lập có thể sửa | ~10 | Vượt là trùng file |

Nếu một pool vượt trần, thêm agent chỉ tốn token mà không tăng tính độc lập. Hiện
tượng này gọi là **correlated failure**: bốn agent cùng đọc một file sai sẽ cùng sai
theo một kiểu, và bất đồng bằng 0 không phải bằng chứng đúng.

---

## 6. Hợp đồng đầu ra

Mọi agent con trả về **đúng một tin nhắn** chứa JSON, đồng thời ghi cùng nội dung
vào `docs/diary/audit/<cycle>/<pool>/<slot>.json`.

```json
{
  "agent": "check/falsifier",
  "slot": "falsifier",
  "task_id": "T-2026-10-24-01",
  "snapshot": "a1b2c3d",
  "verdict": "PASS | FAIL | INCONCLUSIVE | SKIPPED",
  "summary": "2-4 cau",
  "findings": [
    {
      "id": "F1",
      "target_id": "C3",
      "severity": "critical | major | minor",
      "statement": "dieu khang dinh",
      "evidence": "src/x.py:120",
      "confidence": 0.9
    }
  ],
  "open_questions": []
}
```

### Vai trò của `target_id`

Đây là khóa để đếm phiếu, và là lý do schema có trường này thay vì để Leader tự
đoán hai finding có giống nhau không.

- Nếu finding nói về một claim đã có trong task packet: dùng đúng mã `C1..Cn`.
- Nếu là phát hiện mới: dùng dạng `NEW:<path>:<line>`.

Nhờ vậy, việc đếm phiếu là **so khớp chuỗi chính xác**, không phụ thuộc model, và
tái lập được. Đây là cổng chặn khách quan duy nhất trong toàn bộ workflow.

### Quy tắc lọc

| Trường hợp | Xử lý |
| :--- | :--- |
| `evidence` rỗng | Loại, không báo lên |
| `confidence` < 0.7 | Hạ xuống mục "chưa đủ đồng thuận", không bỏ |
| `verdict` lệch giữa các agent | Là finding riêng, phải ghi rõ |
| `verdict = SKIPPED` | Không tính vào mẫu số khi đếm phiếu |

Ngưỡng đồng thuận: **≥ 3 phiếu** với pool 4 hoặc 6; **≥ 50% pool** với pool 8 hoặc 10.

---

## 7. Chuỗi chạy

```
Bước 0  Đóng băng trạng thái: ghi commit hash và git status vào 00_snapshot.md
Bước 1  Đóng gói task packet + claim registry (C1..Cn)
Bước 2  Chọn cỡ pool theo bảng mục 4
Bước 3  Research pool chạy song song, cùng packet, khác slot
Bước 4  Leader đọc báo cáo research, lập danh sách claim kèm bằng chứng
Bước 5  Impl pool chạy song song, chia scope, không trùng file
Bước 6  Check pool chạy song song, cùng artifact, khác lập trường
Bước 7  Đối chiếu chéo research với check, chỗ lệch là finding
Bước 8  Đếm phiếu theo target_id
Bước 9  Ra Fix Directive, quay lại Bước 6, tối đa 3 vòng
Bước 10 Tất cả PASS: xuất báo cáo và DỪNG
```

Điểm dừng bắt buộc: sau báo cáo Markdown, Leader không được tự chuyển giai đoạn.

---

## 8. Giới hạn kỹ thuật của OpenCode và cách xử lý

| Giới hạn | Hệ quả | Cách xử lý đã áp dụng |
| :--- | :--- | :--- |
| Độ sâu lồng agent tối đa 1 | Agent con không gọi được agent cháu | Leader làm mọi việc chia nhỏ |
| Agent con chỉ trả một tin nhắn | Không có hội thoại nhiều lượt | Ép đầu ra JSON, ghi file làm blackboard |
| Không có kênh peer-to-peer | Các checker không nói chuyện trực tiếp | Đối chiếu qua Leader và file, không qua chat |
| Không có state bền vững | Phiên chết là mất | Mọi agent ghi `docs/diary/audit/` |
| Không có task board dùng chung | Không hàng đợi công việc | Leader gọi tuần tự theo vòng |

---

## 9. Cách kích hoạt

1. Mở phiên mới hoặc khởi động lại app. Agent được nạp lúc tạo phiên, nên phiên
   đang mở sẽ không thấy file mới.
2. Gõ `/agents` để kiểm tra danh sách. Phải thấy `leader` ở nhóm primary và 13
   subagent dưới ba namespace `research/`, `impl/`, `check/`.
3. Chọn `leader` làm agent chính.
4. Giao việc bằng một câu, ví dụ: "Chạy một chu kỳ kiểm định cho 5 thí nghiệm
   trong `experiments/colab_run_3_multiscale_sota_v2/`, cỡ pool 8."

Nếu `/agents` không thấy agent mới, kiểm tra app đã được khởi động lại hoàn toàn
chưa, và thư mục hiện tại có đúng là gốc dự án chứa `.opencode/` không.

---

## 10. Việc còn thiếu, cần người dùng duyệt

| Việc | Vì sao cần | Trạng thái |
| :--- | :--- | :--- |
| Script đếm phiếu `scripts/consensus_vote.py` | Pool 8-10 sinh 25-31 báo cáo, đếm tay dễ sai | **Chưa tạo**, cần duyệt |
| `opencode.jsonc` cấp dự án | Khóa quyền toàn cục, chặn sửa `product/`, `docs/papers/` | **Chưa tạo**, cần duyệt |
| Sửa `INVARIANT-26` từ "6 đến 10" thành "4 đến 10" | Khớp pool tối thiểu 4 | **Chưa sửa**, cần duyệt |
| Khai báo `model:` riêng cho từng pool | Giảm correlated failure | **Chưa làm**, cần người dùng chọn model |
| Chạy thử một chu kỳ cỡ 4 trên artifact nhỏ | Kiểm chứng workflow trước khi dùng thật | **Chưa chạy** |

---

## 11. Điều phải chấp nhận

| Rủi ro | Cách giảm |
| :--- | :--- |
| Agent chấm chủ quan, không tất định | Cổng cuối vẫn là `pytest` và `target_id` đếm được |
| Tốn token | Scope hẹp, giao danh sách file cụ thể |
| Model free yếu, chấm nông | Tăng `steps`, chia task nhỏ, đổi model cho pool check |
| Bốn agent cùng sai | Đa dạng nguồn, lập trường, model; báo người dùng khi pool trùng nguồn |
| Impl sửa lệch scope | Quyền `edit` chặn ở tầng cấu hình, không chỉ ở lời dặn |
