# Experimental Benchmarks & Evaluation Evolution

Thư mục này lưu trữ kết quả đối chiếu và phân tích định lượng của 3 lần chạy thực nghiệm thật trên toàn bộ tập dữ liệu BCSS-WSSS (4.986 ảnh test, 23 bệnh nhân):

- [Báo cáo đối chiếu chi tiết 3 lần chạy thật](file:///experiments/benchmarks/COLAB_3_FULL_RUNS_EVOLUTION_COMPARISON.md)

## Tóm tắt 3 phiên chạy thực nghiệm thật:
1. **Lần 1 (`experiments/colab_run_1_stage2_baseline/`)**: Chạy toàn bộ 4.986 ảnh test với Stage-2 DeepLabV3+ đơn lẻ (Baseline: 68.71% mIoU).
2. **Lần 2 (`experiments/colab_run_2_dual_stage_ensemble/`)**: Chạy toàn bộ 4.986 ảnh test với kiến trúc Dual-Stage tích hợp Stage-1 CGM và đánh giá 5 phương pháp `M1..M5` (Đạt 70.49% mIoU).
3. **Lần 3 (`experiments/colab_run_3_multiscale_sota_v2/`)**: Chạy toàn bộ 4.986 ảnh test qua Thin Drive Runner v2, đánh giá 7 phương pháp `M1..M7` với Mixed Precision FP16 và Multi-Scale Pyramid D4 Ensemble (Đạt đỉnh 70.53% mIoU, vượt trội SOTA quốc tế).
