# 🐄 BoQueChan — Phát hiện sớm bệnh què chân ở bò sữa bằng học máy

Mã nguồn, dữ liệu và mô hình của đề tài luận văn thạc sĩ
**“Ứng dụng trí tuệ nhân tạo trong phát hiện sớm bệnh què chân ở bò sữa”**
(Công nghệ thông tin, mã ngành 8480201) — Đại học Cần Thơ.

- **Học viên:** Nguyễn Hoàng Duy (MSHV M5124033)
- **Hướng dẫn:** TS. Trương Minh Thái, PGS.TS. Hồ Thanh Thâm
- **Bài báo đi kèm:** *Cow-Independent Lameness Screening in Dairy Cattle Using Self-Referential Features and a Rank-Averaged Ensemble of Heterogeneous Classifiers*

## ✨ Điểm cải tiến chính (vòng v5)

| Cấu hình | AUC (độc lập cá thể) |
|---|---|
| HGB đơn — *chưa chỉnh sửa* | 0,8923 |
| Stacking xác suất (kiểu cũ) — *chưa chỉnh sửa* | 0,8835 |
| **Ensemble xếp hạng 4 mô hình (HGB+RF+SVM+LR) — ĐÃ CẢI TIẾN** | **0,9044** |
| Ensemble 8 mô hình tại ngưỡng sàng lọc | recall bò què **90,0%** |

Ý tưởng: (i) đặc trưng **tự tham chiếu** — mỗi giá trị đo lệch bao nhiêu so với mức nền
của chính con bò đó; (ii) **trộn theo xếp hạng** thay vì trộn xác suất, loại bỏ lệch
chuẩn định giữa các họ mô hình. Hai hướng thử nhưng bị loại (kèm lý do):
mã hóa Empirical-Bayes theo cá thể (rò rỉ nhãn) và giao thức per-cow (≈21 bản ghi/bò — quá thưa).

## 🔬 Kiểm định vững chắc & kiểm chứng ngoài (bổ sung 10/2026)

| Kiểm định (độc lập cá thể) | Kết quả |
|---|---|
| AUC ensemble trên 8 seeds | **0.9049 ± 0.0005** (HGB đơn 0.8923 mọi seed; Wilcoxon p = 0.0039) |
| CI bootstrap 95% chênh lệch AUC | [0.0048, 0.0194] (loại trừ 0) |
| Tăng cường nhiễu cảm biến 5% σ (dữ liệu tổng hợp, có khai báo) | ensemble 0.9036 (gần như bất biến), HGB đơn rớt còn 0.8712 |
| Đường cong học tập 40/60/80/100% số bò | 0.870 / 0.863 / 0.886 / 0.905 |
| Probe giao thức trên **CowScreeningDB** (41 bò, IMU, Tây Ban Nha) | random-split AUC 0.999 → cow-independent AUC 0.737: split lạc quan thổi phồng mọi mô hình |

Chạy lại: `python3 src/robustness_checks.py` (kết quả lưu `results/robustness_lameness.json`;
xem `figures/lame_robustness.png`). Mô hình lưu trong `models/` có nhúng sẵn các chỉ số này
trong khoá `robustness`.

## 📁 Cấu trúc kho

```
data/      locomotion_data.xlsx          bộ dữ liệu Zenodo 18462150 (CC-BY 4.0, 3,1 MB)
notebooks/ BoKhoe_Colab_SoSanh_QueChan.ipynb   chạy 1-click trên Google Colab
src/       features.py                   đặc trưng dùng chung
           experiment_baseline_and_improved.py  tái hiện đầy đủ chưa-sửa vs đã-sửa
           train_save_models.py          huấn luyện + lưu mô hình
           predict.py                    nạp mô hình và chấm điểm sàng lọc
models/    ensemble_quetchan.joblib      4 pipeline đã huấn luyện trên toàn bộ dữ liệu
results/   compare_lameness.json         kết quả máy đọc được
figures/   so sánh & đường cong ROC
```

## 🚀 Chạy nhanh

**Trên Google Colab (khuyến nghị, không cần cài gì):** mở
[notebooks/BoKhoe_Colab_SoSanh_QueChan.ipynb](notebooks/BoKhoe_Colab_SoSanh_QueChan.ipynb)
qua nút *Open in Colab* rồi Runtime → Run all (~5–10 phút).

**Trên máy cục bộ:**
```bash
pip install -r requirements.txt
python src/experiment_baseline_and_improved.py   # ~5 phút, in bảng so sánh
python src/train_save_models.py                  # huấn luyện và lưu mô hình
python src/predict.py                            # demo chấm điểm sàng lọc
```

## 📊 Kết quả chính

- Giao thức **độc lập cá thể** (GroupKFold 10 theo mã bò): mọi con bò chỉ nằm ở một phía học/chấm.
- SVM-RBF tái hiện theo phương pháp CowScreeningDB trên dữ liệu này: 89,5% acc / AUC 0,961
  (bài báo gốc đạt ≈77% trên dữ liệu của họ).
- Ensemble xếp hạng 4 mô hình: **AUC 0,9044**; tại ngưỡng Youden 0,733 bắt đúng 82,1% bò què;
  bản 8 mô hình tại ngưỡng sàng lọc đạt **recall 90,0%**.
- Toàn bộ số liệu: `results/compare_lameness.json`; hình: `figures/`.

## 📎 Dữ liệu & trích dẫn

Bộ dữ liệu: Kovács, L. (2026). *Production, physiological (HRV) and behavioral data
(locomotion scores) of dairy cows.* Zenodo record 18462150, giấy phép **CC-BY 4.0**,
doi:[10.5281/zenodo.18462150](https://doi.org/10.5281/zenodo.18462150).
File được lưu kèm trong `data/` để thuận tiện; khi dùng lại xin trích dẫn nguồn gốc.

Nếu bạn sử dụng mã nguồn của kho này, xin trích dẫn đề tài/bài báo tương ứng ở trên.

## 📄 Giấy phép

Mã nguồn: giấy phép MIT (xem `LICENSE`). Dữ liệu: CC-BY 4.0 thuộc tác giả gốc (Zenodo).
