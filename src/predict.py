# -*- coding: utf-8 -*-
"""Nạp mô hình đã lưu và chấm điểm sàng lọc què chân.

Cách dùng:
  python src/predict.py                      # demo trên 5 bản ghi mẫu từ bộ dữ liệu
  python src/predict.py du_lieu_cua_ban.xlsx # chấm toàn bộ file của bạn (cùng cấu trúc cột Zenodo)
"""
import os, sys
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import load_raw, build_features

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

model = joblib.load(os.path.join(ROOT, "models", "ensemble_quetchan.joblib"))
members = model["members"]
thr = model["threshold_youden"]

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "locomotion_data.xlsx")
ld = load_raw(path)
X_dev, _, y, groups = build_features(ld)

probs = {name: m.predict_proba(X_dev.values)[:, 1] for name, m in members.items()}
n = len(X_dev)
score = np.mean([rankdata(p) / (n + 1) for p in probs.values()], axis=0)
pred = (score > thr).astype(int)

df = pd.DataFrame({"cow": groups, "score_rank_ensemble": score.round(4),
                   "du_doan": [model["label_map"][v] for v in pred],
                   "nhan_that": y})
print(df.head(5).to_string(index=False))
tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
print("(Lưu ý: chấm trên chính dữ liệu huấn luyện nên recall ở đây cao hơn thực tế;\n số độc lập cá thể trung thực: recall 82,1% tại ngưỡng Youden — xem README.)")
print("\nTrên toàn bộ %d bản ghi: recall tại ngưỡng %.3f = %.1f%% (bắt %d, sót %d con què)"
      % (len(df), thr, 100 * tp / max(tp + fn, 1), tp, fn))
