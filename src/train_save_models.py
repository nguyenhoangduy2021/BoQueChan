# -*- coding: utf-8 -*-
"""Huấn luyện lần cuối trên TOÀN BỘ dữ liệu và lưu mô hình vào models/.

Sản phẩm:
  models/ensemble_quetchan.joblib  — 4 pipeline (HGB, RF, SVM, LR) + cấu hình rank-averaging
Sản phẩm dùng kèm: src/predict.py để nạp và dự đoán.
"""
import os, sys, warnings
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier as HGB

warnings.filterwarnings("ignore")
RNG = 42
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import load_raw, build_features

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ld = load_raw(os.path.join(ROOT, "data", "locomotion_data.xlsx"))
X_dev, _, y, _ = build_features(ld)

def pipe(m, scaled):
    st = [("imp", SimpleImputer(strategy="median"))]
    if scaled:
        st.append(("sc", StandardScaler()))
    st.append(("clf", m))
    return Pipeline(st)

members = {}
for name, fn, scaled in [
    ("hgb", lambda: HGB(random_state=RNG, max_iter=300), False),
    ("rf", lambda: RandomForestClassifier(300, class_weight="balanced", random_state=RNG), False),
    ("svm", lambda: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RNG), True),
    ("lr", lambda: LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RNG), True)]:
    print("Đang huấn luyện", name, "...")
    members[name] = pipe(fn(), scaled).fit(X_dev.values, y)

model = {
    "members": members,                       # 4 pipeline hoàn chỉnh
    "fusion": "rank_average",                 # điểm = trung bình hạng của 4 mô hình
    "feature_names": list(X_dev.columns),
    "threshold_youden": 0.7334,               # ngưỡng sàng lọc (từ kết quả OOF)
    "label_map": {0: "khỏe", 1: "què (cần khám)"},
    "training_note": "Huấn luyện trên toàn bộ 4.467 bản ghi Zenodo 18462150 (209 bò). "
                     "Đánh giá độc lập cá thể: AUC 0.9044; recall tại ngưỡng Youden 82.1%.",
    "robustness": {"auc_8seeds": "0.9049 +/- 0.0005", "wilcoxon_p_one_sided": 0.0039,
                   "bootstrap_ci95_diff_auc": [0.0048, 0.0194],
                   "external_probe_cowscreeningdb": "random-split AUC 0.999 vs cow-independent AUC 0.737"},
}
out = os.path.join(ROOT, "models", "ensemble_quetchan.joblib")
os.makedirs(os.path.dirname(out), exist_ok=True)
joblib.dump(model, out)
print("Đã lưu", out, "(%.1f MB)" % (os.path.getsize(out) / 1e6))
