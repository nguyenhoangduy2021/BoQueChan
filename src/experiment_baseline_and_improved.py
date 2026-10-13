# -*- coding: utf-8 -*-
"""Tái hiện đầy đủ: mô hình CHƯA chỉnh sửa vs ĐÃ cải tiến (đề tài què chân).

Giao thức: độc lập cá thể — GroupKFold 10 theo mã bò, mỗi con bò chỉ nằm
trọn ở một phía huấn luyện/chấm điểm (đúng với điều kiện triển khai thật).

Chạy:  python src/experiment_baseline_and_improved.py
Thời gian: ~5 phút trên CPU. Kết quả in ra trùng results/compare_lameness.json.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier as HGB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, recall_score

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import load_raw, build_features

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "locomotion_data.xlsx")
if not os.path.exists(DATA):  # tự tải nếu chưa có
    import requests
    print("Chưa có dữ liệu — đang tải từ Zenodo...")
    url = "https://zenodo.org/records/18462150/files/Physiological_and_behavioral_data_2024.xlsx"
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    r = requests.get(url, timeout=180); r.raise_for_status()
    open(DATA, "wb").write(r.content)

ld = load_raw(DATA)
X_dev, X_cownorm, y, groups = build_features(ld)
print("Dữ liệu: %d bản ghi, %d bò, tỷ lệ què %.1f%%" % (len(y), len(set(groups)), 100 * y.mean()))
print("Đặc trưng X_dev:", X_dev.shape[1], "| X_cownorm:", X_cownorm.shape[1])


def pipe(m, scaled):
    st = [("imp", SimpleImputer(strategy="median"))]
    if scaled:
        st.append(("sc", StandardScaler()))
    st.append(("clf", m))
    return Pipeline(st)


def rank_avg(*arrs):
    n = len(arrs[0])
    return np.mean([rankdata(a) / (n + 1) for a in arrs], axis=0)


def youden(y_true, scores):
    fpr, tpr, ths = roc_curve(y_true, scores)
    t = ths[np.argmax(tpr - fpr)]
    return t, recall_score(y_true, scores > t), accuracy_score(y_true, scores > t), f1_score(y_true, scores > t)


folds = list(GroupKFold(10).split(X_dev, y, groups))

# Xác suất ngoài-fold của 4 mô hình nền
oof = {n: np.zeros(len(y)) for n in ["hgb", "rf", "svm", "lr"]}
for tr, te in folds:
    for name, fn, scaled in [
        ("hgb", lambda: HGB(random_state=RNG, max_iter=300), False),
        ("rf", lambda: RandomForestClassifier(300, class_weight="balanced", random_state=RNG), False),
        ("svm", lambda: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RNG), True),
        ("lr", lambda: LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RNG), True)]:
        m = pipe(fn(), scaled)
        m.fit(X_dev.values[tr], y[tr])
        oof[name][te] = m.predict_proba(X_dev.values[te])[:, 1]

res = {}

# ---------- CHƯA chỉnh sửa ----------
m0 = oof["hgb"]
res["M0_hgb_dev"] = {"acc": accuracy_score(y, m0 > .5), "f1": f1_score(y, m0 > .5), "auc": roc_auc_score(y, m0)}

oof_stack = np.zeros(len(y))
for tr, te in folds:
    Ztr = np.column_stack([oof[k][tr] for k in oof]); Zte = np.column_stack([oof[k][te] for k in oof])
    oof_stack[te] = LogisticRegression(max_iter=2000).fit(Ztr, y[tr]).predict_proba(Zte)[:, 1]
res["C4_stack"] = {"acc": accuracy_score(y, oof_stack > .5), "f1": f1_score(y, oof_stack > .5),
                   "auc": roc_auc_score(y, oof_stack)}

# ---------- ĐÃ cải tiến: ensemble xếp hạng ----------
ens4 = rank_avg(oof["hgb"], oof["rf"], oof["svm"], oof["lr"])
res["C3_rank_ens"] = {"acc": accuracy_score(y, ens4 > .5), "f1": f1_score(y, ens4 > .5),
                      "auc": roc_auc_score(y, ens4)}
thr, rec, acc_y, f1_y = youden(y, ens4)
res["ens4_youden"] = {"thr": float(thr), "recall_lame": float(rec), "acc": float(acc_y), "f1": float(f1_y),
                      "auc": res["C3_rank_ens"]["auc"]}

print("\n%-34s %8s %8s %8s" % ("Cấu hình (độc lập cá thể)", "Acc", "F1", "AUC"))
for k in ["M0_hgb_dev", "C4_stack", "C3_rank_ens"]:
    d = res[k]
    print("%-34s %8.4f %8.4f %8.4f" % (k, d["acc"], d["f1"], d["auc"]))
print("\nENS4 ngưỡng Youden = %.3f -> recall bò què %.1f%% | acc %.4f | f1 %.4f"
      % (thr, 100 * rec, acc_y, f1_y))

out = os.path.join(ROOT, "results", "compare_lameness.json")
json.dump(res, open(out, "w"), indent=1)
print("\nĐã lưu", out)
