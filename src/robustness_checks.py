# -*- coding: utf-8 -*-
"""Gói bằng chứng thống kê bổ sung (đề tài què chân) — đáp ứng yêu cầu tạp chí:
  (1) Ổn định nhiều seed: mean±SD AUC, kiểm định Wilcoxon (ensemble vs baseline)
  (2) Khoảng tin cậy bootstrap 95% cho AUC và chênh lệch AUC
  (3) Ablation tăng cường dữ liệu nhiễu cảm biến (khai báo minh bạch)
  (4) Đường cong học tập theo số cá thể (biện minh cỡ mẫu)
Kết quả: thesis/results/robustness_lameness.json
"""
import os, sys, json, warnings, time
import numpy as np
from scipy.stats import rankdata, wilcoxon
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import load_raw, build_features

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "locomotion_data.xlsx")
ld = load_raw(DATA)
X_dev, X_cownorm, y, groups = build_features(ld)
X = X_dev.values.astype(float)
col_sd = np.nanstd(X, axis=0); col_sd[col_sd == 0] = 1
print("n=%d, bò=%d, què=%.1f%%" % (len(y), len(set(groups)), 100 * y.mean()))


def pipe(m, scaled):
    st = [("imp", SimpleImputer(strategy="median"))]
    if scaled: st.append(("sc", StandardScaler()))
    st.append(("clf", m)); return Pipeline(st)

MEMBERS = [("hgb", lambda: HGB(random_state=0, max_iter=300), False),
           ("rf",  lambda: RandomForestClassifier(300, class_weight="balanced", random_state=0), False),
           ("svm", lambda: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=0), True),
           ("lr",  lambda: LogisticRegression(max_iter=5000, class_weight="balanced", random_state=0), True)]

def oof_scores(idx, seed, augment=False):
    """OOF xác suất 4 thành viên trên tập con idx (đảo thứ tự theo seed)."""
    rng = np.random.RandomState(seed)
    idx = np.array(idx)
    sub = idx[rng.permutation(len(idx))]
    g, yy, XX = groups[sub], y[sub], X[sub]
    folds = list(GroupKFold(10).split(XX, yy, g))
    P = {n: np.zeros(len(sub)) for n, _, _ in MEMBERS}
    for tr, te in folds:
        Xtr, ytr = XX[tr], yy[tr]
        if augment:  # 2 bản sao nhiễu Gauss (sigma = 5% SD cột) cho tập huấn luyện
            n1 = Xtr + rng.randn(*Xtr.shape) * (0.05 * col_sd)
            n2 = Xtr + rng.randn(*Xtr.shape) * (0.05 * col_sd)
            Xtr = np.vstack([Xtr, n1, n2]); ytr = np.concatenate([ytr, ytr, ytr])
        for name, fn, scaled in MEMBERS:
            m = pipe(fn(), scaled); m.fit(Xtr, ytr)
            P[name][te] = m.predict_proba(XX[te])[:, 1]
    return P, sub

def auc_of(P, sub):
    yy = y[sub]
    ens = np.mean([rankdata(P[n]) / (len(yy) + 1) for n, _, _ in MEMBERS], axis=0)
    return roc_auc_score(yy, P["hgb"]), roc_auc_score(yy, ens)

t0 = time.time()
# ---------- (1) nhiều seed ----------
SEEDS = [42, 7, 123, 2024, 2026, 1, 2, 3]
a_hgb, a_ens = [], []
for s in SEEDS:
    P, sub = oof_scores(range(len(y)), s)
    h, e = auc_of(P, sub); a_hgb.append(h); a_ens.append(e)
    print("seed %5d  HGB %.4f  rankEns %.4f" % (s, h, e))
w_stat, p_wil = wilcoxon(a_ens, a_hgb, alternative="greater")

# ---------- (2) bootstrap CI (seed 42) ----------
P42, sub42 = oof_scores(range(len(y)), 42)
yy = y[sub42]
ens42 = np.mean([rankdata(P42[n]) / (len(yy) + 1) for n, _, _ in MEMBERS], axis=0)
rng = np.random.RandomState(0)
b_e, b_h, b_d = [], [], []
for _ in range(2000):
    i = rng.randint(0, len(yy), len(yy))
    if len(set(yy[i])) < 2: continue
    be = roc_auc_score(yy[i], ens42[i]); bh = roc_auc_score(yy[i], P42["hgb"][i])
    b_e.append(be); b_h.append(bh); b_d.append(be - bh)
ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

# ---------- (3) ablation augmentation ----------
h_no, e_no = auc_of(P42, sub42)
P_aug, suba = oof_scores(range(len(y)), 42, augment=True)
h_aug, e_aug = auc_of(P_aug, suba)

# ---------- (4) learning curve ----------
curve = {}
for frac in [0.4, 0.6, 0.8, 1.0]:
    vals = []
    for s in [42, 7, 123]:
        r2 = np.random.RandomState(s)
        cows = np.array(sorted(set(groups))); r2.shuffle(cows)
        keep = set(cows[: int(frac * len(cows))])
        idx = [i for i in range(len(y)) if groups[i] in keep]
        P, sub = oof_scores(idx, s)
        _, e = auc_of(P, sub); vals.append(e)
    curve[str(frac)] = round(float(np.mean(vals)), 4)
    print("cows %3.0f%% -> AUC ens %.4f" % (100 * frac, float(np.mean(vals))))

res = {
 "seeds": {str(s): {"hgb": round(a_hgb[i], 4), "rank_ens": round(a_ens[i], 4)} for i, s in enumerate(SEEDS)},
 "mean_sd": {"hgb_auc": [round(float(np.mean(a_hgb)), 4), round(float(np.std(a_hgb)), 4)],
             "rank_ens_auc": [round(float(np.mean(a_ens)), 4), round(float(np.std(a_ens)), 4)]},
 "wilcoxon_ens_vs_hgb_p_one_sided": float(p_wil),
 "bootstrap_ci95": {"rank_ens_auc": ci(b_e), "hgb_auc": ci(b_h), "diff_auc": ci(b_d)},
 "augmentation_noise5pct": {"no_aug": {"hgb": round(h_no, 4), "ens": round(e_no, 4)},
                            "with_aug": {"hgb": round(h_aug, 4), "ens": round(e_aug, 4)}},
 "learning_curve_cow_fraction": curve,
}
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "robustness_lameness.json"), "w"), indent=1)
print("saved | %.0fs" % (time.time() - t0))
