# -*- coding: utf-8 -*-
"""Đặc trưng dùng chung cho đề tài phát hiện sớm què chân ở bò sữa.

Đặc trưng tự tham chiếu (self-referential): mỗi giá trị được biểu diễn
dưới dạng độ lệch so với mức nền của CHÍNH con bò đó — thể hiện ý tưởng
lâm sàng "so với bình thường của chính nó" mà vẫn tổng quát được sang bò lạ.
"""
import numpy as np
import pandas as pd

NUM = ["Laktáció száma", "Tejtermelés (kg) vizsgálat idején", "Kondíció",
       "Mean RR  (ms)", "STD RR (ms)", "Mean HR (1/min)", "RMSSD (ms)", "NN50 (count)", "pNN50 (%)",
       "RR tri index", "TINN (ms)", "LF (ms2)", "HF (ms2)", "VLF (%)", "LF (%)", "HF (%)",
       "LF (n.u.)", "HF (n.u.)", "LF/HF ratio", "SD1 (ms)", "SD2 (ms)", "SD2/SD1",
       "Mean line length (beats)"]


def load_raw(path):
    """Đọc file dữ liệu Zenodo (sheet Munka1) và làm sạch."""
    ld = pd.read_excel(path, sheet_name="Munka1")
    ld = ld.drop(columns=["Szívritmus paraméterek"], errors="ignore")
    ld.columns = [c.strip() for c in ld.columns]
    sc = [c for c in ld.columns if c.startswith("Mozgáskép")][0]
    ld = ld.rename(columns={sc: "score", "Fülszám": "cow", "Vizsgálat időpontja": "date"})
    ld["cow"] = ld["cow"].astype(str)
    ld = ld.dropna(subset=["score"])
    ld["y"] = (ld["score"] >= 2).astype(int)  # điểm vận động >= 2: QUÈ
    return ld


def base_behavior(s):
    if not isinstance(s, str):
        return "other"
    if s.startswith("Fekve kérődzés"):
        return "lr"
    if s.startswith("Állva kérődzés"):
        return "sr"
    if s.startswith("Fekvés"):
        return "lying"
    if s.startswith("Étkezés"):
        return "eat"
    if s.startswith("Állás"):
        return "stand"
    return "other"


def build_features(ld):
    """Trả về (X_dev, X_cownorm, y, groups).

    X_dev     : 23 đặc trưng thô + ngữ cảnh + hệ thống vắt sữa
                + 23 đặc trưng tự tham chiếu + sữa so với đàn cùng ngày.
    X_cownorm : chỉ đặc trưng tự tham chiếu + ngữ cảnh + sữa so đàn.
    """
    num = [c for c in NUM if c in ld.columns]
    beh = pd.get_dummies(ld["Eseményszótár"].map(base_behavior), prefix="b").astype(float)
    ms = pd.get_dummies(ld["Fejési rendszer kód"].astype(str), prefix="ms").astype(float)

    cow_med = ld.groupby("cow")[num].transform("median")
    cow_std = ld.groupby("cow")[num].transform("std").replace(0, np.nan).fillna(1)
    dev = (ld[num] - cow_med) / cow_std
    dev.columns = ["dev_" + c for c in num]

    herd_med = ld.groupby("date")["Tejtermelés (kg) vizsgálat idején"].transform("median")
    ld = ld.assign(milk_rel_herd=ld["Tejtermelés (kg) vizsgálat idején"] - herd_med)

    X_dev = pd.concat([ld[num].astype(float), beh, ms, dev, ld[["milk_rel_herd"]]], axis=1)
    X_cownorm = pd.concat([dev, beh, ms, ld[["milk_rel_herd"]]], axis=1)
    return X_dev, X_cownorm, ld["y"].values, ld["cow"].values
