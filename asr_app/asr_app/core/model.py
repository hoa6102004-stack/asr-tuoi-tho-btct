"""Huấn luyện, đánh giá và lưu mô hình dự đoán độ giãn nở ASR (Chương 6-7).

- Chia tập THEO CẤP PHỐI (GroupShuffleSplit) để tránh rò rỉ giữa các mốc thời gian.
- So sánh Linear Regression, Random Forest, XGBoost, ANN (MLP).
- Tiền xử lý (chuẩn hóa) được đóng gói trong Pipeline, chỉ fit trên tập huấn luyện.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.data_gen import FEATURES, RANGES, TARGET, generate

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # xgboost chưa cài
    HAS_XGB = False

SEED = 42
MODEL_DIR = "models"
DATA_DIR = "data"


def load_data():
    """Ưu tiên dữ liệu thật nếu có; nếu không dùng dữ liệu mô phỏng."""
    real = os.path.join(DATA_DIR, "real_asr.csv")
    if os.path.exists(real):
        df = pd.read_csv(real)
        df["source"] = df.get("source", "REAL")
        return df, "REAL"
    sim = os.path.join(DATA_DIR, "simulated_asr.csv")
    if not os.path.exists(sim):
        os.makedirs(DATA_DIR, exist_ok=True)
        generate().to_csv(sim, index=False)
    return pd.read_csv(sim), "SIMULATED"


def build_models():
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "ANN (MLP)": TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=3000,
                             learning_rate_init=3e-3, n_iter_no_change=40,
                             random_state=SEED)),
            transformer=StandardScaler()),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, random_state=SEED, n_jobs=2)
    return models


def metrics(y, p):
    return {
        "R2": float(r2_score(y, p)),
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
    }


def split(df):
    groups = df["mix_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    trval, test = next(gss.split(df, groups=groups))
    df_trval, df_test = df.iloc[trval], df.iloc[test]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + 1)
    tr, val = next(gss2.split(df_trval, groups=df_trval["mix_id"]))
    return df_trval.iloc[tr], df_trval.iloc[val], df_test


def train_all():
    df, source = load_data()
    df = df.drop_duplicates()
    df = df[(df[TARGET] >= 0) & df[FEATURES].notna().all(axis=1)]
    tr, val, te = split(df)

    results, fitted = [], {}
    for name, m in build_models().items():
        t0 = time.time()
        m.fit(tr[FEATURES], tr[TARGET])
        r = {"model": name}
        for tag, part in (("val", val), ("test", te)):
            for k, v in metrics(part[TARGET], m.predict(part[FEATURES])).items():
                r[f"{tag}_{k}"] = v
        r["train_sec"] = round(time.time() - t0, 2)
        results.append(r)
        fitted[name] = m

    res = pd.DataFrame(results).sort_values("val_RMSE")   # chọn theo tập KIỂM ĐỊNH
    best = res.iloc[0]["model"]

    # Miền dữ liệu huấn luyện để cảnh báo ngoại suy (6.10, 9.7)
    domain = {f: [float(df[f].min()), float(df[f].max())] for f in FEATURES}

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(fitted, os.path.join(MODEL_DIR, "models.joblib"))
    meta = {
        "version": time.strftime("%Y.%m.%d-%H%M"),
        "data_source": source,
        "n_rows": int(len(df)), "n_mixes": int(df["mix_id"].nunique()),
        "split": {"train_mixes": int(tr["mix_id"].nunique()),
                  "val_mixes": int(val["mix_id"].nunique()),
                  "test_mixes": int(te["mix_id"].nunique()), "seed": SEED},
        "features": FEATURES, "target": TARGET, "units": "% (độ giãn nở)",
        "domain": domain, "best_model": best,
        "default_ranges": RANGES,
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    res.to_csv(os.path.join(MODEL_DIR, "comparison.csv"), index=False)
    te.assign(**{f"pred_{n}": m.predict(te[FEATURES]) for n, m in fitted.items()}) \
      .to_csv(os.path.join(MODEL_DIR, "test_predictions.csv"), index=False)
    return res, meta


def load_artifacts():
    models = joblib.load(os.path.join(MODEL_DIR, "models.joblib"))
    with open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return models, meta


def predict_curve(model, params: dict, times):
    X = pd.DataFrame([{**params, "age_days": float(t)} for t in times])[FEATURES]
    return np.clip(model.predict(X), 0, None)


def out_of_domain(params: dict, meta: dict):
    msgs = []
    for f, (lo, hi) in meta["domain"].items():
        if f in params and not (lo <= params[f] <= hi):
            msgs.append(f"{f} = {params[f]:.3g} nằm ngoài miền huấn luyện [{lo:.3g}; {hi:.3g}]")
    return msgs


if __name__ == "__main__":
    res, meta = train_all()
    print(res.round(4).to_string(index=False))
    print("Mô hình tốt nhất (theo kiểm định):", meta["best_model"])
