"""Train one LightGBM model per forecast horizon (2, 3, 4 weeks ahead).

Run:  python -m src.train
Outputs (in models/): model_h{2,3,4}.joblib, meta.joblib, metrics.json, backtest_test.parquet
"""
import json
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from .features import load_raw, build_features, add_horizon, feature_columns
from .tiers import compute_thresholds, to_tier_idx

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")

PARAMS = dict(objective="poisson", learning_rate=0.05, num_leaves=31,
              min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
              n_jobs=2, verbose=-1)


def split_masks(df):
    d = df["date"]
    train = d <= C.TRAIN_END
    val = (d > C.TRAIN_END) & (d <= C.VAL_END)
    test = d > C.VAL_END
    return train, val, test


def fit_model(X_tr, y_tr, X_va, y_va):
    m = lgb.LGBMRegressor(n_estimators=2000, **PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="l1",
          callbacks=[lgb.early_stopping(60, verbose=False)])
    return m


def interval_table(pred_va, y_va):
    """Split-conformal half-widths by predicted-magnitude bin (80% intervals)."""
    edges = [0, 1, 3, 6, 10, np.inf]
    res = np.abs(np.asarray(y_va) - np.asarray(pred_va))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pred_va >= lo) & (pred_va < hi)
        hw = float(np.quantile(res[m], 1 - C.CONFORMAL_ALPHA)) if m.sum() >= 30 else float(np.quantile(res, 1 - C.CONFORMAL_ALPHA))
        out.append((lo, hi, hw))
    return out


def apply_interval(pred, table):
    pred = np.asarray(pred, dtype=float)
    hw = np.zeros_like(pred)
    for lo, hi, w in table:
        hw[(pred >= lo) & (pred < hi)] = w
    return np.clip(pred - hw, 0, None), pred + hw


def metrics(y, p, cuts):
    ti, pi = to_tier_idx(y, cuts), to_tier_idx(p, cuts)
    hi_true, hi_pred = ti >= 2, pi >= 2
    conf = np.zeros((4, 4), dtype=int)
    for a, b in zip(ti, pi):
        conf[a, b] += 1
    return dict(MAE=float(np.mean(np.abs(y - p))),
                RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
                tier_accuracy=float(np.mean(ti == pi)),
                within_one_tier=float(np.mean(np.abs(ti - pi) <= 1)),
                high_or_critical_recall=float(hi_pred[hi_true].mean()) if hi_true.any() else None,
                high_or_critical_precision=float(hi_true[hi_pred].mean()) if hi_pred.any() else None,
                confusion_matrix=conf.tolist())


def main():
    t0 = time.time()
    raw = load_raw()
    feat_w = build_features(raw, use_case_history=False)   # deployed: weather + ward + season only
    feat_c = build_features(raw, use_case_history=True)    # ablation: also uses past case counts
    cols = feature_columns(feat_w, False)
    cols_c = feature_columns(feat_c, True)

    cuts = compute_thresholds(raw.loc[raw["date"] <= C.TRAIN_END, "dengue_cases"])
    print("Tier cut points (cases/week):", cuts, "-> Low <", cuts[0], "| Medium <", cuts[1],
          "| High <", cuts[2], "| Critical >=", cuts[2])

    all_metrics, backtests, intervals = {}, [], {}
    for h in C.HORIZONS:
        d = add_horizon(feat_w, h).dropna(subset=cols + ["target"]).reset_index(drop=True)
        dc = add_horizon(feat_c, h).dropna(subset=cols_c + ["target"]).reset_index(drop=True)
        tr, va, te = split_masks(d)
        trc, vac, tec = split_masks(dc)

        # --- model fitted on TRAIN only, early-stopped on VALIDATION ---
        m = fit_model(d.loc[tr, cols], d.loc[tr, "target"], d.loc[va, cols], d.loc[va, "target"])
        best_it = int(m.best_iteration_ or m.n_estimators)
        p_va = m.predict(d.loc[va, cols])
        table = interval_table(p_va, d.loc[va, "target"].values)
        intervals[h] = table

        y_te = d.loc[te, "target"].values
        p_te = m.predict(d.loc[te, cols])
        lo, hi = apply_interval(p_te, table)
        res = metrics(y_te, p_te, cuts)
        res["interval_80_coverage"] = float(np.mean((y_te >= lo) & (y_te <= hi)))
        res["best_iteration"] = best_it

        # --- baselines on the same test rows ---
        ward_mean = d.loc[tr].groupby("ward_id")["target"].mean()
        res["baseline_ward_mean_MAE"] = float(np.mean(np.abs(y_te - d.loc[te, "ward_id"].map(ward_mean).values)))
        res["baseline_persistence_MAE"] = float(np.mean(np.abs(y_te - d.loc[te, "dengue_cases"].values)))

        # --- ablation: adding past case counts as features ---
        mc = fit_model(dc.loc[trc, cols_c], dc.loc[trc, "target"], dc.loc[vac, cols_c], dc.loc[vac, "target"])
        p_c = mc.predict(dc.loc[tec, cols_c])
        res["with_case_history_MAE"] = float(np.mean(np.abs(dc.loc[tec, "target"].values - p_c)))
        res["with_case_history_RMSE"] = float(np.sqrt(np.mean((dc.loc[tec, "target"].values - p_c) ** 2)))
        all_metrics[f"h{h}"] = res

        bt = d.loc[te, ["ward_id", "city", "date", "target_date", "horizon"]].copy()
        bt["actual"], bt["pred"], bt["lo"], bt["hi"] = y_te, p_te, lo, hi
        backtests.append(bt)

        # --- deployed model: refit on TRAIN+VAL with the chosen iteration count ---
        final = lgb.LGBMRegressor(n_estimators=max(50, int(best_it * 1.15)), **PARAMS)
        final.fit(d.loc[tr | va, cols], d.loc[tr | va, "target"])
        joblib.dump(final, C.MODEL_DIR / f"model_h{h}.joblib", compress=3)
        print(f"h={h}w  test MAE={res['MAE']:.3f} RMSE={res['RMSE']:.3f} tier_acc={res['tier_accuracy']:.3f} "
              f"| baselines MAE ward-mean={res['baseline_ward_mean_MAE']:.3f} persistence={res['baseline_persistence_MAE']:.3f} "
              f"| +case-history MAE={res['with_case_history_MAE']:.3f} | 80% cov={res['interval_80_coverage']:.2f}")

    meta = dict(features=cols, tier_cuts=cuts, tier_names=list(C.TIER_NAMES), intervals=intervals,
                horizons=list(C.HORIZONS), train_end=C.TRAIN_END, val_end=C.VAL_END,
                data_start=str(raw["date"].min().date()), data_end=str(raw["date"].max().date()))
    joblib.dump(meta, C.MODEL_DIR / "meta.joblib", compress=3)
    pd.concat(backtests).to_parquet(C.MODEL_DIR / "backtest_test.parquet", index=False)
    all_metrics["_info"] = dict(tier_cuts=cuts, train_period=f"{meta['data_start']} to {C.TRAIN_END}",
                                validation_period=f"{C.TRAIN_END} (exclusive) to {C.VAL_END}",
                                test_period=f"after {C.VAL_END} to {meta['data_end']}",
                                n_features=len(cols), training_seconds=round(time.time() - t0, 1))
    (C.MODEL_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    print(f"Done in {time.time() - t0:.1f}s. Saved models and metrics to {C.MODEL_DIR}")


if __name__ == "__main__":
    main()
