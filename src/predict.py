"""Inference: load serialized models once, build features once, predict + explain."""
import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import load_raw, build_features, add_horizon, driver_group
from .tiers import to_tier, to_tier_idx
from .train import apply_interval


def _label(col: str, row) -> str:
    """Plain-language label (with the actual value) for a feature."""
    v = row.get(col, np.nan)
    f = lambda x, d=1: "n/a" if pd.isna(x) else f"{x:.{d}f}"
    if col.startswith("rain_lag_"):
        k = col.split("_")[2].rstrip("w")
        return f"Rain {k} wk before forecast week ({f(v)} mm)"
    if col.startswith("rain_sum_"):
        k = col.split("_")[2].rstrip("w")
        return f"Total rain, last {k} wks ({f(v)} mm)"
    if col == "rain_max_4w":
        return f"Heaviest weekly rain, last 4 wks ({f(v)} mm)"
    names = {"water_logging_risk_index": "Drainage / water-logging risk", "sanitation_score": "Sanitation score",
             "population_density_sqkm": "Population density", "rain4w_x_waterlog": "Rain x water-logging risk",
             "rain4w_x_density": "Rain x crowding", "poor_sanitation_x_waterlog": "Poor sanitation x water-logging",
             "min_temp_mean_4w": "Avg night temperature, 4 wks", "city_code": "City"}
    if col in names:
        return f"{names[col]} ({f(v, 2)})" if col not in ("population_density_sqkm", "city_code") else names[col]
    if col.startswith("temp_lag_"):
        return f"Temperature {col.split('_')[2].rstrip('w')} wk ago ({f(v)} C)"
    if col.startswith("temp_mean_"):
        return f"Avg temperature, last {col.split('_')[2].rstrip('w')} wks ({f(v)} C)"
    if col.startswith("hum_lag_"):
        return f"Humidity {col.split('_')[2].rstrip('w')} wk ago ({f(v)} %)"
    if col.startswith("hum_mean_"):
        return f"Avg humidity, last {col.split('_')[2].rstrip('w')} wks ({f(v)} %)"
    if "week_" in col:
        return "Time of year"
    return col


class Predictor:
    def __init__(self):
        self.meta = joblib.load(C.MODEL_DIR / "meta.joblib")
        self.models = {h: joblib.load(C.MODEL_DIR / f"model_h{h}.joblib") for h in self.meta["horizons"]}
        self.cols = self.meta["features"]
        self.cuts = self.meta["tier_cuts"]
        raw = load_raw()
        self.raw = raw
        self.feat = build_features(raw, use_case_history=False)   # computed once
        self.wards = sorted(raw["ward_id"].unique())
        self.city_of = raw.drop_duplicates("ward_id").set_index("ward_id")["city"].to_dict()
        self.dates = sorted(raw["date"].unique())
        self._by_h = {h: add_horizon(self.feat, h) for h in self.meta["horizons"]}

    # ---------- batch (all wards at one origin week) ----------
    def predict_all_wards(self, origin, h: int) -> pd.DataFrame:
        d = self._by_h[h]
        rows = d[d["date"] == pd.Timestamp(origin)].dropna(subset=self.cols)
        pred = self.models[h].predict(rows[self.cols])
        lo, hi = apply_interval(pred, self.meta["intervals"][h])
        out = rows[["ward_id", "city", "date", "target_date"]].copy()
        out["pred"], out["lo"], out["hi"] = pred, lo, hi
        out["tier"] = [to_tier(p, self.cuts) for p in pred]
        return out.reset_index(drop=True)

    # ---------- single ward with drivers ----------
    def predict_ward(self, ward_id: str, origin, h: int, top_k: int = 6) -> dict:
        d = self._by_h[h]
        row = d[(d["ward_id"] == ward_id) & (d["date"] == pd.Timestamp(origin))]
        if row.empty or row[self.cols].isna().any(axis=None):
            raise ValueError("Not enough history before this date (need 8 prior weeks).")
        X = row[self.cols]
        model = self.models[h]
        pred = float(model.predict(X)[0])
        lo, hi = apply_interval(np.array([pred]), self.meta["intervals"][h])
        contrib = model.predict(X, pred_contrib=True)[0]       # log-scale contributions + bias
        contrib = pd.Series(contrib[:-1], index=self.cols)
        grp = contrib.groupby([driver_group(c) for c in self.cols]).sum()
        r = row.iloc[0].to_dict()
        labelled = contrib.groupby([_label(c, r) for c in self.cols]).sum()   # merge duplicate labels
        top = labelled.reindex(labelled.abs().sort_values(ascending=False).index)[:top_k]
        return dict(ward_id=ward_id, city=self.city_of[ward_id], origin=pd.Timestamp(origin),
                    target_date=r["target_date"], horizon=h, pred=pred, lo=float(lo[0]), hi=float(hi[0]),
                    tier=to_tier(pred, self.cuts),
                    groups=(np.exp(grp) - 1).mul(100).sort_values(),   # approx % change in expected cases
                    top=[(lab, float((np.exp(v) - 1) * 100)) for lab, v in top.items()])

    def history(self, ward_id: str, weeks: int = 26, until=None) -> pd.DataFrame:
        d = self.raw[self.raw["ward_id"] == ward_id]
        if until is not None:
            d = d[d["date"] <= pd.Timestamp(until)]
        return d.tail(weeks)
