"""Ward-level multi-week lag pipeline.

Every feature row describes one (ward_id, origin_week). All features use data
from the origin week or EARLIER (shift/rolling over past values only), so they
are available at prediction time. The target for horizon h is the number of
dengue cases h weeks AFTER the origin week.
"""
import numpy as np
import pandas as pd

from . import config as C

REQUIRED = ["date", "city", "ward_id", "population_density_sqkm",
            "water_logging_risk_index", "sanitation_score", "avg_temp_c",
            "max_temp_c", "min_temp_c", "rainfall_mm",
            "relative_humidity_pct", "dengue_cases"]


def load_raw(path=C.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    df = df.sort_values(["ward_id", "date"]).reset_index(drop=True)
    if df.duplicated(["ward_id", "date"]).any():
        raise ValueError("Duplicate (ward_id, date) rows found")
    gaps = df.groupby("ward_id")["date"].diff().dropna().dt.days
    if not (gaps == 7).all():
        raise ValueError("Weekly spacing is not regular; lag shifts would be wrong")
    return df


def _grp_roll(df, col, window, how):
    r = df.groupby("ward_id")[col].rolling(window, min_periods=window)
    out = getattr(r, how)()
    return out.reset_index(level=0, drop=True)


def build_features(df: pd.DataFrame, use_case_history: bool = True) -> pd.DataFrame:
    """Return df plus per-ward lag/rolling/seasonal features (no target)."""
    df = df.sort_values(["ward_id", "date"]).reset_index(drop=True).copy()
    g = df.groupby("ward_id")
    new = {}

    for k in C.RAIN_LAGS:
        new[f"rain_lag_{k}w"] = g["rainfall_mm"].shift(k)
    for w in C.ROLL_WINDOWS:
        new[f"rain_sum_{w}w"] = _grp_roll(df, "rainfall_mm", w, "sum")
    new["rain_max_4w"] = _grp_roll(df, "rainfall_mm", 4, "max")

    for k in (0, 2, 4):
        new[f"temp_lag_{k}w"] = g["avg_temp_c"].shift(k)
        new[f"hum_lag_{k}w"] = g["relative_humidity_pct"].shift(k)
    for w in (4, 8):
        new[f"temp_mean_{w}w"] = _grp_roll(df, "avg_temp_c", w, "mean")
        new[f"hum_mean_{w}w"] = _grp_roll(df, "relative_humidity_pct", w, "mean")
    new["min_temp_mean_4w"] = _grp_roll(df, "min_temp_c", 4, "mean")

    if use_case_history:
        for k in C.CASE_LAGS:
            new[f"cases_lag_{k}w"] = g["dengue_cases"].shift(k)
        new["cases_mean_4w"] = _grp_roll(df, "dengue_cases", 4, "mean")

    feat = pd.concat([df, pd.DataFrame(new)], axis=1)

    # ward vulnerability interactions
    feat["rain4w_x_waterlog"] = feat["rain_sum_4w"] * feat["water_logging_risk_index"]
    feat["rain4w_x_density"] = feat["rain_sum_4w"] * feat["population_density_sqkm"] / 1e4
    feat["poor_sanitation_x_waterlog"] = (1 - feat["sanitation_score"]) * feat["water_logging_risk_index"]

    # origin-week seasonality (cyclical)
    woy = feat["date"].dt.isocalendar().week.astype(int).clip(upper=52)
    feat["origin_week_sin"] = np.sin(2 * np.pi * woy / 52)
    feat["origin_week_cos"] = np.cos(2 * np.pi * woy / 52)
    feat["city_code"] = feat["city"].astype("category").cat.codes
    return feat


def add_horizon(feat: pd.DataFrame, h: int) -> pd.DataFrame:
    """Add horizon-specific columns: target-week seasonality and target (cases at origin + h)."""
    out = feat.copy()
    target_date = out["date"] + pd.to_timedelta(7 * h, unit="D")
    woy = target_date.dt.isocalendar().week.astype(int).clip(upper=52)
    out["target_date"] = target_date
    out["target_week_sin"] = np.sin(2 * np.pi * woy / 52)
    out["target_week_cos"] = np.cos(2 * np.pi * woy / 52)
    out["horizon"] = h
    out["target"] = out.groupby("ward_id")["dengue_cases"].shift(-h)
    return out


def feature_columns(feat: pd.DataFrame, use_case_history: bool = True):
    base = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score",
            "city_code", "origin_week_sin", "origin_week_cos",
            "target_week_sin", "target_week_cos",
            "rain4w_x_waterlog", "rain4w_x_density", "poor_sanitation_x_waterlog",
            "rain_max_4w", "min_temp_mean_4w"]
    prefixes = ("rain_lag_", "rain_sum_", "temp_lag_", "hum_lag_", "temp_mean_", "hum_mean_")
    cols = base + [c for c in feat.columns if c.startswith(prefixes)]
    if use_case_history:
        cols += [c for c in feat.columns if c.startswith("cases_")]
    return cols


def driver_group(col: str) -> str:
    """Map a feature to a plain-language driver group for the dashboard."""
    if col.startswith("rain") and "x_" not in col:
        return "Recent rainfall (lagged)"
    if col.startswith("temp") or col.startswith("min_temp"):
        return "Temperature"
    if col.startswith("hum"):
        return "Humidity"
    if col.startswith("cases"):
        return "Recent case trend"
    if col.endswith("week_sin") or col.endswith("week_cos"):
        return "Season (time of year)"
    return "Ward vulnerability"
