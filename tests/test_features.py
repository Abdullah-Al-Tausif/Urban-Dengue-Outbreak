"""Correctness + leakage tests for the lag pipeline. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.features import load_raw, build_features, add_horizon, feature_columns
from src.tiers import compute_thresholds, to_tier


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def feat(raw):
    return build_features(raw, use_case_history=False)


def test_lags_match_hand_calculation(raw, feat):
    ward = raw["ward_id"].iloc[0]
    w = raw[raw["ward_id"] == ward].reset_index(drop=True)
    f = feat[feat["ward_id"] == ward].reset_index(drop=True)
    i = 100
    assert f.loc[i, "rain_lag_3w"] == pytest.approx(w.loc[i - 3, "rainfall_mm"])
    assert f.loc[i, "rain_sum_4w"] == pytest.approx(w.loc[i - 3:i, "rainfall_mm"].sum())
    assert f.loc[i, "hum_mean_8w"] == pytest.approx(w.loc[i - 7:i, "relative_humidity_pct"].mean())


def test_lags_do_not_leak_across_wards(raw, feat):
    first_rows = feat.groupby("ward_id").head(8)
    assert first_rows["rain_lag_8w"].isna().sum() >= feat["ward_id"].nunique()
    for _, g in feat.groupby("ward_id"):
        assert g["rain_lag_8w"].iloc[:8].isna().all()


def test_no_future_leakage(raw):
    """Changing FUTURE weather must not change features of an earlier origin week."""
    ward = raw["ward_id"].iloc[0]
    cutoff = raw[raw["ward_id"] == ward]["date"].iloc[200]
    base = build_features(raw, False)
    mod = raw.copy()
    mod.loc[mod["date"] > cutoff, ["rainfall_mm", "avg_temp_c", "relative_humidity_pct", "min_temp_c"]] *= 3
    mod_f = build_features(mod, False)
    cols = feature_columns(base, False)
    cols = [c for c in cols if "target_week" not in c]
    a = base[base["date"] <= cutoff][cols].fillna(-1).values
    b = mod_f[mod_f["date"] <= cutoff][cols].fillna(-1).values
    assert np.allclose(a, b)


def test_target_is_h_weeks_ahead(raw, feat):
    ward = raw["ward_id"].iloc[0]
    for h in C.HORIZONS:
        d = add_horizon(feat, h)
        g = d[d["ward_id"] == ward].reset_index(drop=True)
        w = raw[raw["ward_id"] == ward].reset_index(drop=True)
        assert g.loc[50, "target"] == w.loc[50 + h, "dengue_cases"]
        assert g.loc[50, "target_date"] == w.loc[50, "date"] + pd.Timedelta(weeks=h)


def test_chronological_split_has_no_overlap(raw):
    d = raw["date"]
    train, val, test = d <= C.TRAIN_END, (d > C.TRAIN_END) & (d <= C.VAL_END), d > C.VAL_END
    assert train.sum() and val.sum() and test.sum()
    assert d[train].max() < d[val].min() and d[val].max() < d[test].min()


def test_tiers_monotonic():
    cuts = compute_thresholds(np.arange(0, 100))
    assert cuts == sorted(cuts) and len(set(cuts)) == 3
    assert [to_tier(v, cuts) for v in (0, cuts[0], cuts[1], cuts[2] + 5)] == ["Low", "Medium", "High", "Critical"]
