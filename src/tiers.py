"""Risk tiers: Low / Medium / High / Critical."""
import numpy as np
from . import config as C


def compute_thresholds(train_cases) -> list:
    """Cut points from TRAIN-period weekly case quantiles (documented in ARCHITECTURE.md)."""
    q = np.quantile(np.asarray(train_cases), C.TIER_QUANTILES)
    cuts, prev = [], -1
    for v in q:  # keep strictly increasing integers
        v = max(int(np.ceil(v)), prev + 1)
        cuts.append(v)
        prev = v
    return cuts


def to_tier(value, cuts) -> str:
    return C.TIER_NAMES[int(np.searchsorted(cuts, value, side="right"))]


def to_tier_idx(values, cuts):
    return np.searchsorted(cuts, np.asarray(values), side="right")
