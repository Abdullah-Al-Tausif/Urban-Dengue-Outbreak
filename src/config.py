"""Central configuration: paths, horizons, split dates, tier quantiles."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dengue_dataset.csv"
MODEL_DIR = ROOT / "models"

HORIZONS = (2, 3, 4)          # weeks ahead (the 2-4 week early-warning window)
RAIN_LAGS = tuple(range(0, 9))  # lag 0 = origin week, lag k = k weeks earlier
ROLL_WINDOWS = (2, 3, 4, 6, 8)
CASE_LAGS = (0, 1, 2, 3)

# Chronological split on the FORECAST ORIGIN date (no shuffling)
TRAIN_END = "2022-12-31"       # train: 2016-2022
VAL_END = "2023-12-31"         # validation: 2023
# test: 2024-2025 (never used for fitting or tuning)

TIER_NAMES = ("Low", "Medium", "High", "Critical")
TIER_QUANTILES = (0.75, 0.90, 0.96)  # of weekly cases in the TRAIN period
CONFORMAL_ALPHA = 0.20               # 80% prediction interval
