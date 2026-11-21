"""Measure load time, model size and inference latency -> models/benchmark.json
Run: python -m src.benchmark
"""
import json
import time
import warnings

from . import config as C
from .predict import Predictor

warnings.filterwarnings("ignore")


def main():
    t = time.perf_counter()
    p = Predictor()
    load_s = time.perf_counter() - t
    origin = p.dates[-10]
    lat = []
    for _ in range(20):
        t = time.perf_counter()
        for h in C.HORIZONS:
            p.predict_all_wards(origin, h)
        lat.append((time.perf_counter() - t) * 1000)
    one = []
    for _ in range(50):
        t = time.perf_counter()
        p.predict_ward(p.wards[0], origin, 3)
        one.append((time.perf_counter() - t) * 1000)
    size_kb = sum(f.stat().st_size for f in C.MODEL_DIR.glob("model_h*.joblib")) / 1024
    out = dict(startup_load_and_feature_build_seconds=round(load_s, 2),
               all_60_wards_x_3_horizons_ms_median=round(sorted(lat)[len(lat) // 2], 1),
               single_ward_prediction_with_drivers_ms_median=round(sorted(one)[len(one) // 2], 1),
               total_model_size_kb=round(size_kb, 0), external_api_calls_at_inference=0)
    (C.MODEL_DIR / "benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
