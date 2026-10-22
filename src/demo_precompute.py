"""Step D2.2: precomputes the demo results, used by the demo page when the API is not reachable.

Usage: python src/demo_precompute.py
Writes results/demo_cache.json: for each demo series, the injected spike positions and, for each
method, the forecast interval (low, high) and the flagged points. The history is not stored:
the page reads it from data/series.npz.
"""

import json

import numpy as np
import torch

from data import inject_spikes, load_series, split_history_future
from detectors import BaselineDetector, TotoDetector, find_anomalies
from plot_data import RESULTS

DEMO_INDICES = [42, 0, 3, 7]  # same series as the demo page
CACHE_PATH = RESULTS / "demo_cache.json"


def make_detector(method):
    """A fresh detector, built right after seeding: same random state as plot_detectors.py,
    so the cached results match the figures results/series_<index>_detectors.png."""
    torch.manual_seed(0)
    if method == "baseline":
        return BaselineDetector(k=3)
    return TotoDetector(num_samples=256, level=0.99)  # same settings as the API


def main():
    values, ids, _ = load_series()

    cache = {}
    for index in DEMO_INDICES:
        history, future = split_history_future(values[index])
        spiked, positions = inject_spikes(future, history, seed=0)  # same spikes as the figures
        entry = {"id": str(ids[index]), "positions": positions.tolist()}
        for method in ("baseline", "toto"):
            low, high = make_detector(method).predict_interval(history)
            entry[method] = {
                "low": low.tolist(),
                "high": high.tolist(),
                "anomalies": find_anomalies(spiked, low, high).tolist(),
            }
            false_alarms = len(np.setdiff1d(entry[method]["anomalies"], positions))
            hits = len(np.intersect1d(entry[method]["anomalies"], positions))
            print(f"Series {index} ({ids[index]}), {method:>8}: "
                  f"{hits}/{len(positions)} spikes found, {false_alarms} false alarms")
        cache[str(index)] = entry

    CACHE_PATH.write_text(json.dumps(cache))
    print(f"Cache saved: {CACHE_PATH}")


if __name__ == "__main__":
    main()
