"""Step 3: evaluation of the detectors on the 200 series, outside the API.

Usage: python src/evaluate.py   (~10 min on CPU, mostly Toto with 256 samples)
Writes results/eval.csv and prints the table.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import inject_spikes, load_series, split_history_future
from detectors import BaselineDetector, TotoDetector, find_anomalies

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Compared configurations; Toto is only loaded when it is evaluated
CONFIGS = {
    "Baseline": lambda: BaselineDetector(window=200, k=3),
    "Toto (32 samples)": lambda: TotoDetector(num_samples=32),
    "Toto (256 samples)": lambda: TotoDetector(num_samples=256),
}


# --- 3.1 Metrics ---

def true_positives(detected, positions):
    """True positives (TP) = injected spikes that are flagged, summed over all series.

    detected and positions: one list per series of positions in the future
    (points flagged by the detector, injected spikes).
    """
    return sum(len(np.intersect1d(d, p)) for d, p in zip(detected, positions))


def recall(detected, positions):
    """Recall = detected spikes / injected spikes, summed over all series."""
    return true_positives(detected, positions) / sum(len(p) for p in positions)


def false_alarms(detected, positions):
    """False alarms (false positives, FP) = flagged points that are not spikes, summed."""
    return sum(len(np.setdiff1d(d, p)) for d, p in zip(detected, positions))


def detect_all(detector, histories, futures):
    """Detects anomalies series by series; returns the flagged points and the mean time in ms.

    The measured time covers the interval forecast and the comparison with the future.
    A first warm-up call (slower: allocations, caches) is not counted.
    """
    detector.predict_interval(histories[0])  # warm-up
    detected, elapsed = [], 0.0
    for history, future in zip(histories, futures):
        start = time.perf_counter()
        low, high = detector.predict_interval(history)
        detected.append(find_anomalies(future, low, high))
        elapsed += time.perf_counter() - start
    return detected, elapsed / len(histories) * 1000


# --- 3.4 Precision, F1, false alarms per series ---

def scores(detected, positions):
    """All metrics, computed on the totals over all series.

    Precision is computed on the totals (not averaged per series) to avoid
    divisions by zero on series without any alarm.
    """
    tp = true_positives(detected, positions)
    fp = false_alarms(detected, positions)
    rec = recall(detected, positions)
    prec = tp / (tp + fp) if tp + fp else 0.0  # no alarm at all: precision set to 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "Recall": rec,
        "Precision": prec,
        "F1": f1,
        "TP": tp,
        "False alarms": fp,
        "False alarms / series": fp / len(positions),
    }


# Display format of the columns in the Markdown tables
FORMATS = {
    "Recall": "{:.1%}",
    "Precision": "{:.1%}",
    "F1": "{:.3f}",
    "False alarms / series": "{:.2f}",
    "ms / series": "{:.2f}",
}


def print_table(df, columns, formats=None):
    """Prints the chosen columns of a table in Markdown format (formats: extra formats)."""
    formats = {**FORMATS, **(formats or {})}
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for _, row in df.iterrows():
        print("| " + " | ".join(formats.get(c, "{}").format(row[c]) for c in columns) + " |")


# --- 3.2 Evaluation loop ---

def make_eval_set(n_series=None):
    """Histories, futures with spikes and spike positions of the first n_series series.

    The spikes of series i are injected with seed i: they are the same for all
    detectors and for any subset of the series (validation, test).
    """
    values, _, _ = load_series()
    history, future = split_history_future(values[:n_series])
    spiked, positions = zip(*(inject_spikes(f, h, seed=i) for i, (h, f) in enumerate(zip(history, future))))
    return history, np.stack(spiked), np.stack(positions)


def evaluate_configs(configs, history, spiked, positions):
    """Evaluates each configuration on the same series and the same spikes; one row per detector."""
    rows = []
    for name, make_detector in configs.items():
        torch.manual_seed(0)  # reproducible Toto samples
        detector = make_detector()
        print(f"{name}: {len(history)} series...", flush=True)
        detected, ms = detect_all(detector, history, spiked)
        rows.append({"Detector": name, **scores(detected, positions), "ms / series": ms})
    return rows


def run_evaluation(n_series=None):
    """Step 3 evaluation: the three configurations of CONFIGS on all series."""
    return evaluate_configs(CONFIGS, *make_eval_set(n_series))


# --- 3.3 Results table ---

def save_and_print(rows, path=RESULTS / "eval.csv"):
    """Saves the table as CSV and prints it in Markdown format."""
    df = pd.DataFrame(rows)
    df.round(4).to_csv(path, index=False)
    print()
    print_table(df, ["Detector", "Recall", "Precision", "F1", "False alarms / series", "ms / series"])
    print(f"\nTable saved: {path}")


if __name__ == "__main__":
    save_and_print(run_evaluation())
