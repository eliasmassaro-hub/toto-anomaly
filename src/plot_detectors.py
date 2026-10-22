"""Step 2.4: a series with the intervals of both detectors, the injected spikes and the detected points.

Usage: python src/plot_detectors.py [--index 0] [--seed 0] [--num-samples 256]
The image is saved to results/series_<index>_detectors.png.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import CONTEXT, HORIZON, inject_spikes, load_series, split_history_future
from detectors import BaselineDetector, TotoDetector, find_anomalies
from plot_data import AXIS, BLUE, GREY, MUTED, RED, RESULTS, SHADE, SURFACE, ZOOM

ORANGE, AQUA, INK = "#eb6834", "#1baf7a", "#0b0b0b"  # baseline interval, Toto interval, flagged points


def draw(ax, series, positions, low, high, detected, color, title):
    """Plots the end of the history, the future with spikes, the interval and the flagged points."""
    start, end = CONTEXT - ZOOM, len(series)
    x_future = np.arange(CONTEXT, end)
    ax.axvspan(CONTEXT - 0.5, end - 0.5, color=SHADE)
    ax.fill_between(x_future, low, high, color=color, alpha=0.3, lw=0, label="forecast interval")
    ax.plot(np.arange(start, end), series[start:], color=BLUE, lw=1.2, label="series")
    ax.scatter(CONTEXT + positions, series[CONTEXT + positions], s=150, facecolor="none",
               edgecolor=RED, linewidth=1.8, zorder=3, label="injected spike")
    ax.scatter(CONTEXT + detected, series[CONTEXT + detected], s=22, color=INK, zorder=4,
               label="flagged point")

    hits = len(np.intersect1d(detected, positions))
    false_alarms = len(np.setdiff1d(detected, positions))
    plural = "" if false_alarms == 1 else "s"
    ax.set_title(f"{title}: {hits}/{len(positions)} spikes detected, "
                 f"{false_alarms} false alarm{plural}",
                 color=GREY, fontsize=10, loc="left")
    ax.set_xlim(start, end - 1)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_ylabel("value", color=GREY, fontsize=9)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    return hits, false_alarms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="series index (0 to 199)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the spike positions")
    parser.add_argument("--num-samples", type=int, default=256, help="Toto samples")
    args = parser.parse_args()
    torch.manual_seed(0)  # reproducible Toto samples

    values, ids, freqs = load_series()
    history, future = split_history_future(values[args.index])
    spiked, positions = inject_spikes(future, history, seed=args.seed)
    series = np.concatenate([history, spiked])

    baseline = BaselineDetector()
    toto = TotoDetector(num_samples=args.num_samples)
    intervals = {
        "baseline": baseline.predict_interval(history),
        "toto": toto.predict_interval(history),
    }

    fig, (ax_baseline, ax_toto) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True,
                                               facecolor=SURFACE)
    low, high = intervals["baseline"]
    draw(ax_baseline, series, positions, low, high, find_anomalies(spiked, low, high), ORANGE,
         f"Baseline (median ± {baseline.k} × 1.4826 × MAD of the last {baseline.window} points)")
    low, high = intervals["toto"]
    draw(ax_toto, series, positions, low, high, find_anomalies(spiked, low, high), AQUA,
         f"Toto ({args.num_samples} samples, 0.005 and 0.995 quantiles)")
    ax_toto.set_xlabel("time step", color=GREY, fontsize=9)
    fig.suptitle(f"Series {ids[args.index]} (frequency {freqs[args.index]}): "
                 f"last {ZOOM} history points + {HORIZON} future points with spikes",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"series_{args.index}_detectors.png"
    fig.savefig(out, dpi=150)
    print(f"Spikes injected at future positions {positions.tolist()}")
    for name, (low, high) in intervals.items():
        print(f"  {name:>8}: flagged points {find_anomalies(spiked, low, high).tolist()}")
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
