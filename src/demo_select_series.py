"""Step D3.1: finds demo series where Toto clearly beats the classic method.

Usage: python src/demo_select_series.py               (~20 min on CPU)
       python src/demo_select_series.py --plot-only   (redraws the figure from the CSV, ~1 min)

For each of the 200 series, with the spikes of the demo page (inject_spikes(..., seed=0)), runs the
classic method (BaselineDetector(k=3)) and Toto (256 samples, 99% level, torch.manual_seed(0) before
each series), then keeps the candidates:
- Toto catches 3/3 spikes with at most 1 false alarm;
- the classic method does clearly worse: at least 5 more false alarms, or at least 1 missed spike;
- the chart is readable: the series is not almost flat, and Toto's interval has no outlier peaks.
Toto samples are random and the API does not fix the seed: Toto is run again on each candidate with
10 other seeds, to count how often the result holds in a live demo.

Writes results/demo_candidates.csv (all series) and results/demo_candidates.png (best candidates).
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # no window: we only save the image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data import inject_spikes, load_series, split_history_future
from detectors import BaselineDetector, TotoDetector, find_anomalies
from plot_data import RESULTS, SURFACE, ZOOM
from plot_detectors import draw

CSV_PATH = RESULTS / "demo_candidates.csv"
PNG_PATH = RESULTS / "demo_candidates.png"
CLASSIC_COLOR, TOTO_COLOR = "#B34A12", "#2F5FD0"  # same colors as the demo page

MAX_TOTO_FALSE_ALARMS = 1
MIN_EXTRA_FALSE_ALARMS = 5  # the classic method must make at least this many more false alarms...
# ...or miss at least one spike
MIN_FLATNESS = 0.1  # interquartile range / standard deviation of the history: close to 0 for a flat
# series with a few isolated jumps (Gaussian noise gives 1.35)
MAX_OVERSHOOT = 2.0  # how far Toto's interval goes beyond the plotted series, in plotted ranges
CHECK_SEEDS = range(1, 11)  # extra Toto runs on the candidates
N_PLOTTED = 12  # candidates shown in the figure


def demo_case(values, index):
    """History, future with the demo spikes and spike positions of one series."""
    history, future = split_history_future(values[index])
    spiked, positions = inject_spikes(future, history, seed=0)
    return history, spiked, positions


def caught_and_false_alarms(anomalies, positions):
    return len(np.intersect1d(anomalies, positions)), len(np.setdiff1d(anomalies, positions))


def toto_interval(toto, history, seed):
    torch.manual_seed(seed)
    return toto.predict_interval(history)


def screen(values, ids, freqs, baseline, toto):
    """Both detectors on every series, with the readability measures of the chart."""
    rows = []
    for index in range(len(values)):
        history, spiked, positions = demo_case(values, index)
        classic = caught_and_false_alarms(find_anomalies(spiked, *baseline.predict_interval(history)), positions)
        low, high = toto_interval(toto, history, seed=0)
        toto_result = caught_and_false_alarms(find_anomalies(spiked, low, high), positions)

        shown = np.concatenate([history[-ZOOM:], spiked])  # what draw() plots
        overshoot = max(0, high.max() - shown.max()) + max(0, shown.min() - low.min())
        q75, q25 = np.percentile(history, [75, 25])
        rows.append({
            "Index": index,
            "Series": ids[index],
            "Frequency": freqs[index],
            "Classic caught": classic[0],
            "Classic false alarms": classic[1],
            "Toto caught": toto_result[0],
            "Toto false alarms": toto_result[1],
            "Flatness": (q75 - q25) / np.std(history),
            "Interval overshoot": overshoot / np.ptp(shown),
        })
        if (index + 1) % 20 == 0:
            print(f"  {index + 1}/{len(values)} series", flush=True)
    return pd.DataFrame(rows)


def is_candidate(df):
    toto_ok = (df["Toto caught"] == 3) & (df["Toto false alarms"] <= MAX_TOTO_FALSE_ALARMS)
    classic_worse = ((df["Classic false alarms"] >= df["Toto false alarms"] + MIN_EXTRA_FALSE_ALARMS)
                     | (df["Classic caught"] < 3))
    readable = (df["Flatness"] >= MIN_FLATNESS) & (df["Interval overshoot"] <= MAX_OVERSHOOT)
    return toto_ok & classic_worse & readable


def count_stable_runs(values, indices, toto):
    """For each series, number of extra seeds where Toto still catches 3/3 with few false alarms."""
    runs = {}
    for index in indices:
        history, spiked, positions = demo_case(values, index)
        ok = 0
        for seed in CHECK_SEEDS:
            caught, false_alarms = caught_and_false_alarms(
                find_anomalies(spiked, *toto_interval(toto, history, seed)), positions)
            ok += caught == 3 and false_alarms <= MAX_TOTO_FALSE_ALARMS
        runs[index] = ok
        print(f"  series {index}: {ok}/{len(CHECK_SEEDS)} runs OK", flush=True)
    return runs


def ranked_candidates(df):
    """Candidates, the most stable first, then those where the classic method does worst."""
    candidates = df[df["Candidate"]].copy()
    candidates["Gap"] = (candidates["Classic false alarms"] - candidates["Toto false alarms"]
                         + 10 * (3 - candidates["Classic caught"]))
    return candidates.sort_values(["Toto OK runs", "Gap"], ascending=False)


def plot_candidates(df, values, baseline, toto):
    """One row per candidate: classic method and Toto side by side, drawn with draw()."""
    best = ranked_candidates(df).head(N_PLOTTED)
    fig, axes = plt.subplots(len(best), 2, figsize=(15, 2.6 * len(best)), facecolor=SURFACE, squeeze=False)
    for row, (_, case) in zip(axes, best.iterrows()):
        index = int(case["Index"])
        history, spiked, positions = demo_case(values, index)
        series = np.concatenate([history, spiked])
        name = f"#{index} {case['Series']} ({case['Frequency']}, Toto OK in {case['Toto OK runs']:.0f}/10 runs)"
        for ax, (label, color, (low, high)) in zip(row, [
            ("classic", CLASSIC_COLOR, baseline.predict_interval(history)),
            ("Toto", TOTO_COLOR, toto_interval(toto, history, seed=0)),
        ]):
            draw(ax, series, positions, low, high, find_anomalies(spiked, low, high), color, f"{name} · {label}")
            ax.title.set_fontsize(9)
            ax.get_legend().remove()
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=90)
    print(f"Figure saved: {PNG_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="redraw the figure from the saved CSV")
    args = parser.parse_args()

    values, ids, freqs = load_series()
    baseline = BaselineDetector(k=3)
    toto = TotoDetector(num_samples=256, level=0.99)

    if args.plot_only:
        df = pd.read_csv(CSV_PATH)
    else:
        print("Both detectors on the 200 series...")
        df = screen(values, ids, freqs, baseline, toto)
        df["Candidate"] = is_candidate(df)
        print(f"{df['Candidate'].sum()} candidates; Toto again with {len(CHECK_SEEDS)} other seeds...")
        stable = count_stable_runs(values, df.loc[df["Candidate"], "Index"], toto)
        df["Toto OK runs"] = df["Index"].map(stable)  # empty for the other series
        df.round(3).to_csv(CSV_PATH, index=False)
        print(f"Table saved: {CSV_PATH}")

    print(ranked_candidates(df).drop(columns="Gap").to_string(index=False))
    plot_candidates(df, values, baseline, toto)


if __name__ == "__main__":
    main()
