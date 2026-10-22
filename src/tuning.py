"""Step 3 bis: tuning of the detectors and comparison at equal false alarms.

Usage: python src/tuning.py   (~15 min on CPU)
Settings are chosen on the validation set; final results are reported on the test set.
Writes results/tuning_*.csv, results/eval_test.csv and results/recall_vs_false_alarms.png.
"""

import matplotlib

matplotlib.use("Agg")  # no window: we only save the image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from detectors import BaselineDetector, TotoDetector, find_anomalies
from evaluate import RESULTS, evaluate_configs, make_eval_set, print_table, scores

SPLIT_SEED = 0  # seed of the validation / test split
K_GRID = [2, 3, 4, 5, 6, 8, 10]  # 3.6: values of k for the baseline
LEVELS = [0.90, 0.95, 0.98, 0.99, 0.995]  # 3.7: Toto interval levels
SAMPLES_GRID = [32, 64, 128, 256]  # 3.8: numbers of Toto samples
F1_TOLERANCE = 0.01  # 3.8: F1 loss accepted to use fewer samples
FA_CLOSE = 0.10  # 3.9: max relative gap to call two false alarm rates close
# 3.9: fine grid, step of 0.5. It goes beyond k = 10 because the baseline still has
# more false alarms than tuned Toto there (3.4 against 2.6 per series on the validation set).
FINE_K_GRID = np.arange(2, 40.5, 0.5)
METRICS = ["Recall", "Precision", "F1", "False alarms / series"]


def save(df, name, columns, formats=None):
    """Saves a table in results/ and prints it in Markdown format."""
    path = RESULTS / name
    df.round(4).to_csv(path, index=False)
    print_table(df, columns, formats)
    print(f"Saved: {path}\n")


# --- 3.5 Validation / test split ---

def split_val_test(n_series, seed=SPLIT_SEED):
    """Splits the indices 0..n_series-1 into two random halves: (validation, test).

    The validation set is only used to choose the settings, the test set only to report
    the final results.
    """
    order = np.random.default_rng(seed).permutation(n_series)
    half = n_series // 2
    return np.sort(order[:half]), np.sort(order[half:])


# --- 3.6 Baseline: sweep over k ---

def sweep_baseline(history, spiked, positions, k_grid=K_GRID):
    """Metrics of the baseline for each value of k."""
    rows = []
    for k in k_grid:
        detector = BaselineDetector(k=k)
        detected = [find_anomalies(f, *detector.predict_interval(h)) for h, f in zip(history, spiked)]
        rows.append({"k": k, **scores(detected, positions)})
    return pd.DataFrame(rows)


# --- 3.7 Toto: interval level ---

def sweep_toto_levels(history, spiked, positions, levels=LEVELS, num_samples=256):
    """Toto metrics for each interval level; returns the table and the level with the best F1.

    Toto runs only once per series: all quantiles are computed on the same samples.
    """
    torch.manual_seed(0)  # reproducible samples
    print(f"Toto, {num_samples} samples: {len(history)} series...", flush=True)
    samples = TotoDetector(num_samples=num_samples).predict_samples_batch(history)
    rows = []
    for level in levels:
        q = (1 - level) / 2  # same formula as TotoDetector
        low, high = np.quantile(samples, [q, 1 - q], axis=-1)
        detected = [find_anomalies(f, lo, hi) for f, lo, hi in zip(spiked, low, high)]
        rows.append({"Level": level, "Low quantile": q, "High quantile": 1 - q, **scores(detected, positions)})
    df = pd.DataFrame(rows)
    return df, float(df.loc[df["F1"].idxmax(), "Level"])


# --- 3.8 Toto: number of samples ---

def sweep_toto_samples(history, spiked, positions, level, samples_grid=SAMPLES_GRID):
    """Metrics and real computation time of Toto at the chosen level, for each number of samples.

    Returns the table and the smallest number of samples whose F1 is within F1_TOLERANCE
    of the best: the best trade-off between quality and cost.
    """
    configs = {
        f"Toto, {s} samples": (lambda s=s: TotoDetector(num_samples=s, level=level)) for s in samples_grid
    }
    df = pd.DataFrame(evaluate_configs(configs, history, spiked, positions))
    df.insert(0, "Samples", samples_grid)
    df = df.drop(columns="Detector")
    good = df[df["F1"] >= df["F1"].max() - F1_TOLERANCE]
    return df, int(good["Samples"].min())


# --- 3.9 Baseline at the same number of false alarms as Toto ---

def match_false_alarms(history, spiked, positions, baseline_df, target):
    """k of the baseline whose false alarms per series are closest to target.

    If no k of the grid is close (relative gap > FA_CLOSE), we start again with the fine
    grid FINE_K_GRID. Returns k and the table of the sweep used.
    """
    gap = (baseline_df["False alarms / series"] - target).abs()
    if gap.min() > FA_CLOSE * target:
        baseline_df = sweep_baseline(history, spiked, positions, FINE_K_GRID)
        gap = (baseline_df["False alarms / series"] - target).abs()
    return float(baseline_df.loc[gap.idxmin(), "k"]), baseline_df


# --- 3.10 Final results on the test set ---

def final_test(history, spiked, positions, k_matched, level, num_samples):
    """The three final configurations, evaluated on the test set."""
    configs = {
        "Baseline (k = 3, default)": lambda: BaselineDetector(k=3),
        f"Baseline (k = {k_matched:g}, same false alarms as Toto)": lambda: BaselineDetector(k=k_matched),
        f"Tuned Toto ({level * 100:g}% level, {num_samples} samples)":
            lambda: TotoDetector(num_samples=num_samples, level=level),
    }
    return pd.DataFrame(evaluate_configs(configs, history, spiked, positions))


# --- 3.11 Recall / false alarms figure ---

def plot_tradeoff(baseline_df, levels_df, k_matched, level, toto_tuned,
                  path=RESULTS / "recall_vs_false_alarms.png"):
    """Recall as a function of false alarms per series, one curve per method (validation set).

    level: level chosen in 3.7; toto_tuned: row of 3.8 for the chosen Toto setting.
    """
    orange, aqua, ink, grey = "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
    fa = "False alarms / series"
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#fcfcfb")
    ax.axvline(toto_tuned[fa], color=grey, lw=1, ls=":", label="same false alarms (3.9)")

    # Baseline: one point per k, labels below the curve (k of the base grid and chosen k)
    ax.plot(baseline_df[fa], baseline_df["Recall"], "-o", color=orange, lw=2, ms=4,
            label="Baseline (one point per k)")
    for x, y, k in zip(baseline_df[fa], baseline_df["Recall"], baseline_df["k"]):
        if np.isclose(k, [*K_GRID, k_matched]).any():
            ax.annotate(f"k = {k:g}", (x, y), textcoords="offset points", xytext=(6, -12), fontsize=8, color=grey)
    chosen = baseline_df[np.isclose(baseline_df["k"], k_matched)].iloc[0]
    ax.scatter(chosen[fa], chosen["Recall"], s=160, color=orange, edgecolor=ink, linewidth=1.5, zorder=3,
               label=f"chosen setting: k = {k_matched:g}")

    # Toto: one point per interval level, labels above the curve
    ax.plot(levels_df[fa], levels_df["Recall"], "-o", color=aqua, lw=2, ms=4,
            label="Toto, 256 samples (one point per level, 3.7)")
    for x, y, n in zip(levels_df[fa], levels_df["Recall"], levels_df["Level"]):
        ax.annotate(f"{n * 100:g}%", (x, y), textcoords="offset points", xytext=(-6, 8), ha="right",
                    fontsize=8, color=grey)
    ax.scatter(toto_tuned[fa], toto_tuned["Recall"], s=160, color=aqua, edgecolor=ink, linewidth=1.5, zorder=3,
               label=f"chosen setting: {level * 100:g}% level, {toto_tuned['Samples']:.0f} samples (run of 3.8)")

    ax.set_xlabel("false alarms per series (out of 61 normal points)", color=grey)
    ax.set_ylabel("recall (share of spikes detected)", color=grey)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(left=0)
    ax.set_title("Validation (100 series): at equal false alarms, the highest curve is the best",
                 color=grey, fontsize=10, loc="left")
    ax.set_facecolor("#fcfcfb")
    ax.grid(color="#e1e0d9", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Figure saved: {path}")


def main():
    # Spikes of step 3 (seed i for series i), then split by indices
    history, spiked, positions = make_eval_set()
    val, test = split_val_test(len(history))
    v = (history[val], spiked[val], positions[val])
    print(f"Validation: {len(val)} series | test: {len(test)} series\n")

    print("3.6 Baseline: sweep over k (validation)")
    baseline = sweep_baseline(*v)
    save(baseline, "tuning_baseline.csv", ["k", *METRICS], {"k": "{:g}"})

    print("3.7 Toto: interval level, 256 samples (validation)")
    levels, level = sweep_toto_levels(*v)
    save(levels, "tuning_toto_quantiles.csv", ["Level", *METRICS], {"Level": "{:.1%}"})
    print(f"Chosen level (best F1): {level:.1%}\n")

    print(f"3.8 Toto: number of samples at the {level:.1%} level (validation)")
    samples, num_samples = sweep_toto_samples(*v, level)
    save(samples, "tuning_toto_samples.csv", ["Samples", *METRICS, "ms / series"], {"Samples": "{:.0f}"})
    print(f"Chosen samples: {num_samples}\n")

    print("3.9 Baseline at equal false alarms (validation)")
    target = samples.loc[samples["Samples"] == num_samples, "False alarms / series"].item()
    k_matched, baseline_grid = match_false_alarms(*v, baseline, target)
    if len(baseline_grid) > len(baseline):  # refined grid
        baseline_grid.round(4).to_csv(RESULTS / "tuning_baseline_fine.csv", index=False)
    matched = baseline_grid[np.isclose(baseline_grid["k"], k_matched)].iloc[0]
    print(f"Tuned Toto: {target:.2f} false alarms / series -> baseline with k = {k_matched:g}: "
          f"{matched['False alarms / series']:.2f} false alarms / series, recall {matched['Recall']:.1%}\n")

    print("3.10 Final results (test)")
    final = final_test(history[test], spiked[test], positions[test], k_matched, level, num_samples)
    save(final, "eval_test.csv", ["Detector", *METRICS, "ms / series"])

    print("3.11 Recall / false alarms figure (validation)")
    toto_tuned = samples[samples["Samples"] == num_samples].iloc[0]
    plot_tradeoff(baseline_grid, levels, k_matched, level, toto_tuned)


if __name__ == "__main__":
    main()
