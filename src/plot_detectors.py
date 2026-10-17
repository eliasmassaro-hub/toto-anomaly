"""Étape 2.4 : une série avec les intervalles des deux détecteurs, les pics injectés et les points détectés.

Usage : python src/plot_detectors.py [--index 0] [--seed 0] [--num-samples 256]
L'image est enregistrée dans results/series_<index>_detectors.png.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import CONTEXT, HORIZON, inject_spikes, load_series, split_history_future
from detectors import BaselineDetector, TotoDetector, find_anomalies
from plot_data import AXIS, BLUE, GREY, MUTED, RED, RESULTS, SHADE, SURFACE, ZOOM

ORANGE, AQUA, INK = "#eb6834", "#1baf7a", "#0b0b0b"  # intervalle simple, intervalle Toto, points signalés


def draw(ax, series, positions, low, high, detected, color, title):
    """Trace la fin de l'historique, le futur avec pics, l'intervalle et les points signalés."""
    start, end = CONTEXT - ZOOM, len(series)
    x_future = np.arange(CONTEXT, end)
    ax.axvspan(CONTEXT - 0.5, end - 0.5, color=SHADE)
    ax.fill_between(x_future, low, high, color=color, alpha=0.3, lw=0, label="intervalle prévu")
    ax.plot(np.arange(start, end), series[start:], color=BLUE, lw=1.2, label="série")
    ax.scatter(CONTEXT + positions, series[CONTEXT + positions], s=150, facecolor="none",
               edgecolor=RED, linewidth=1.8, zorder=3, label="pic injecté")
    ax.scatter(CONTEXT + detected, series[CONTEXT + detected], s=22, color=INK, zorder=4,
               label="point signalé")

    hits = len(np.intersect1d(detected, positions))
    false_alarms = len(np.setdiff1d(detected, positions))
    plural = "s" if false_alarms > 1 else ""
    ax.set_title(f"{title} : {hits}/{len(positions)} pics détectés, "
                 f"{false_alarms} fausse{plural} alerte{plural}",
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
    ax.set_ylabel("valeur", color=GREY, fontsize=9)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    return hits, false_alarms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="indice de la série (0 à 199)")
    parser.add_argument("--seed", type=int, default=0, help="seed des positions des pics")
    parser.add_argument("--num-samples", type=int, default=256, help="échantillons de Toto")
    args = parser.parse_args()
    torch.manual_seed(0)  # tirages de Toto reproductibles

    values, ids, freqs = load_series()
    history, future = split_history_future(values[args.index])
    spiked, positions = inject_spikes(future, history, seed=args.seed)
    series = np.concatenate([history, spiked])

    baseline = BaselineDetector()
    toto = TotoDetector(num_samples=args.num_samples)
    intervals = {
        "simple": baseline.predict_interval(history),
        "toto": toto.predict_interval(history),
    }

    fig, (ax_simple, ax_toto) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True,
                                             facecolor=SURFACE)
    low, high = intervals["simple"]
    draw(ax_simple, series, positions, low, high, find_anomalies(spiked, low, high), ORANGE,
         f"Méthode simple (médiane ± {baseline.k} × 1,4826 × MAD des {baseline.window} derniers points)")
    low, high = intervals["toto"]
    draw(ax_toto, series, positions, low, high, find_anomalies(spiked, low, high), AQUA,
         f"Toto ({args.num_samples} échantillons, quantiles 0,005 et 0,995)")
    ax_toto.set_xlabel("pas de temps", color=GREY, fontsize=9)
    fig.suptitle(f"Série {ids[args.index]} (fréquence {freqs[args.index]}) : "
                 f"{ZOOM} derniers points d'historique + {HORIZON} points de futur avec pics",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"series_{args.index}_detectors.png"
    fig.savefig(out, dpi=150)
    print(f"Pics injectés aux positions {positions.tolist()} du futur")
    for name, (low, high) in intervals.items():
        print(f"  {name:>6} : points signalés {find_anomalies(spiked, low, high).tolist()}")
    print(f"Figure enregistrée : {out}")


if __name__ == "__main__":
    main()
