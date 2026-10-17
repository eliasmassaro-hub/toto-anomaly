"""Étape 1.4 : affiche une série (historique + futur) avec les pics injectés marqués.

Usage : python src/plot_data.py [--index 0] [--seed 0]
L'image est enregistrée dans results/series_<index>_spikes.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : on enregistre seulement l'image
import matplotlib.pyplot as plt
import numpy as np

from data import CONTEXT, HORIZON, inject_spikes, load_series, split_history_future

RESULTS = Path(__file__).resolve().parent.parent / "results"
ZOOM = 200  # nombre de points d'historique montrés dans le zoom

# Couleurs : série en bleu, pics en rouge, zone future grisée, axes discrets
BLUE, RED, GREY = "#2a78d6", "#d03b3b", "#52514e"
SHADE, SURFACE, AXIS, MUTED = "#ecebe6", "#fcfcfb", "#c3c2b7", "#898781"


def draw(ax, series, future, positions, start):
    """Trace la série à partir de l'indice start, la zone future et les pics."""
    end = len(series)
    ax.axvspan(CONTEXT - 0.5, end - 0.5, color=SHADE, label=f"futur ({HORIZON} points)")
    ax.plot(np.arange(start, end), series[start:], color=BLUE, lw=1.2, label="série")
    ax.plot(np.arange(CONTEXT, end), future, color=GREY, lw=0.9, ls="--", label="futur sans pics")
    ax.scatter(CONTEXT + positions, series[CONTEXT + positions], s=60, color=RED,
               edgecolor=SURFACE, linewidth=1.5, zorder=3, label="pic injecté")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="indice de la série (0 à 199)")
    parser.add_argument("--seed", type=int, default=0, help="seed des positions des pics")
    args = parser.parse_args()

    values, ids, freqs = load_series()
    history, future = split_history_future(values[args.index])
    spiked, positions = inject_spikes(future, history, seed=args.seed)
    series = np.concatenate([history, spiked])

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(12, 7), facecolor=SURFACE)
    draw(ax_full, series, future, positions, start=0)
    draw(ax_zoom, series, future, positions, start=CONTEXT - ZOOM)
    ax_full.set_title(f"Vue complète : {CONTEXT} points d'historique + {HORIZON} points de futur",
                      color=GREY, fontsize=10, loc="left")
    ax_zoom.set_title(f"Zoom : {ZOOM} derniers points d'historique + futur",
                      color=GREY, fontsize=10, loc="left")
    ax_zoom.set_xlabel("pas de temps", color=GREY, fontsize=9)
    fig.suptitle(f"Série {ids[args.index]} (fréquence {freqs[args.index]}) : "
                 f"{len(positions)} pics de 5 × écart-type de l'historique",
                 color="#0b0b0b", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"series_{args.index}_spikes.png"
    fig.savefig(out, dpi=150)
    print(f"Pics aux positions {positions.tolist()} du futur, hauteur {5 * np.std(history):.3f}")
    print(f"Figure enregistrée : {out}")


if __name__ == "__main__":
    main()
