"""Étape 3 bis : réglage des détecteurs et comparaison à fausses alertes égales.

Usage : python src/tuning.py   (~15 min sur CPU)
Les réglages sont choisis sur la validation ; les résultats finaux sont annoncés sur le test.
Écrit results/tuning_*.csv, results/eval_test.csv et results/recall_vs_false_alarms.png.
"""

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : on enregistre seulement l'image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from detectors import BaselineDetector, TotoDetector, find_anomalies
from evaluate import RESULTS, evaluate_configs, make_eval_set, print_table, scores

SPLIT_SEED = 0  # seed du tirage validation / test
K_GRID = [2, 3, 4, 5, 6, 8, 10]  # 3.6 : valeurs de k de la méthode simple
LEVELS = [0.90, 0.95, 0.98, 0.99, 0.995]  # 3.7 : niveaux d'intervalle de Toto
SAMPLES_GRID = [32, 64, 128, 256]  # 3.8 : nombres de tirages de Toto
# 3.8 : perte de F1 acceptée pour prendre moins de tirages. Fixée au niveau du bruit des tirages :
# le même réglage (99 %, 256 tirages) donne un F1 de 0,626 en 3.7 et 0,609 en 3.8.
F1_TOLERANCE = 0.02
FA_CLOSE = 0.10  # 3.9 : écart relatif max pour dire que deux taux de fausses alertes sont proches
# 3.9 : grille fine, pas de 0,5. Elle va au-delà de k = 10 car la méthode simple y garde
# encore plus de fausses alertes que Toto réglé (3,4 contre 2,6 par série sur la validation).
FINE_K_GRID = np.arange(2, 40.5, 0.5)
METRICS = ["Rappel", "Précision", "F1", "Fausses alertes / série"]


def save(df, name, columns, formats=None):
    """Enregistre un tableau dans results/ et l'affiche au format Markdown."""
    path = RESULTS / name
    df.round(4).to_csv(path, index=False)
    print_table(df, columns, formats)
    print(f"Enregistré : {path}\n")


# --- 3.5 Séparation validation / test ---

def split_val_test(n_series, seed=SPLIT_SEED):
    """Coupe les indices 0..n_series-1 en deux moitiés tirées au hasard : (validation, test).

    La validation sert uniquement à choisir les réglages, le test uniquement à annoncer
    les résultats finaux.
    """
    order = np.random.default_rng(seed).permutation(n_series)
    half = n_series // 2
    return np.sort(order[:half]), np.sort(order[half:])


# --- 3.6 Méthode simple : balayage de k ---

def sweep_baseline(history, spiked, positions, k_grid=K_GRID):
    """Métriques de la méthode simple pour chaque valeur de k."""
    rows = []
    for k in k_grid:
        detector = BaselineDetector(k=k)
        detected = [find_anomalies(f, *detector.predict_interval(h)) for h, f in zip(history, spiked)]
        rows.append({"k": k, **scores(detected, positions)})
    return pd.DataFrame(rows)


# --- 3.7 Toto : niveau de l'intervalle ---

def sweep_toto_levels(history, spiked, positions, levels=LEVELS, num_samples=256):
    """Métriques de Toto pour chaque niveau d'intervalle ; renvoie le tableau et le niveau au meilleur F1.

    Toto ne tourne qu'une fois par série : tous les quantiles sont calculés sur les mêmes tirages.
    """
    torch.manual_seed(0)  # tirages reproductibles
    print(f"Toto, {num_samples} tirages : {len(history)} séries...", flush=True)
    samples = TotoDetector(num_samples=num_samples).predict_samples_batch(history)
    rows = []
    for level in levels:
        q = (1 - level) / 2  # même formule que TotoDetector
        low, high = np.quantile(samples, [q, 1 - q], axis=-1)
        detected = [find_anomalies(f, lo, hi) for f, lo, hi in zip(spiked, low, high)]
        rows.append({"Niveau": level, "Quantile bas": q, "Quantile haut": 1 - q, **scores(detected, positions)})
    df = pd.DataFrame(rows)
    return df, float(df.loc[df["F1"].idxmax(), "Niveau"])


# --- 3.8 Toto : nombre de tirages ---

def sweep_toto_samples(history, spiked, positions, level, samples_grid=SAMPLES_GRID):
    """Métriques et vrai temps de calcul de Toto au niveau retenu, pour chaque nombre de tirages.

    Renvoie le tableau et le plus petit nombre de tirages dont le F1 est à moins de
    F1_TOLERANCE du meilleur : le meilleur compromis entre qualité et coût.
    """
    configs = {
        f"Toto, {s} tirages": (lambda s=s: TotoDetector(num_samples=s, level=level)) for s in samples_grid
    }
    df = pd.DataFrame(evaluate_configs(configs, history, spiked, positions))
    df.insert(0, "Tirages", samples_grid)
    df = df.drop(columns="Détecteur")
    good = df[df["F1"] >= df["F1"].max() - F1_TOLERANCE]
    return df, int(good["Tirages"].min())


# --- 3.9 Méthode simple au même nombre de fausses alertes que Toto ---

def match_false_alarms(history, spiked, positions, baseline_df, target):
    """k de la méthode simple dont les fausses alertes par série sont les plus proches de target.

    Si aucun k de la grille n'est proche (écart relatif > FA_CLOSE), on reprend avec la grille
    fine FINE_K_GRID. Renvoie k et le tableau du balayage utilisé.
    """
    gap = (baseline_df["Fausses alertes / série"] - target).abs()
    if gap.min() > FA_CLOSE * target:
        baseline_df = sweep_baseline(history, spiked, positions, FINE_K_GRID)
        gap = (baseline_df["Fausses alertes / série"] - target).abs()
    return float(baseline_df.loc[gap.idxmin(), "k"]), baseline_df


# --- 3.10 Résultats finaux sur le test ---

def final_test(history, spiked, positions, k_matched, level, num_samples):
    """Les trois configurations finales, évaluées sur le test."""
    configs = {
        "Méthode simple (k = 3, défaut)": lambda: BaselineDetector(k=3),
        f"Méthode simple (k = {k_matched:g}, mêmes fausses alertes que Toto)": lambda: BaselineDetector(k=k_matched),
        f"Toto réglé (niveau {level * 100:g} %, {num_samples} tirages)":
            lambda: TotoDetector(num_samples=num_samples, level=level),
    }
    return pd.DataFrame(evaluate_configs(configs, history, spiked, positions))


# --- 3.11 Figure rappel / fausses alertes ---

def plot_tradeoff(baseline_df, levels_df, k_matched, level, toto_tuned,
                  path=RESULTS / "recall_vs_false_alarms.png"):
    """Rappel en fonction des fausses alertes par série, une courbe par méthode (validation).

    level : niveau retenu en 3.7 ; toto_tuned : ligne de 3.8 pour le réglage retenu de Toto.
    """
    orange, aqua, ink, grey = "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
    fa = "Fausses alertes / série"
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#fcfcfb")
    ax.axvline(toto_tuned[fa], color=grey, lw=1, ls=":", label="mêmes fausses alertes (3.9)")

    # Méthode simple : un point par k, étiquettes sous la courbe (k de la grille de base et k retenu)
    ax.plot(baseline_df[fa], baseline_df["Rappel"], "-o", color=orange, lw=2, ms=4,
            label="Méthode simple (un point par k)")
    for x, y, k in zip(baseline_df[fa], baseline_df["Rappel"], baseline_df["k"]):
        if np.isclose(k, [*K_GRID, k_matched]).any():
            ax.annotate(f"k = {k:g}", (x, y), textcoords="offset points", xytext=(6, -12), fontsize=8, color=grey)
    chosen = baseline_df[np.isclose(baseline_df["k"], k_matched)].iloc[0]
    ax.scatter(chosen[fa], chosen["Rappel"], s=160, color=orange, edgecolor=ink, linewidth=1.5, zorder=3,
               label=f"réglage retenu : k = {k_matched:g}")

    # Toto : un point par niveau d'intervalle, étiquettes au-dessus de la courbe
    ax.plot(levels_df[fa], levels_df["Rappel"], "-o", color=aqua, lw=2, ms=4,
            label="Toto, 256 tirages (un point par niveau, 3.7)")
    for x, y, n in zip(levels_df[fa], levels_df["Rappel"], levels_df["Niveau"]):
        ax.annotate(f"{n * 100:g} %", (x, y), textcoords="offset points", xytext=(-6, 8), ha="right",
                    fontsize=8, color=grey)
    ax.scatter(toto_tuned[fa], toto_tuned["Rappel"], s=160, color=aqua, edgecolor=ink, linewidth=1.5, zorder=3,
               label=f"réglage retenu : niveau {level * 100:g} %, {toto_tuned['Tirages']:.0f} tirages (mesure de 3.8)")

    ax.set_xlabel("fausses alertes par série (sur 61 points normaux)", color=grey)
    ax.set_ylabel("rappel (part des pics détectés)", color=grey)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(left=0)
    ax.set_title("Validation (100 séries) : à fausses alertes égales, la courbe la plus haute est la meilleure",
                 color=grey, fontsize=10, loc="left")
    ax.set_facecolor("#fcfcfb")
    ax.grid(color="#e1e0d9", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Figure enregistrée : {path}")


def main():
    # Pics de l'étape 3 (seed i pour la série i), puis découpage par indices
    history, spiked, positions = make_eval_set()
    val, test = split_val_test(len(history))
    v = (history[val], spiked[val], positions[val])
    print(f"Validation : {len(val)} séries | test : {len(test)} séries\n")

    print("3.6 Méthode simple : balayage de k (validation)")
    baseline = sweep_baseline(*v)
    save(baseline, "tuning_baseline.csv", ["k", *METRICS], {"k": "{:g}"})

    print("3.7 Toto : niveau de l'intervalle, 256 tirages (validation)")
    levels, level = sweep_toto_levels(*v)
    save(levels, "tuning_toto_quantiles.csv", ["Niveau", *METRICS], {"Niveau": "{:.1%}"})
    print(f"Niveau retenu (meilleur F1) : {level:.1%}\n")

    print(f"3.8 Toto : nombre de tirages au niveau {level:.1%} (validation)")
    samples, num_samples = sweep_toto_samples(*v, level)
    save(samples, "tuning_toto_samples.csv", ["Tirages", *METRICS, "ms / série"], {"Tirages": "{:.0f}"})
    print(f"Tirages retenus : {num_samples}\n")

    print("3.9 Méthode simple à fausses alertes égales (validation)")
    target = samples.loc[samples["Tirages"] == num_samples, "Fausses alertes / série"].item()
    k_matched, baseline_grid = match_false_alarms(*v, baseline, target)
    if len(baseline_grid) > len(baseline):  # grille affinée
        baseline_grid.round(4).to_csv(RESULTS / "tuning_baseline_fine.csv", index=False)
    matched = baseline_grid[np.isclose(baseline_grid["k"], k_matched)].iloc[0]
    print(f"Toto réglé : {target:.2f} fausses alertes / série -> méthode simple avec k = {k_matched:g} : "
          f"{matched['Fausses alertes / série']:.2f} fausses alertes / série, rappel {matched['Rappel']:.1%}\n")

    print("3.10 Résultats finaux (test)")
    final = final_test(history[test], spiked[test], positions[test], k_matched, level, num_samples)
    save(final, "eval_test.csv", ["Détecteur", *METRICS, "ms / série"])

    print("3.11 Figure rappel / fausses alertes (validation)")
    toto_tuned = samples[samples["Tirages"] == num_samples].iloc[0]
    plot_tradeoff(baseline_grid, levels, k_matched, level, toto_tuned)


if __name__ == "__main__":
    main()
