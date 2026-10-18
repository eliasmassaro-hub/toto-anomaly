"""Étape 3 bis : réglage des détecteurs et comparaison à fausses alertes égales.

Usage : python src/tuning.py
Les réglages sont choisis sur la validation ; les résultats finaux sont annoncés sur le test.
"""

import numpy as np
import pandas as pd
import torch

from detectors import BaselineDetector, TotoDetector, find_anomalies
from evaluate import RESULTS, evaluate_configs, make_eval_set, print_table, scores

SPLIT_SEED = 0  # seed du tirage validation / test
K_GRID = [2, 3, 4, 5, 6, 8, 10]  # 3.6 : valeurs de k de la méthode simple
LEVELS = [0.90, 0.95, 0.98, 0.99, 0.995]  # 3.7 : niveaux d'intervalle de Toto
SAMPLES_GRID = [32, 64, 128, 256]  # 3.8 : nombres de tirages de Toto
F1_TOLERANCE = 0.01  # 3.8 : perte de F1 acceptée pour prendre moins de tirages
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


if __name__ == "__main__":
    main()
