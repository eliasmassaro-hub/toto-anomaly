"""Étape 3 bis : réglage des détecteurs et comparaison à fausses alertes égales.

Usage : python src/tuning.py
Les réglages sont choisis sur la validation ; les résultats finaux sont annoncés sur le test.
"""

import numpy as np
import pandas as pd

from detectors import BaselineDetector, find_anomalies
from evaluate import RESULTS, make_eval_set, print_table, scores

SPLIT_SEED = 0  # seed du tirage validation / test
K_GRID = [2, 3, 4, 5, 6, 8, 10]  # 3.6 : valeurs de k de la méthode simple
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


def main():
    # Pics de l'étape 3 (seed i pour la série i), puis découpage par indices
    history, spiked, positions = make_eval_set()
    val, test = split_val_test(len(history))
    v = (history[val], spiked[val], positions[val])
    print(f"Validation : {len(val)} séries | test : {len(test)} séries\n")

    print("3.6 Méthode simple : balayage de k (validation)")
    baseline = sweep_baseline(*v)
    save(baseline, "tuning_baseline.csv", ["k", *METRICS], {"k": "{:g}"})


if __name__ == "__main__":
    main()
