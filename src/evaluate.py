"""Étape 3 : évaluation des détecteurs sur les 200 séries, hors API.

Usage : python src/evaluate.py   (~10 min sur CPU, surtout Toto avec 256 échantillons)
Écrit results/eval.csv et affiche le tableau.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import inject_spikes, load_series, split_history_future
from detectors import BaselineDetector, TotoDetector, find_anomalies

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Configurations comparées ; Toto n'est chargé qu'au moment de son évaluation
CONFIGS = {
    "Méthode simple": lambda: BaselineDetector(window=200, k=3),
    "Toto (32 échantillons)": lambda: TotoDetector(num_samples=32),
    "Toto (256 échantillons)": lambda: TotoDetector(num_samples=256),
}


# --- 3.1 Métriques ---

def true_positives(detected, positions):
    """Vrais positifs (TP) = pics injectés signalés, cumulé sur toutes les séries.

    detected et positions : une liste par série des positions dans le futur
    (points signalés par le détecteur, pics injectés).
    """
    return sum(len(np.intersect1d(d, p)) for d, p in zip(detected, positions))


def recall(detected, positions):
    """Rappel = pics détectés / pics injectés, cumulé sur toutes les séries."""
    return true_positives(detected, positions) / sum(len(p) for p in positions)


def false_alarms(detected, positions):
    """Fausses alertes (faux positifs, FP) = points signalés qui ne sont pas des pics, cumulé."""
    return sum(len(np.setdiff1d(d, p)) for d, p in zip(detected, positions))


def detect_all(detector, histories, futures):
    """Détecte les anomalies série par série ; renvoie les points signalés et le temps moyen en ms.

    Le temps mesuré couvre la prévision de l'intervalle et la comparaison avec le futur.
    Un premier appel d'échauffement (plus lent : allocations, caches) n'est pas compté.
    """
    detector.predict_interval(histories[0])  # échauffement
    detected, elapsed = [], 0.0
    for history, future in zip(histories, futures):
        start = time.perf_counter()
        low, high = detector.predict_interval(history)
        detected.append(find_anomalies(future, low, high))
        elapsed += time.perf_counter() - start
    return detected, elapsed / len(histories) * 1000


# --- 3.4 Précision, F1, fausses alertes par série ---

def scores(detected, positions):
    """Toutes les métriques, calculées sur les totaux de toutes les séries.

    La précision est calculée sur les totaux (et non en moyenne par série) pour éviter
    les divisions par zéro sur les séries sans alerte.
    """
    tp = true_positives(detected, positions)
    fp = false_alarms(detected, positions)
    rec = recall(detected, positions)
    prec = tp / (tp + fp) if tp + fp else 0.0  # aucune alerte : précision fixée à 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "Rappel": rec,
        "Précision": prec,
        "F1": f1,
        "TP": tp,
        "Fausses alertes": fp,
        "Fausses alertes / série": fp / len(positions),
    }


# Format d'affichage des colonnes dans les tableaux Markdown
FORMATS = {
    "Rappel": "{:.1%}",
    "Précision": "{:.1%}",
    "F1": "{:.3f}",
    "Fausses alertes / série": "{:.2f}",
    "ms / série": "{:.2f}",
}


def print_table(df, columns, formats=None):
    """Affiche les colonnes choisies d'un tableau au format Markdown (formats : formats en plus)."""
    formats = {**FORMATS, **(formats or {})}
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for _, row in df.iterrows():
        print("| " + " | ".join(formats.get(c, "{}").format(row[c]) for c in columns) + " |")


# --- 3.2 Boucle d'évaluation ---

def make_eval_set(n_series=None):
    """Historiques, futurs avec pics et positions des pics des n_series premières séries.

    Les pics de la série i sont injectés avec la seed i : ils sont identiques pour tous
    les détecteurs et pour tout sous-ensemble des séries (validation, test).
    """
    values, _, _ = load_series()
    history, future = split_history_future(values[:n_series])
    spiked, positions = zip(*(inject_spikes(f, h, seed=i) for i, (h, f) in enumerate(zip(history, future))))
    return history, np.stack(spiked), np.stack(positions)


def evaluate_configs(configs, history, spiked, positions):
    """Évalue chaque configuration sur les mêmes séries et les mêmes pics ; une ligne par détecteur."""
    rows = []
    for name, make_detector in configs.items():
        torch.manual_seed(0)  # tirages de Toto reproductibles
        detector = make_detector()
        print(f"{name} : {len(history)} séries...", flush=True)
        detected, ms = detect_all(detector, history, spiked)
        rows.append({"Détecteur": name, **scores(detected, positions), "ms / série": ms})
    return rows


def run_evaluation(n_series=None):
    """Évaluation de l'étape 3 : les trois configurations de CONFIGS sur toutes les séries."""
    return evaluate_configs(CONFIGS, *make_eval_set(n_series))


# --- 3.3 Tableau des résultats ---

def save_and_print(rows, path=RESULTS / "eval.csv"):
    """Enregistre le tableau en CSV et l'affiche au format Markdown."""
    df = pd.DataFrame(rows)
    df.round(4).to_csv(path, index=False)
    print()
    print_table(df, ["Détecteur", "Rappel", "Précision", "F1", "Fausses alertes / série", "ms / série"])
    print(f"\nTableau enregistré : {path}")


if __name__ == "__main__":
    save_and_print(run_evaluation())
