"""Étape 3 : évaluation des détecteurs sur les 200 séries, hors API.

Usage : python src/evaluate.py
"""

import time

import numpy as np
import torch

from data import inject_spikes, load_series, split_history_future
from detectors import BaselineDetector, TotoDetector, find_anomalies

# Configurations comparées ; Toto n'est chargé qu'au moment de son évaluation
CONFIGS = {
    "Méthode simple": lambda: BaselineDetector(window=200, k=3),
    "Toto (32 échantillons)": lambda: TotoDetector(num_samples=32),
    "Toto (256 échantillons)": lambda: TotoDetector(num_samples=256),
}


# --- 3.1 Métriques ---

def recall(detected, positions):
    """Rappel = pics détectés / pics injectés, cumulé sur toutes les séries.

    detected et positions : une liste par série des positions dans le futur
    (points signalés par le détecteur, pics injectés).
    """
    hits = sum(len(np.intersect1d(d, p)) for d, p in zip(detected, positions))
    return hits / sum(len(p) for p in positions)


def false_alarms(detected, positions):
    """Fausses alertes = points signalés qui ne sont pas des pics, cumulé sur toutes les séries."""
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


# --- 3.2 Boucle d'évaluation ---

def run_evaluation(n_series=None):
    """Évalue chaque configuration sur les mêmes séries et les mêmes pics ; une ligne par détecteur."""
    values, _, _ = load_series()
    history, future = split_history_future(values[:n_series])
    # Pics injectés une seule fois (seed i pour la série i) : identiques pour tous les détecteurs
    spiked, positions = zip(*(inject_spikes(f, h, seed=i) for i, (h, f) in enumerate(zip(history, future))))

    rows = []
    for name, make_detector in CONFIGS.items():
        torch.manual_seed(0)  # tirages de Toto reproductibles
        detector = make_detector()
        print(f"{name} : {len(history)} séries...", flush=True)
        detected, ms = detect_all(detector, history, spiked)
        rows.append({
            "Détecteur": name,
            "Rappel": recall(detected, positions),
            "Fausses alertes": false_alarms(detected, positions),
            "ms / série": ms,
        })
    return rows
