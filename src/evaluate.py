"""Étape 3 : évaluation des détecteurs sur les 200 séries, hors API.

Usage : python src/evaluate.py
"""

import time

import numpy as np

from detectors import find_anomalies


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
