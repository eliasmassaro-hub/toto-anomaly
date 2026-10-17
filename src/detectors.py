"""Étape 2 : détecteurs d'anomalies (méthode simple et Toto).

Interface commune : predict_interval(history) -> (low, high), deux tableaux de HORIZON points.
Un point du futur hors de [low, high] est une anomalie (voir find_anomalies).
"""

import numpy as np

from data import HORIZON

MAD_TO_STD = 1.4826  # pour une loi normale, 1,4826 × MAD estime l'écart-type


class BaselineDetector:
    """Méthode simple : intervalle constant = médiane ± k × 1,4826 × MAD des window derniers points."""

    def __init__(self, window=200, k=3):
        self.window = window
        self.k = k

    def predict_interval(self, history):
        recent = np.asarray(history, dtype=np.float64)[-self.window:]
        median = np.median(recent)
        scale = MAD_TO_STD * np.median(np.abs(recent - median))
        if scale == 0:
            # MAD nulle (plus de la moitié des points sont égaux) : on prend l'écart-type,
            # avec un minimum de 1e-6 si la fenêtre est entièrement constante
            scale = max(np.std(recent), 1e-6)
        # Intervalle plat : il ne suit pas les cycles de la série (faiblesse voulue de la méthode)
        low = np.full(HORIZON, median - self.k * scale)
        high = np.full(HORIZON, median + self.k * scale)
        return low, high
