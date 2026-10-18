"""Étape 2 : détecteurs d'anomalies (méthode simple et Toto).

Interface commune : predict_interval(history) -> (low, high), deux tableaux de HORIZON points.
Un point du futur hors de [low, high] est une anomalie (voir find_anomalies).
"""

import numpy as np
import torch
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto

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


def find_anomalies(future, low, high):
    """Positions des points de future hors de [low, high]."""
    future = np.asarray(future)
    return np.flatnonzero((future < low) | (future > high))


class TotoDetector:
    """Intervalle de Toto : quantiles d'un niveau donné sur num_samples trajectoires prévues."""

    MODEL_ID = "Datadog/Toto-Open-Base-1.0"

    def __init__(self, num_samples=256, level=0.99, max_samples_per_call=512, device=None):
        self.num_samples = num_samples
        # Intervalle centré de niveau level : level = 0,99 donne les quantiles 0,005 et 0,995
        self.low_q = (1 - level) / 2
        self.high_q = 1 - self.low_q
        # Chaque série est dupliquée num_samples fois dans le batch du modèle : on limite le
        # nombre de séries × échantillons par appel pour borner la mémoire. Sur CPU (16 Go),
        # 512 utilise ~5 Go ; à 4096 la machine swappe et l'appel devient 10 fois plus lent.
        # Sur GPU, on peut augmenter cette valeur.
        self.max_samples_per_call = max_samples_per_call
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Chargement unique du modèle (téléchargé depuis Hugging Face au premier lancement)
        toto = Toto.from_pretrained(self.MODEL_ID).to(self.device)
        toto.eval()
        self.forecaster = TotoForecaster(toto.model)

    def predict_interval(self, history):
        low, high = self.predict_interval_batch([history])
        return low[0], high[0]

    def predict_interval_batch(self, histories):
        """Intervalles de plusieurs séries de même longueur ; renvoie low et high de forme (n, HORIZON)."""
        samples = self.predict_samples_batch(histories)
        low, high = np.quantile(samples, [self.low_q, self.high_q], axis=-1)
        return low, high

    def predict_samples_batch(self, histories):
        """Tirages de Toto à l'échelle d'origine, de forme (n, HORIZON, num_samples).

        Un seul appel au modèle tant que n × num_samples <= max_samples_per_call, sinon
        les séries sont découpées en paquets.
        """
        histories = np.asarray(histories, dtype=np.float64)
        if histories.ndim != 2:
            raise ValueError(f"histories doit être de forme (n, longueur), reçu {histories.shape}")

        # Normalisation de chaque série : le normaliseur interne de Toto a une échelle minimale
        # qui le rend aveugle sur les séries très plates (écart-type ~1e-5, il y en a dans BOOM).
        # On prévoit donc (history - moyenne) / écart-type, puis on remet les tirages à l'échelle.
        mean = histories.mean(axis=1, keepdims=True)
        std = histories.std(axis=1, keepdims=True)
        std = np.where(std > 0, std, 1.0)  # série constante : pas de division
        scaled = ((histories - mean) / std).astype(np.float32)

        chunk = max(1, self.max_samples_per_call // self.num_samples)
        samples = np.concatenate(
            [self._sample_chunk(scaled[start:start + chunk]) for start in range(0, len(scaled), chunk)]
        )
        return samples * std[:, :, None] + mean[:, :, None]

    def _sample_chunk(self, histories):
        """Un appel au modèle pour un paquet de séries de forme (n, longueur)."""
        # Toto attend (batch, variables, temps) : ici une seule variable par série
        series = torch.from_numpy(histories).to(self.device).unsqueeze(1)
        n = series.shape[0]
        # Même construction que dans check_toto.py : timestamps non utilisés par le modèle
        inputs = MaskedTimeseries(
            series=series,
            padding_mask=torch.ones_like(series, dtype=torch.bool),  # True = vraie valeur
            id_mask=torch.zeros_like(series, dtype=torch.int),
            timestamp_seconds=torch.zeros_like(series, dtype=torch.int),
            time_interval_seconds=torch.full((n, 1), 60, device=self.device, dtype=torch.int),
        )
        with torch.inference_mode():
            forecast = self.forecaster.forecast(
                inputs,
                prediction_length=HORIZON,
                num_samples=self.num_samples,
                samples_per_batch=self.num_samples,
                use_kv_cache=False,  # un seul patch à prévoir : le cache ne sert pas et coûte de la mémoire
            )
        # samples : (n, 1, HORIZON, num_samples) -> (n, HORIZON, num_samples)
        return forecast.samples[:, 0].cpu().numpy()
