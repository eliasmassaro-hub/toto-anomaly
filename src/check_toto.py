"""Étape 0.3 : vérifie que Toto se charge et fait une prévision sur une série aléatoire.

Usage : python src/check_toto.py [--num-samples 256]
"""

import argparse
import time

import numpy as np
import torch
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto

CONTEXT = 1024  # longueur de l'historique
HORIZON = 64  # nombre de points prévus (= 1 patch Toto)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # Chargement du modèle (téléchargé depuis Hugging Face au premier lancement)
    t0 = time.perf_counter()
    toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").to(device)
    toto.eval()
    forecaster = TotoForecaster(toto.model)
    print(f"Modèle chargé en {time.perf_counter() - t0:.1f} s")

    # Série aléatoire : sinusoïde + bruit, forme (variables=1, temps=CONTEXT)
    rng = np.random.default_rng(0)
    t = np.arange(CONTEXT)
    values = 10 + 3 * np.sin(2 * np.pi * t / 100) + rng.normal(0, 0.5, CONTEXT)
    series = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)

    # Toto attend un MaskedTimeseries. Les timestamps ne sont pas utilisés par le modèle actuel,
    # mais l'API les demande : on met des zéros et un pas de 60 s.
    inputs = MaskedTimeseries(
        series=series,
        padding_mask=torch.ones_like(series, dtype=torch.bool),  # True = vraie valeur
        id_mask=torch.zeros_like(series, dtype=torch.int),
        timestamp_seconds=torch.zeros_like(series, dtype=torch.int),
        time_interval_seconds=torch.full((1,), 60, device=device, dtype=torch.int),
    )

    # Prévision probabiliste : num_samples trajectoires, dont on tire les quantiles
    t0 = time.perf_counter()
    with torch.inference_mode():
        forecast = forecaster.forecast(
            inputs,
            prediction_length=HORIZON,
            num_samples=args.num_samples,
            samples_per_batch=args.num_samples,
        )
        low = forecast.quantile(0.005)[0, 0].cpu().numpy()
        high = forecast.quantile(0.995)[0, 0].cpu().numpy()
    elapsed = time.perf_counter() - t0

    print(f"Échantillons : {tuple(forecast.samples.shape)}  (batch, variables, horizon, num_samples)")
    print(f"Intervalle [q0.005, q0.995] sur les 5 premiers points :")
    for i in range(5):
        print(f"  t+{i + 1} : [{low[i]:.2f}, {high[i]:.2f}]")
    print(f"Temps de la prévision ({args.num_samples} échantillons) : {elapsed * 1000:.0f} ms")


if __name__ == "__main__":
    main()
