"""Step 0.3: checks that Toto loads and makes a forecast on a random series.

Usage: python src/check_toto.py [--num-samples 256]
"""

import argparse
import time

import numpy as np
import torch
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto

CONTEXT = 1024  # history length
HORIZON = 64  # number of forecast points (= 1 Toto patch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Model loading (downloaded from Hugging Face on the first run)
    t0 = time.perf_counter()
    toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").to(device)
    toto.eval()
    forecaster = TotoForecaster(toto.model)
    print(f"Model loaded in {time.perf_counter() - t0:.1f} s")

    # Random series: sine wave + noise, shape (variables=1, time=CONTEXT)
    rng = np.random.default_rng(0)
    t = np.arange(CONTEXT)
    values = 10 + 3 * np.sin(2 * np.pi * t / 100) + rng.normal(0, 0.5, CONTEXT)
    series = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)

    # Toto expects a MaskedTimeseries. Timestamps are not used by the current model,
    # but the API requires them: we pass zeros and a 60 s step.
    inputs = MaskedTimeseries(
        series=series,
        padding_mask=torch.ones_like(series, dtype=torch.bool),  # True = real value
        id_mask=torch.zeros_like(series, dtype=torch.int),
        timestamp_seconds=torch.zeros_like(series, dtype=torch.int),
        time_interval_seconds=torch.full((1,), 60, device=device, dtype=torch.int),
    )

    # Probabilistic forecast: num_samples trajectories, from which we take the quantiles
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

    print(f"Samples: {tuple(forecast.samples.shape)}  (batch, variables, horizon, num_samples)")
    print(f"Interval [q0.005, q0.995] on the first 5 points:")
    for i in range(5):
        print(f"  t+{i + 1}: [{low[i]:.2f}, {high[i]:.2f}]")
    print(f"Forecast time ({args.num_samples} samples): {elapsed * 1000:.0f} ms")


if __name__ == "__main__":
    main()
