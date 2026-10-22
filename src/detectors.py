"""Step 2: anomaly detectors (baseline method and Toto).

Common interface: predict_interval(history) -> (low, high), two arrays of HORIZON points,
and predict_interval_batch(histories) for several series at once (used by the API worker).
A future point outside [low, high] is an anomaly (see find_anomalies).
"""

import numpy as np
import torch
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto

from data import HORIZON

MAD_TO_STD = 1.4826  # for a normal distribution, 1.4826 × MAD estimates the standard deviation


class BaselineDetector:
    """Baseline method: constant interval = median ± k × 1.4826 × MAD of the last window points."""

    def __init__(self, window=200, k=3):
        self.window = window
        self.k = k

    def predict_interval(self, history):
        recent = np.asarray(history, dtype=np.float64)[-self.window:]
        median = np.median(recent)
        scale = MAD_TO_STD * np.median(np.abs(recent - median))
        if scale == 0:
            # Zero MAD (more than half of the points are equal): we use the standard deviation,
            # with a minimum of 1e-6 if the window is entirely constant
            scale = max(np.std(recent), 1e-6)
        # Flat interval: it does not follow the cycles of the series (a deliberate weakness of the method)
        low = np.full(HORIZON, median - self.k * scale)
        high = np.full(HORIZON, median + self.k * scale)
        return low, high

    def predict_interval_batch(self, histories):
        """Same interface as TotoDetector: low and high of shape (n, HORIZON). The computation is instant,
        a loop is enough."""
        intervals = [self.predict_interval(history) for history in histories]
        return np.array([low for low, _ in intervals]), np.array([high for _, high in intervals])


def find_anomalies(future, low, high):
    """Positions of the future points outside [low, high]."""
    future = np.asarray(future)
    return np.flatnonzero((future < low) | (future > high))


class TotoDetector:
    """Toto interval: quantiles of a given level over num_samples forecast trajectories."""

    MODEL_ID = "Datadog/Toto-Open-Base-1.0"

    def __init__(self, num_samples=256, level=0.99, max_samples_per_call=512, device=None):
        self.num_samples = num_samples
        # Centered interval of the given level: level = 0.99 gives the 0.005 and 0.995 quantiles
        self.low_q = (1 - level) / 2
        self.high_q = 1 - self.low_q
        # Each series is duplicated num_samples times in the model batch: we cap the number of
        # series × samples per call to bound memory. On CPU (16 GB), 512 uses ~5 GB; at 4096
        # the machine swaps and the call becomes 10 times slower. On GPU, this value can be raised.
        self.max_samples_per_call = max_samples_per_call
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # The model is loaded once (downloaded from Hugging Face on the first run)
        toto = Toto.from_pretrained(self.MODEL_ID).to(self.device)
        toto.eval()
        self.forecaster = TotoForecaster(toto.model)

    def predict_interval(self, history):
        low, high = self.predict_interval_batch([history])
        return low[0], high[0]

    def predict_interval_batch(self, histories):
        """Intervals of several series of the same length; returns low and high of shape (n, HORIZON)."""
        samples = self.predict_samples_batch(histories)
        low, high = np.quantile(samples, [self.low_q, self.high_q], axis=-1)
        return low, high

    def predict_samples_batch(self, histories):
        """Toto samples at the original scale, of shape (n, HORIZON, num_samples).

        A single model call as long as n × num_samples <= max_samples_per_call, otherwise
        the series are split into chunks.
        """
        histories = np.asarray(histories, dtype=np.float64)
        if histories.ndim != 2:
            raise ValueError(f"histories must have shape (n, length), got {histories.shape}")

        # Per-series normalization: Toto's internal scaler has a minimum scale that makes it
        # blind on very flat series (standard deviation ~1e-5, BOOM has some). We therefore
        # forecast (history - mean) / std, then rescale the samples.
        mean = histories.mean(axis=1, keepdims=True)
        std = histories.std(axis=1, keepdims=True)
        std = np.where(std > 0, std, 1.0)  # constant series: no division
        scaled = ((histories - mean) / std).astype(np.float32)

        chunk = max(1, self.max_samples_per_call // self.num_samples)
        samples = np.concatenate(
            [self._sample_chunk(scaled[start:start + chunk]) for start in range(0, len(scaled), chunk)]
        )
        return samples * std[:, :, None] + mean[:, :, None]

    def _sample_chunk(self, histories):
        """One model call for a chunk of series of shape (n, length)."""
        # Toto expects (batch, variables, time): here a single variable per series
        series = torch.from_numpy(histories).to(self.device).unsqueeze(1)
        n = series.shape[0]
        # Same construction as in check_toto.py: timestamps are not used by the model
        inputs = MaskedTimeseries(
            series=series,
            padding_mask=torch.ones_like(series, dtype=torch.bool),  # True = real value
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
                use_kv_cache=False,  # a single patch to forecast: the cache is useless and costs memory
            )
        # samples: (n, 1, HORIZON, num_samples) -> (n, HORIZON, num_samples)
        return forecast.samples[:, 0].cpu().numpy()
