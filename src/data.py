"""Step 1: loading BOOM series and injecting spikes.

Usage: python src/data.py   (downloads the series on the first run)
"""

from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars

REPO_ID = "Datadog/BOOM"
N_SERIES = 200
CONTEXT = 1024  # history length
HORIZON = 64  # number of points to forecast (= 1 Toto patch)
# Large files are skipped (series with ~100 variables, up to 13 MB): we only keep one variable anyway.
MAX_FILE_BYTES = 2_000_000
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "series.npz"


def download_series(n_series=N_SERIES, seed=0):
    """Downloads BOOM series one by one until n_series valid ones are found (step 1.1).

    Each folder of the dataset (ds-<id>-<freq>) holds a single series, univariate or
    multivariate. We keep the first variable and its last CONTEXT + HORIZON points.
    Returns values (n_series, CONTEXT + HORIZON), ids (folder names) and freqs.
    """
    disable_progress_bars()
    # Listing the files downloads nothing; we sort the list then shuffle it with a seed
    # so that the selection is random but reproducible.
    files = HfApi().list_repo_tree(REPO_ID, repo_type="dataset", recursive=True)
    files = sorted(
        (f for f in files if f.path.endswith(".arrow") and f.size <= MAX_FILE_BYTES),
        key=lambda f: f.path,
    )
    order = np.random.default_rng(seed).permutation(len(files))

    values, ids, freqs = [], [], []
    n_downloaded = 0
    for i in order:
        path = hf_hub_download(REPO_ID, files[i].path, repo_type="dataset")
        n_downloaded += 1
        row = Dataset.from_file(path)[0]  # one file = one row = one series

        target = np.asarray(row["target"], dtype=np.float64)
        if target.ndim == 2:  # multivariate series, shape (variables, time)
            target = target[0]  # a single variable per series

        window = target[-(CONTEXT + HORIZON):]
        if len(window) < CONTEXT + HORIZON:  # series too short
            continue
        if not np.isfinite(window).all():  # missing values
            continue
        history = window[:CONTEXT]
        if history.max() == history.min():  # constant history: zero standard deviation, spikes of height 0
            continue

        values.append(window)
        ids.append(files[i].path.split("/")[0])
        freqs.append(row["freq"])
        if len(values) % 25 == 0:
            print(f"  {len(values)}/{n_series} series ({n_downloaded} files downloaded)")
        if len(values) == n_series:
            break

    if len(values) < n_series:
        raise RuntimeError(f"Only {len(values)} valid series found out of {n_series}")
    return np.stack(values), np.array(ids), np.array(freqs)


def load_series(path=DATA_PATH):
    """Loads the saved series; downloads and saves them first if needed."""
    if not path.exists():
        print(f"{path} missing: downloading {N_SERIES} series from {REPO_ID}")
        values, ids, freqs = download_series()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, values=values, ids=ids, freqs=freqs)
        print(f"Saved to {path}")
    data = np.load(path)
    return data["values"], data["ids"], data["freqs"]


def split_history_future(values):
    """Splits into history (first CONTEXT points) and future (next HORIZON points) (step 1.2).

    Works for a single series (1D) as well as for an array of series (2D).
    """
    return values[..., :CONTEXT], values[..., CONTEXT:CONTEXT + HORIZON]


def inject_spikes(future, history, n_spikes=3, k=5, seed=None):
    """Adds n_spikes spikes of height k × std(history) at distinct positions of future (step 1.3).

    Returns the modified series (a copy, future is left unchanged) and the sorted spike positions.
    """
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(future), size=n_spikes, replace=False))
    spiked = np.array(future, dtype=np.float64)  # copy
    spiked[positions] += k * np.std(history)
    return spiked, positions


if __name__ == "__main__":
    values, ids, freqs = load_series()
    history, future = split_history_future(values)
    print(f"{len(values)} series | history {history.shape} | future {future.shape}")
    print(f"Frequencies: {dict(Counter(freqs.tolist()))}")

    spiked, positions = inject_spikes(future[0], history[0], seed=0)
    print(
        f"Example ({ids[0]}): spikes at positions {positions.tolist()}, "
        f"height 5 × std = {5 * np.std(history[0]):.3f}"
    )
