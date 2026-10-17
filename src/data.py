"""Étape 1 : chargement de séries BOOM et injection de pics.

Usage : python src/data.py   (télécharge les séries au premier lancement)
"""

from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars

REPO_ID = "Datadog/BOOM"
N_SERIES = 200
CONTEXT = 1024  # longueur de l'historique
HORIZON = 64  # nombre de points à prévoir (= 1 patch Toto)
# On saute les gros fichiers (séries à ~100 variables, jusqu'à 13 Mo) : on n'en garde qu'une variable.
MAX_FILE_BYTES = 2_000_000
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "series.npz"


def download_series(n_series=N_SERIES, seed=0):
    """Télécharge des séries BOOM une par une jusqu'à en avoir n_series valides (étape 1.1).

    Chaque dossier du dataset (ds-<id>-<freq>) contient une seule série, univariée ou
    multivariée. On garde la première variable et ses CONTEXT + HORIZON derniers points.
    Renvoie values (n_series, CONTEXT + HORIZON), ids (noms des dossiers) et freqs.
    """
    disable_progress_bars()
    # La liste des fichiers ne télécharge rien ; on la trie puis on la mélange avec une seed
    # pour que la sélection soit aléatoire mais reproductible.
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
        row = Dataset.from_file(path)[0]  # un fichier = une ligne = une série

        target = np.asarray(row["target"], dtype=np.float64)
        if target.ndim == 2:  # série multivariée, forme (variables, temps)
            target = target[0]  # une seule variable par série

        window = target[-(CONTEXT + HORIZON):]
        if len(window) < CONTEXT + HORIZON:  # série trop courte
            continue
        if not np.isfinite(window).all():  # valeurs manquantes
            continue
        history = window[:CONTEXT]
        if history.max() == history.min():  # historique constant : écart-type nul, pics de hauteur 0
            continue

        values.append(window)
        ids.append(files[i].path.split("/")[0])
        freqs.append(row["freq"])
        if len(values) % 25 == 0:
            print(f"  {len(values)}/{n_series} séries ({n_downloaded} fichiers téléchargés)")
        if len(values) == n_series:
            break

    if len(values) < n_series:
        raise RuntimeError(f"Seulement {len(values)} séries valides trouvées sur {n_series}")
    return np.stack(values), np.array(ids), np.array(freqs)


def load_series(path=DATA_PATH):
    """Charge les séries sauvegardées ; les télécharge et les sauvegarde d'abord si besoin."""
    if not path.exists():
        print(f"{path} absent : téléchargement de {N_SERIES} séries depuis {REPO_ID}")
        values, ids, freqs = download_series()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, values=values, ids=ids, freqs=freqs)
        print(f"Sauvegardé dans {path}")
    data = np.load(path)
    return data["values"], data["ids"], data["freqs"]


def split_history_future(values):
    """Découpe en historique (CONTEXT premiers points) et futur (HORIZON suivants) (étape 1.2).

    Marche pour une série seule (1D) comme pour un tableau de séries (2D).
    """
    return values[..., :CONTEXT], values[..., CONTEXT:CONTEXT + HORIZON]


def inject_spikes(future, history, n_spikes=3, k=5, seed=None):
    """Ajoute n_spikes pics de hauteur k × std(history) à des positions distinctes de future (étape 1.3).

    Renvoie la série modifiée (une copie, future n'est pas modifié) et les positions triées des pics.
    """
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(future), size=n_spikes, replace=False))
    spiked = np.array(future, dtype=np.float64)  # copie
    spiked[positions] += k * np.std(history)
    return spiked, positions


if __name__ == "__main__":
    values, ids, freqs = load_series()
    history, future = split_history_future(values)
    print(f"{len(values)} séries | history {history.shape} | future {future.shape}")
    print(f"Fréquences : {dict(Counter(freqs.tolist()))}")

    spiked, positions = inject_spikes(future[0], history[0], seed=0)
    print(
        f"Exemple ({ids[0]}) : pics aux positions {positions.tolist()}, "
        f"hauteur 5 × std = {5 * np.std(history[0]):.3f}"
    )
