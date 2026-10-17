"""Ouvre un fichier .arrow de BOOM, affiche son contenu et l'exporte en CSV.

Usage : python src/explore_arrow.py [--id ds-277-T]
Le CSV est enregistré dans data/<id>.csv (dossier ignoré par git).
"""

import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset
from huggingface_hub import constants

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="ds-277-T", help="identifiant BOOM de la série")
    args = parser.parse_args()

    # Cache Hugging Face : les fichiers téléchargés par data.py sont ici
    cache = Path(constants.HF_HUB_CACHE) / "datasets--Datadog--BOOM" / "snapshots"
    path = next(cache.glob(f"*/{args.id}/**/*.arrow"), None)
    if path is None:
        print(f"Aucun fichier .arrow trouvé pour {args.id} dans {cache}")
        print("Vérifie l'identifiant : il doit faire partie des séries téléchargées par data.py.")
        return
    print(f"Fichier : {path}\n")

    ds = Dataset.from_file(str(path))
    print(ds, "\n")  # noms des colonnes et nombre de lignes

    row = ds[0]  # un fichier = une série
    for name, value in row.items():
        if name != "target":
            print(f"{name} : {value}")  # métadonnées : fréquence, date de début...

    target = np.asarray(row["target"], dtype=np.float64)
    print(f"\nforme de target : {target.shape}  (variables, temps) ou (temps,)")
    print("10 premières valeurs :")
    print(target[..., :10])

    # Export CSV : une colonne par variable, une ligne par pas de temps
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / f"{args.id}.csv"
    np.savetxt(out, np.atleast_2d(target).T, delimiter=",")
    print(f"\nCSV enregistré : {out}")


if __name__ == "__main__":
    main()
