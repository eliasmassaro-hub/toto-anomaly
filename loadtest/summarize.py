"""Étape 5.3 : résume les deux tests de charge de Locust dans results/loadtest.csv.

Usage, après les deux tests (voir README) : python loadtest/summarize.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "loadtest" / "out"  # sorties brutes de Locust (--csv loadtest/out/batch<N>)
RESULTS_PATH = ROOT / "results" / "loadtest.csv"

rows = []
for batch in (1, 16):
    stats = pd.read_csv(OUT_DIR / f"batch{batch}_stats.csv")
    job = stats[stats["Name"] == "bout en bout"].iloc[0]  # la métrique personnalisée du locustfile
    rows.append({
        "Batch": batch,
        "Séries / s": round(job["Requests/s"], 2),  # un job = une série
        "Latence p50 (s)": job["50%"] / 1000,
        "Latence p95 (s)": job["95%"] / 1000,
        "Jobs terminés": int(job["Request Count"]),
        "Échecs": int(job["Failure Count"]),
    })

table = pd.DataFrame(rows)
table.to_csv(RESULTS_PATH, index=False)
print(table.to_string(index=False))
print(f"Enregistré dans {RESULTS_PATH}")
