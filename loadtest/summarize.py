"""Step 5.3: summarizes the two Locust load tests in results/loadtest.csv.

Usage, after the two tests (see README): python loadtest/summarize.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "loadtest" / "out"  # raw Locust outputs (--csv loadtest/out/batch<N>)
RESULTS_PATH = ROOT / "results" / "loadtest.csv"

rows = []
for batch in (1, 16):
    stats = pd.read_csv(OUT_DIR / f"batch{batch}_stats.csv")
    job = stats[stats["Name"] == "end-to-end"].iloc[0]  # the custom metric of the locustfile
    rows.append({
        "Batch": batch,
        "Series / s": round(job["Requests/s"], 2),  # one job = one series
        "p50 latency (s)": job["50%"] / 1000,
        "p95 latency (s)": job["95%"] / 1000,
        "Completed jobs": int(job["Request Count"]),
        "Failures": int(job["Failure Count"]),
    })

table = pd.DataFrame(rows)
table.to_csv(RESULTS_PATH, index=False)
print(table.to_string(index=False))
print(f"Saved to {RESULTS_PATH}")
