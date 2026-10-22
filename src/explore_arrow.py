"""Opens a BOOM .arrow file, prints its content and exports it to CSV.

Usage: python src/explore_arrow.py [--id ds-277-T]
The CSV is saved to data/<id>.csv (folder ignored by git).
"""

import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset
from huggingface_hub import constants

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="ds-277-T", help="BOOM identifier of the series")
    args = parser.parse_args()

    # Hugging Face cache: the files downloaded by data.py are here
    cache = Path(constants.HF_HUB_CACHE) / "datasets--Datadog--BOOM" / "snapshots"
    path = next(cache.glob(f"*/{args.id}/**/*.arrow"), None)
    if path is None:
        print(f"No .arrow file found for {args.id} in {cache}")
        print("Check the identifier: it must be one of the series downloaded by data.py.")
        return
    print(f"File: {path}\n")

    ds = Dataset.from_file(str(path))
    print(ds, "\n")  # column names and number of rows

    row = ds[0]  # one file = one series
    for name, value in row.items():
        if name != "target":
            print(f"{name}: {value}")  # metadata: frequency, start date...

    target = np.asarray(row["target"], dtype=np.float64)
    print(f"\ntarget shape: {target.shape}  (variables, time) or (time,)")
    print("First 10 values:")
    print(target[..., :10])

    # CSV export: one column per variable, one row per time step
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / f"{args.id}.csv"
    np.savetxt(out, np.atleast_2d(target).T, delimiter=",")
    print(f"\nCSV saved: {out}")


if __name__ == "__main__":
    main()
