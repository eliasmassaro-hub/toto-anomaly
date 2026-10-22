"""Demo client: sends a series to the API and prints the result."""

import argparse
import time

import requests

from data import inject_spikes, load_series, split_history_future

BASE_URL = "http://localhost:8000"
TIMEOUT_S = 120  # polling gives up beyond this


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="dataset series (0 to 199)")
    parser.add_argument("--method", default="toto", help="toto or baseline")
    args = parser.parse_args()

    values, ids, _ = load_series()
    history, future = split_history_future(values[args.index])
    spiked, positions = inject_spikes(future, history, seed=0)

    payload = {
        "method": args.method,
        "series": [
            {
                "id": str(ids[args.index]),
                "history": history.tolist(),
                "recent": spiked.tolist(),
            }
        ],
    }
    response = requests.post(f"{BASE_URL}/v1/detection-jobs", json=payload)
    print(response.status_code, response.json())
    if response.status_code != 202:  # 400 or 429: no job created, no need to wait
        return

    job_id = response.json()["job_id"]

    # Polling: ask for the status again every 0.5 s until done or failed
    start = time.perf_counter()
    while True:
        response = requests.get(f"{BASE_URL}/v1/detection-jobs/{job_id}")
        if response.status_code != 200:  # 404 if the server restarted (in-memory jobs lost)
            print(response.status_code, response.json())
            return
        job = response.json()
        print(f"  {time.perf_counter() - start:.1f} s: {job['status']}")
        if job["status"] in ("done", "failed"):
            break
        if time.perf_counter() - start > TIMEOUT_S:
            print(f"Still {job['status']} after {TIMEOUT_S} s, giving up")
            return
        time.sleep(0.5)

    print(f"Final status: {job['status']} in {time.perf_counter() - start:.1f} s")
    if job["status"] == "done":
        print("Injected spikes:", positions.tolist())
        print("Anomalies found:", job["results"][0]["anomalies"])
    else:
        print("Error:", job["error"]["message"])


if __name__ == "__main__":
    main()
