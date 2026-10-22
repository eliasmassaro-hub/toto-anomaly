"""Client de démonstration : envoie une série à l'API et affiche le résultat."""

import argparse
import time

import requests

from data import inject_spikes, load_series, split_history_future

BASE_URL = "http://localhost:8000"
TIMEOUT_S = 120  # abandon du polling au-delà


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="série du dataset (0 à 199)")
    parser.add_argument("--method", default="toto", help="toto ou baseline")
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
    if response.status_code != 202:  # 400 ou 429 : pas de job créé, inutile d'attendre
        return

    job_id = response.json()["job_id"]

    # Polling : redemander le statut toutes les 0,5 s jusqu'à done ou failed
    start = time.perf_counter()
    while True:
        response = requests.get(f"{BASE_URL}/v1/detection-jobs/{job_id}")
        if response.status_code != 200:  # 404 si le serveur a redémarré (jobs en mémoire perdus)
            print(response.status_code, response.json())
            return
        job = response.json()
        print(f"  {time.perf_counter() - start:.1f} s : {job['status']}")
        if job["status"] in ("done", "failed"):
            break
        if time.perf_counter() - start > TIMEOUT_S:
            print(f"Toujours {job['status']} au bout de {TIMEOUT_S} s, abandon")
            return
        time.sleep(0.5)

    print(f"Statut final : {job['status']} en {time.perf_counter() - start:.1f} s")
    if job["status"] == "done":
        print("Pics injectés      :", positions.tolist())
        print("Anomalies trouvées :", job["results"][0]["anomalies"])
    else:
        print("Erreur :", job["error"]["message"])


if __name__ == "__main__":
    main()
