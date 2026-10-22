"""Étape 5.2 : test de charge de l'API avec Locust.

Chaque utilisateur simulé crée un job d'une série, puis interroge son statut jusqu'à done.
En plus des requêtes HTTP, on envoie à Locust une métrique « bout en bout » : le temps entre
la création du job et le statut done, qui compte l'attente dans la file et le calcul.

Usage (le serveur tourne à part, voir README) :
    locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 32 -r 32 -t 3m --csv loadtest/out/batch16
"""

import random
import sys
import time
from pathlib import Path

from locust import HttpUser, constant, task

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import inject_spikes, load_series, split_history_future  # noqa: E402

POLL_INTERVAL_S = 0.2

# Une requête prête par série du dataset, avec ses pics injectés (seed i pour la série i)
values, ids, _ = load_series()
histories, futures = split_history_future(values)
PAYLOADS = []
for i in range(len(values)):
    spiked, _ = inject_spikes(futures[i], histories[i], seed=i)
    PAYLOADS.append({
        "method": "toto",
        "series": [{"id": str(ids[i]), "history": histories[i].tolist(), "recent": spiked.tolist()}],
    })


class DetectionUser(HttpUser):
    wait_time = constant(0)  # un nouveau job dès que le précédent est fini

    @task
    def detect_one_series(self):
        start = time.perf_counter()
        with self.client.post("/v1/detection-jobs", json=random.choice(PAYLOADS), catch_response=True) as response:
            if response.status_code != 202:  # 429 si la file est pleine
                response.failure(f"{response.status_code} {response.text}")
                return
            job_id = response.json()["job_id"]

        # Polling ; name regroupe toutes les lectures sous une seule ligne dans les statistiques
        while True:
            job = self.client.get(f"/v1/detection-jobs/{job_id}", name="/v1/detection-jobs/{job_id}").json()
            if job["status"] in ("done", "failed"):
                break
            time.sleep(POLL_INTERVAL_S)  # Locust rend time.sleep non bloquant pour les autres utilisateurs

        # Métrique personnalisée : Locust la traite comme une requête, en ms, avec ses percentiles
        self.environment.events.request.fire(
            request_type="JOB",
            name="bout en bout",
            response_time=1000 * (time.perf_counter() - start),
            response_length=0,
            exception=None if job["status"] == "done" else Exception(job["error"]["message"]),
            context={},
        )
