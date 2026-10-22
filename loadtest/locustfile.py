"""Step 5.2: load test of the API with Locust.

Each simulated user creates a job with one series, then polls its status until done.
On top of the HTTP requests, we send Locust an "end-to-end" metric: the time between
the job creation and the done status, which includes the wait in the queue and the computation.

Usage (the server runs separately, see README):
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

# One ready-made request per dataset series, with its injected spikes (seed i for series i)
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
    wait_time = constant(0)  # a new job as soon as the previous one is done

    @task
    def detect_one_series(self):
        start = time.perf_counter()
        with self.client.post("/v1/detection-jobs", json=random.choice(PAYLOADS), catch_response=True) as response:
            if response.status_code != 202:  # 429 if the queue is full
                response.failure(f"{response.status_code} {response.text}")
                return
            job_id = response.json()["job_id"]

        # Polling; name groups all the reads under a single line in the statistics
        while True:
            job = self.client.get(f"/v1/detection-jobs/{job_id}", name="/v1/detection-jobs/{job_id}").json()
            if job["status"] in ("done", "failed"):
                break
            time.sleep(POLL_INTERVAL_S)  # Locust makes time.sleep non-blocking for the other users

        # Custom metric: Locust handles it like a request, in ms, with its percentiles
        self.environment.events.request.fire(
            request_type="JOB",
            name="end-to-end",
            response_time=1000 * (time.perf_counter() - start),
            response_length=0,
            exception=None if job["status"] == "done" else Exception(job["error"]["message"]),
            context={},
        )
