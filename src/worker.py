"""Le worker : prend les jobs dans la file par batchs et lance la détection."""

import asyncio
import os

import numpy as np

from data import CONTEXT
from detectors import find_anomalies

# Batching (étape 5.1) : le worker prend jusqu'à BATCH_SIZE jobs, ou ceux arrivés en MAX_WAIT_MS
# après le premier, et les envoie au modèle en un seul appel
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
MAX_WAIT_MS = float(os.environ.get("MAX_WAIT_MS", "50"))


def run_detection(detector, series_list):
    """Le calcul pur : un seul appel au détecteur pour toutes les séries, puis les anomalies de chacune."""
    histories = np.asarray([s.history[-CONTEXT:] for s in series_list])  # les 1024 derniers points
    lows, highs = detector.predict_interval_batch(histories)  # (n, 64) chacun
    results = []
    for s, low, high in zip(series_list, lows, highs):
        recent = np.asarray(s.recent)
        n = len(recent)
        anomalies = find_anomalies(recent, low[:n], high[:n])
        results.append({
            "id": s.id,
            "anomalies": anomalies.tolist(),
            "low": low[:n].tolist(),
            "high": high[:n].tolist(),
        })
    return results


async def collect_batch(queue):
    """Attend un premier job, puis prend ceux qui arrivent dans les MAX_WAIT_MS suivantes (BATCH_SIZE au plus)."""
    batch = [await queue.get()]
    loop = asyncio.get_running_loop()
    deadline = loop.time() + MAX_WAIT_MS / 1000
    while len(batch) < BATCH_SIZE:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            # Renvoie tout de suite si la file n'est pas vide : sous charge, le batch se remplit sans attendre
            batch.append(await asyncio.wait_for(queue.get(), remaining))
        except TimeoutError:
            break
    return batch


async def run_jobs(detector, group):
    """Traite des jobs de même méthode en un seul appel au détecteur, puis redécoupe les résultats par job."""
    series = [s for job in group for s in job["series"]]
    try:
        results = await asyncio.to_thread(run_detection, detector, series)
    except Exception as exc:
        for job in group:
            job["status"] = "failed"
            # Même format que les erreurs de l'API ; le nom de l'exception aide quand str(exc) est vide
            job["error"] = {"error": "internal_error", "message": f"{type(exc).__name__}: {exc}"}
        return
    start = 0
    for job in group:
        n = len(job["series"])
        job["results"] = results[start:start + n]
        job["status"] = "done"
        start += n


async def worker_loop(queue, jobs, detectors):
    """Boucle infinie : prend un batch de jobs, les traite, recommence."""
    while True:
        batch = [jobs[job_id] for job_id in await collect_batch(queue)]
        print(f"Batch de {len(batch)} job(s)")
        for job in batch:
            job["status"] = "running"
        try:
            # Un appel au détecteur par méthode présente dans le batch (toto et/ou baseline)
            for method, detector in detectors.items():
                group = [job for job in batch if job["method"] == method]
                if group:
                    await run_jobs(detector, group)
        finally:
            for _ in batch:
                queue.task_done()  # signale à la file que ces jobs sont traités
