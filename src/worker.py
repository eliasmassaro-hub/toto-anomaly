"""Le worker : prend les jobs dans la file et lance la détection."""

import asyncio

import numpy as np

from data import CONTEXT
from detectors import find_anomalies


def run_detection(detector, series_list):
    """Le calcul pur : détecte les anomalies de chaque série d'un job."""
    results = []
    for s in series_list:
        history = np.asarray(s.history[-CONTEXT:])  # les 1024 derniers points
        recent = np.asarray(s.recent)
        n = len(recent)
        low, high = detector.predict_interval(history)
        anomalies = find_anomalies(recent, low[:n], high[:n])
        results.append({
            "id": s.id,
            "anomalies": anomalies.tolist(),
            "low": low[:n].tolist(),
            "high": high[:n].tolist(),
        })
    return results


async def worker_loop(queue, jobs, detectors):
    """Boucle infinie : prend un job, le traite, recommence."""
    while True:
        job_id = await queue.get()  # attend un job, sans bloquer le serveur
        job = jobs[job_id]
        job["status"] = "running"
        try:
            detector = detectors[job["method"]]
            job["results"] = await asyncio.to_thread(run_detection, detector, job["series"])
            job["status"] = "done"
        except Exception as exc:
            job["status"] = "failed"
            # Même format que les erreurs de l'API ; le nom de l'exception aide quand str(exc) est vide
            job["error"] = {"error": "internal_error", "message": f"{type(exc).__name__}: {exc}"}
        finally:
            queue.task_done()  # signale à la file que ce job est traité
