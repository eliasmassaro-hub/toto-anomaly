"""The worker: takes the jobs from the queue in batches and runs the detection."""

import asyncio
import os

import numpy as np

from data import CONTEXT
from detectors import find_anomalies

# Batching (step 5.1): the worker takes up to BATCH_SIZE jobs, or those that arrived within MAX_WAIT_MS
# of the first one, and sends them to the model in a single call
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
MAX_WAIT_MS = float(os.environ.get("MAX_WAIT_MS", "50"))


def run_detection(detector, series_list):
    """The pure computation: a single detector call for all series, then the anomalies of each one."""
    histories = np.asarray([s.history[-CONTEXT:] for s in series_list])  # the last 1024 points
    lows, highs = detector.predict_interval_batch(histories)  # (n, 64) each
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
    """Waits for a first job, then takes those arriving within the next MAX_WAIT_MS (BATCH_SIZE at most)."""
    batch = [await queue.get()]
    loop = asyncio.get_running_loop()
    deadline = loop.time() + MAX_WAIT_MS / 1000
    while len(batch) < BATCH_SIZE:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            # Returns immediately if the queue is not empty: under load, the batch fills up without waiting
            batch.append(await asyncio.wait_for(queue.get(), remaining))
        except TimeoutError:
            break
    return batch


async def run_jobs(detector, group):
    """Processes jobs of the same method in a single detector call, then splits the results per job."""
    series = [s for job in group for s in job["series"]]
    try:
        results = await asyncio.to_thread(run_detection, detector, series)
    except Exception as exc:
        for job in group:
            job["status"] = "failed"
            # Same format as the API errors; the exception name helps when str(exc) is empty
            job["error"] = {"error": "internal_error", "message": f"{type(exc).__name__}: {exc}"}
        return
    start = 0
    for job in group:
        n = len(job["series"])
        job["results"] = results[start:start + n]
        job["status"] = "done"
        start += n


async def worker_loop(queue, jobs, detectors):
    """Infinite loop: takes a batch of jobs, processes them, starts again."""
    while True:
        batch = [jobs[job_id] for job_id in await collect_batch(queue)]
        print(f"Batch of {len(batch)} job(s)")
        for job in batch:
            job["status"] = "running"
        try:
            # One detector call per method present in the batch (toto and/or baseline)
            for method, detector in detectors.items():
                group = [job for job in batch if job["method"] == method]
                if group:
                    await run_jobs(detector, group)
        finally:
            for _ in batch:
                queue.task_done()  # tells the queue that these jobs are processed
