"""Step 4: anomaly detection API."""

import asyncio
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from detectors import BaselineDetector, TotoDetector
from schemas import JobRequest
from worker import BATCH_SIZE, MAX_WAIT_MS, worker_loop

# Settings chosen in step 3 bis. The load test uses TOTO_SAMPLES=32 so that a batch
# of 16 series fits in one model call (16 × 32 = 512 samples, see README)
TOTO_SAMPLES = int(os.environ.get("TOTO_SAMPLES", "256"))
TOTO_LEVEL = 0.99
DETECTORS: dict = {}

QUEUE_MAXSIZE = 100  # beyond this, the POST returns 429
QUEUE: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

# Short error name for each status code
ERROR_NAMES = {400: "invalid_request", 404: "not_found", 429: "queue_full", 500: "internal_error"}

# In-memory job storage: {job_id: {"status": ..., "method": ..., "series": ..., "results": ..., "error": ...}}
JOBS: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Server startup ---
    print("Loading the detectors…")
    DETECTORS["baseline"] = BaselineDetector(k=3)
    DETECTORS["toto"] = TotoDetector(num_samples=TOTO_SAMPLES, level=TOTO_LEVEL)
    worker_task = asyncio.create_task(worker_loop(QUEUE, JOBS, DETECTORS))
    print(f"Ready (BATCH_SIZE={BATCH_SIZE}, MAX_WAIT_MS={MAX_WAIT_MS:g}, TOTO_SAMPLES={TOTO_SAMPLES}).")
    yield  # the server runs here, until Ctrl+C
    # --- Server shutdown ---
    worker_task.cancel()


app = FastAPI(title="Anomaly detection API", lifespan=lifespan)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    """Errors raised with HTTPException (404, 429…), and unknown paths."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": ERROR_NAMES.get(exc.status_code, "error"), "message": str(exc.detail)},
    )


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    """Request rejected by Pydantic: 400 instead of the default 422, with the first problem found."""
    first = exc.errors()[0]
    where = ".".join(str(part) for part in first["loc"])  # e.g. "body.series.0.history"
    return JSONResponse(
        status_code=400,
        content={"error": "invalid_request", "message": f"{where}: {first['msg']}"},
    )


@app.exception_handler(Exception)
async def internal_error(request: Request, exc: Exception):
    """Any other unexpected error: 500 in the same format (the details stay in the server logs)."""
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "internal server error"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/detection-jobs", status_code=202)
async def create_job(body: JobRequest):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "pending",
        "method": body.method,
        "series": body.series,
        "results": None,
        "error": None,
    }
    try:
        QUEUE.put_nowait(job_id)
    except asyncio.QueueFull:
        del JOBS[job_id]
        raise HTTPException(status_code=429, detail=f"queue full ({QUEUE_MAXSIZE} jobs), retry later")
    return {"job_id": job_id, "status": "pending"}


@app.get("/v1/detection-jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)  # .get returns None if the key does not exist
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "results": job["results"],
        "error": job["error"],
    }
