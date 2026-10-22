"""Étape 4 : API de détection d'anomalies."""

import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from detectors import BaselineDetector, TotoDetector
from schemas import JobRequest
from worker import worker_loop

TOTO_SAMPLES = 256  # réglages retenus à l'étape 3 bis
TOTO_LEVEL = 0.99
DETECTORS: dict = {}

QUEUE_MAXSIZE = 100  # au-delà, le POST renvoie 429
QUEUE: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

# Nom court de l'erreur selon le code
ERROR_NAMES = {400: "invalid_request", 404: "not_found", 429: "queue_full", 500: "internal_error"}

# Stockage des jobs en mémoire : {job_id: {"status": ..., "method": ..., "series": ..., "results": ..., "error": ...}}
JOBS: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Démarrage du serveur ---
    print("Chargement des détecteurs…")
    DETECTORS["baseline"] = BaselineDetector(k=3)
    DETECTORS["toto"] = TotoDetector(num_samples=TOTO_SAMPLES, level=TOTO_LEVEL)
    worker_task = asyncio.create_task(worker_loop(QUEUE, JOBS, DETECTORS))
    print("Prêt.")
    yield  # le serveur tourne ici, jusqu'à Ctrl+C
    # --- Arrêt du serveur ---
    worker_task.cancel()


app = FastAPI(title="API de détection d'anomalies", lifespan=lifespan)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    """Erreurs levées avec HTTPException (404, 429…), et chemins inconnus."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": ERROR_NAMES.get(exc.status_code, "error"), "message": str(exc.detail)},
    )


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    """Requête refusée par Pydantic : 400 au lieu du 422 par défaut, avec le premier problème trouvé."""
    first = exc.errors()[0]
    where = ".".join(str(part) for part in first["loc"])  # ex. "body.series.0.history"
    return JSONResponse(
        status_code=400,
        content={"error": "invalid_request", "message": f"{where} : {first['msg']}"},
    )


@app.exception_handler(Exception)
async def internal_error(request: Request, exc: Exception):
    """Toute autre erreur imprévue : 500 au même format (le détail reste dans les logs du serveur)."""
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "erreur interne du serveur"},
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
        raise HTTPException(status_code=429, detail=f"file d'attente pleine ({QUEUE_MAXSIZE} jobs), réessayer plus tard")
    return {"job_id": job_id, "status": "pending"}


@app.get("/v1/detection-jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)  # .get renvoie None si la clé n'existe pas
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} introuvable")
    return {
        "job_id": job_id,
        "status": job["status"],
        "results": job["results"],
        "error": job["error"],
    }
