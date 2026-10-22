# Anomaly detection API specification

## Endpoints

| Method | Path | Purpose | Possible response codes |
|---|---|---|---|
| GET | /health | check that the server is running | 200 |
| POST | /v1/detection-jobs | create a detection job | 202, 400, 429 |
| GET | /v1/detection-jobs/{job_id} | read the status and result of a job | 200, 404 |

Any endpoint can also return `500` on an unexpected server error (see the error format).

## GET /health

`200` response:

```json
{"status": "ok"}
```

## POST /v1/detection-jobs

Request body:

```json
{
  "method": "toto",
  "series": [
    {"id": "payment-api-errors",
     "history": [0.30, 0.31, ...],
     "recent": [0.29, 8.1, ...]}
  ]
}
```

Validation rules (otherwise `400`, and no job is created):

- `method` is `"toto"` or `"baseline"` (`"toto"` by default);
- `series` contains between 1 and 100 series;
- each series has an `id` (string);
- `history` contains at least 1024 numbers (only the last 1024 are used);
- `recent` contains between 1 and 64 numbers;
- `NaN` and `Infinity` are rejected in `history` and `recent`.

`202` response:

```json
{"job_id": "a3f9c2...", "status": "pending"}
```

`400` response (invalid request, the message names the first field at fault):

```json
{"error": "invalid_request", "message": "body.series.0.history: List should have at least 1024 items after validation, not 10"}
```

`429` response (queue full, retry later):

```json
{"error": "queue_full", "message": "queue full (100 jobs), retry later"}
```

## GET /v1/detection-jobs/{job_id}

`200` response, job waiting or in progress (`"running"` once the worker has taken it):

```json
{"job_id": "a3f9c2...", "status": "pending", "results": null, "error": null}
```

`200` response, job done:

```json
{"job_id": "a3f9c2...", "status": "done", "error": null,
 "results": [{"id": "payment-api-errors",
              "anomalies": [1],
              "low": [...], "high": [...]}]}
```

- `anomalies`: positions of the anomalous points in `recent` (0 = first point);
- `low` and `high`: the forecast interval, of the same length as `recent`.

`200` response, job failed during the computation (`error` follows the single error format):

```json
{"job_id": "a3f9c2...", "status": "failed", "results": null,
 "error": {"error": "internal_error", "message": "ValueError: ..."}}
```

`404` response (unknown job):

```json
{"error": "not_found", "message": "job a3f9c2... not found"}
```

## Job statuses

`pending` (in the queue) → `running` (being computed) → `done` (finished) or `failed` (error during the computation)

## Two kinds of errors

- **The client made a mistake** (invalid request): `400` response to the POST, no job is created.
- **The server failed during the computation**: the job exists and moves to `failed`.

## Error format (always the same)

```json
{"error": "<short code>", "message": "<human-readable explanation>"}
```

Short codes: `invalid_request` (400), `not_found` (404), `queue_full` (429), `internal_error` (500, or in the `error` field of a `failed` job).

This format is also used for the `error` field of a `failed` job: the client reads all errors the same way.

## Known limitations

- **Jobs stored in memory**: lost if the server restarts, and impossible to share between several servers (Redis or a database would be needed).
- **A single worker**: it processes jobs in batches (`BATCH_SIZE` jobs at most, or those that arrived within `MAX_WAIT_MS`); a batch that fails moves all its jobs to `failed`.
- **No cleanup**: finished jobs stay in memory forever (a time to live, or TTL, would be needed).
- **No idempotency**: if the client sends the same request again (after a network failure, for example), two jobs are created.
- **No authentication** and no per-client limit (rate limiting).
- **Polling**: the client has to ask for the status again; an alternative would be to notify it when the job is done (webhook).
