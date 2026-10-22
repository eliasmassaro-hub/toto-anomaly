# Project: incident detection on metrics with Toto

## Context

Datadog monitors the metrics of its customers' systems: error rate, latency, request count, CPU, memory. When an incident happens, a metric often spikes:

- a buggy deployment makes the error rate of an API explode;
- a slow database query makes latency go up;
- an abnormal traffic surge (bot, attack) makes the request count explode.

The goal is to catch these spikes within minutes, before customers notice, without triggering false alarms (too many false alarms and on-call engineers end up ignoring them). Datadog offers *Anomaly monitors* for this.

This project tests Toto, Datadog's open-source forecasting model, for this use case. Toto forecasts an interval for the next points of a series, and a real point that falls outside this interval is an anomaly. Toto is compared to a baseline method, then served behind an asynchronous API with batching and load tested.

The data are real metrics from the BOOM dataset, on which incidents are simulated as spikes.

## Rules for Claude Code

- Do **one step at a time**, only the one requested. Some steps will be coded by the user: do not implement them without an explicit request.
- This is a learning project: simple, readable, briefly commented code. No over-engineering, no unnecessary dependencies.
- At the end of each step, explain in a few sentences what was done and how to check it, then make a git commit with a message like `Step 1.2: ...`.
- If a file already exists (written by the user), read it and use it as is instead of rewriting it.
- Steps 0 to 3.3 are done (`src/check_toto.py`, `src/data.py`, `src/plot_data.py`, `src/detectors.py`, `src/plot_detectors.py`, `src/evaluate.py`). Reuse these files instead of rewriting them.

## Fixed parameters

| Parameter | Value |
|---|---|
| Number of series | 200 (a single variable per series) |
| History (context) | 1024 points |
| Forecast horizon | 64 points (= 1 Toto patch) |
| Injected spikes | 3 per series, height = 5 × the standard deviation of the history |
| Baseline method | median and MAD of the last 200 points, interval = median ± k × 1.4826 × MAD, k = 3 by default |
| Toto | model `Datadog/Toto-Open-Base-1.0`, 0.005 and 0.995 quantiles, configurable `num_samples` |

## Project structure

```
toto-anomaly/
  src/check_toto.py      # step 0 (done)
  src/data.py            # BOOM loading + spike injection
  src/detectors.py       # baseline method + Toto
  src/evaluate.py        # evaluation outside the API
  src/api.py             # FastAPI API
  src/worker.py          # queue + batching
  client.py              # demo client
  loadtest/locustfile.py
  docs/api_spec.md
  results/               # tables and figures
  requirements.txt
  README.md
```

---

## Step 0: Setup (done)

Folder structure, `.gitignore`, `requirements.txt`, virtual environment, `src/check_toto.py` (Toto loads and forecasts 64 points in about 440 ms on CPU with 32 samples), git repo linked to GitHub.

## Step 1: Data (`src/data.py`) (done)

- **1.1** Load 200 series from the Hugging Face dataset `Datadog/BOOM` without downloading everything (2.8 GB). To do so, first look at the file structure of the dataset and at the loading code in the GitHub repo `DataDog/toto` (`boom` folder). Keep a single variable per series, and only keep the series with at least 1024 + 64 points. If a domain label is available, favor the "application" and "infrastructure" domains. Save locally (`data/series.npz` or parquet).
- **1.2** Split each series into `history` (1024 points) and `future` (the next 64 points).
- **1.3** Write `inject_spikes(future, history, n_spikes=3, k=5, seed=None)`: adds `n_spikes` spikes of height `k × std(history)` at distinct random positions of `future`. Returns the modified series and the spike positions.
- **1.4** Visualization script: plot a series (history + future) with the injected spikes marked. Save it in `results/`.

## Step 2: Detectors (`src/detectors.py`) (done)

Common interface: each detector has a `predict_interval(history) -> (low, high)` method, two arrays of length 64.

- **2.1** `BaselineDetector(window=200, k=3)`: median and MAD of the last `window` points of the history, constant interval = median ± k × 1.4826 × MAD. Two points to watch:
  - **Zero MAD.** On an almost flat series (for example `ds-1525-5T`, which stays at −0.5 with regular jumps), the MAD is 0 and the interval has zero width. If `1.4826 × MAD` is zero, use the standard deviation of the last `window` points instead, with a minimum of `1e-6` if the window is entirely constant.
  - **Flat interval.** The interval is the same for the 64 points and does not follow the cycles of the series. On a cyclic series (for example `ds-277-T`), it must cover the whole cycle and can miss a spike placed in a trough. On a series with regular jumps, it flags the normal jumps as anomalies. This is an expected weakness of the baseline method: **do not fix it**, it is what the comparison with Toto should highlight.
- **2.2** `find_anomalies(future, low, high)`: returns the positions of the points of `future` outside `[low, high]`.
- **2.3** `TotoDetector(num_samples=256)`: loads Toto once, forecasts 64 points from the history, returns the 0.005 and 0.995 quantiles. Add a `predict_interval_batch(histories)` method that processes several series in a single model call. Split it into chunks if memory is not enough.
- **2.4** Visualization script: a series with both intervals, the injected spikes, and the points detected by each method.

## Step 3: Evaluation (`src/evaluate.py`) (done)

- **3.1** Metric functions:
  - recall = detected spikes / injected spikes;
  - false alarms = flagged points that are not spikes;
  - mean computation time per series in ms (`time.perf_counter()`), after a warm-up call.
- **3.2** Evaluation loop over the 200 series for three configurations: baseline method, Toto with 32 samples, Toto with 256 samples (on Google Colab with a GPU if it is too slow locally). Same seed for the spikes in all three cases.
- **3.3** Save the table in `results/eval.csv` and print it.

Results obtained on 200 series (600 spikes):

| Detector | Recall | Precision | False alarms / series | ms / series |
|---|---|---|---|---|
| Baseline (k = 3) | 97.7% | 23.2% | 9.7 | 0.03 |
| Toto, 32 samples | 91.7% | 29.6% | 6.6 | 274 |
| Toto, 256 samples | 82.3% | 46.7% | 2.8 | 2048 |

These configurations are not tuned to the same level of false alarms, so we cannot yet say which method is best. Step 3 bis makes the comparison fair.

## Step 3 bis: tuning and fair comparison (`src/tuning.py`)

Goal: tune each method, then compare them at an equal number of false alarms, without overestimating the results. Reuse the functions of `evaluate.py` and `detectors.py`.

- **3.4** Metrics. Add to `evaluate.py`, on top of recall and false alarms:
  - true positives (TP) = flagged injected spikes; false positives (FP) = flagged points that are not spikes;
  - **precision** = TP / (TP + FP), computed on the totals over all series (not averaged per series, to avoid divisions by zero on series without any alarm);
  - **F1** = 2 × precision × recall / (precision + recall);
  - **false alarms per series** = FP / number of series.

  Add these columns to `results/eval.csv`.
- **3.5** Validation / test split. Split the 200 series into two random halves of 100, with a fixed seed:
  - the **validation** set is only used to choose the settings;
  - the **test** set is only used to report the final results.

  The injected spikes stay the same as in step 3 (seed i for series i).
- **3.6** Baseline method: sweep over `k` on the validation set, with k = 2, 3, 4, 5, 6, 8, 10. For each k, compute recall, precision, F1 and false alarms per series. Save in `results/tuning_baseline.csv`.
- **3.7** Toto: sweep over the interval level on the validation set, with 256 samples.

  | Level | Quantiles |
  |---|---|
  | 90% | 0.05 and 0.95 |
  | 95% | 0.025 and 0.975 |
  | 98% | 0.01 and 0.99 |
  | 99% | 0.005 and 0.995 (current setting) |
  | 99.5% | 0.0025 and 0.9975 |

  **Run Toto only once per series.** Add to `TotoDetector` a `predict_samples_batch(histories)` method that returns the samples rescaled to the original scale (same normalization as `predict_interval_batch`), of shape `(n, 64, num_samples)`. Then compute all the quantiles with `np.quantile` on these same samples. Choose the level that maximizes F1. Save in `results/tuning_toto_quantiles.csv`.
- **3.8** Toto: number of samples on the validation set, at the level chosen in 3.7, with 32, 64, 128 and 256 samples. For each value, run Toto again to get a real computation time, and compute recall, precision, F1 and ms per series. Choose the smallest number of samples whose F1 is within 0.01 of the best: it is the best trade-off between quality and cost. Save in `results/tuning_toto_samples.csv`.
- **3.9** Comparison at equal false alarms. On the validation set, choose the `k` of the baseline method whose number of false alarms per series is closest to that of tuned Toto (3.7 and 3.8). If no `k` of the sweep is close, refine the grid with a step of 0.5.
- **3.10** Final results, on the **test** set only, for three configurations:
  - baseline method with k = 3 (default setting);
  - baseline method with the `k` of 3.9;
  - tuned Toto (level of 3.7, samples of 3.8).

  Save in `results/eval_test.csv` and print:

  | Detector | Recall | Precision | F1 | False alarms / series | ms / series |
  |---|---|---|---|---|---|

- **3.11** Figure `results/recall_vs_false_alarms.png`, on the validation set: recall as a function of false alarms per series, with one curve for the baseline method (one point per `k`) and one for Toto (one point per interval level). Mark the chosen setting of each method. At an equal number of false alarms, the highest curve corresponds to the best method.

## Step 4: API (`src/api.py`, `src/worker.py`)

- **4.1** Write `docs/api_spec.md`. The spec must cover:
  - the endpoints, versioned under `/v1` and named with nouns rather than verbs (for example `/v1/detection-jobs`): job creation, job reading, `/health`;
  - the JSON sent and received for each endpoint;
  - the status codes: `202` on creation, `200` on reading, `400` if a series is invalid or too short, `404` if the job does not exist, `429` if the queue is full;
  - the statuses of a job: `pending`, `running`, `done`, `failed`;
  - a single error format, for example `{"error": "...", "message": "..."}`.
- **4.2** Implement the API following `docs/api_spec.md` **exactly**: FastAPI, Pydantic models, job storage in an in-memory dictionary, `asyncio.Queue` with a maximum size. Load Toto only once, at startup.
- **4.3** Implement a worker without batching: it takes a job from the queue, runs the detection, writes the result to the storage. The model call must go through `asyncio.to_thread` so as not to block the API during the computation.
- **4.4** `client.py` script: sends a series, gets the `job_id`, polls the job until `done`, prints the anomalies.

## Step 5: Batching and load test

- **5.1** Add batching to the worker: it takes up to `BATCH_SIZE` jobs, or whatever arrived after `MAX_WAIT_MS`, and calls `predict_interval_batch` only once. Both parameters are read from environment variables.
- **5.2** Write `loadtest/locustfile.py`: each simulated user creates a job then polls its status until `done`. Also measure the end-to-end latency (from creation to the `done` status) as a custom Locust metric.
- **Watch the number of samples.** Use the number of samples chosen in 3.8. `TotoDetector` only processes `max_samples_per_call // num_samples` series per call (2 series with 256 samples). For a batch of 16 jobs to really be processed in a single call, `16 × num_samples ≤ max_samples_per_call` is needed: adjust one or the other and write this choice down in the README.
- **5.3** Run the test with `BATCH_SIZE=1` then `BATCH_SIZE=16`, same number of users. Save in `results/loadtest.csv`:

| Batch | Series / s | p95 latency (end-to-end) |
|---|---|---|

## Step 6: README

- **6.1** Write `README.md`:
  - the problem in 3 lines (incident → spike on a metric → it must be caught quickly, without false alarms);
  - the architecture diagram (client → API → queue → worker → model, result storage);
  - how to run the project;
  - the results: the final table on the test set (`results/eval_test.csv`), the recall / false alarms figure, the load test table and a few series figures;
  - the evaluation method: recall, precision, F1, settings chosen on the validation set and results reported on the test set;
  - the conclusion: comparison at equal false alarms, quality / cost trade-off between Toto and the baseline method, and the cascade idea (the baseline method filters all series, Toto only processes the suspicious ones);
  - the limitations: simulated incidents on real series, and the two weaknesses of the baseline method (flat interval, zero MAD case);
  - an "At Datadog scale" paragraph: Kafka instead of the in-memory queue, partitioning of the series by identifier, GPU inference with large batches, stateless workers replicated horizontally, persistent storage (Redis or a database) instead of the dictionary.
