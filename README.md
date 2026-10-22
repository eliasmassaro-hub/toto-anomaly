# Incident detection on metrics with Toto

An incident (buggy deployment, slow database query, traffic surge) often shows up as a **spike on a metric**: error rate, latency, request count.
It has to be caught **within minutes**, before customers notice, **without false alarms**: with too many useless alerts, on-call engineers end up ignoring them.
This project tests [Toto](https://github.com/DataDog/toto), Datadog's open-source forecasting model, for this use case. Toto forecasts an interval for the next 64 points, and a point that falls outside it is an anomaly. Toto is compared to a baseline method, then served behind an asynchronous API with batching and load tested.

## Architecture

```
client (src/client.py, Locust)
  │   POST /v1/detection-jobs        ──▶ 202 {"job_id"}   (400 if invalid, 429 if the queue is full)
  │   GET  /v1/detection-jobs/{id}   ──▶ 200 {"status", "results"}   (polling until done)
  ▼
FastAPI API (src/api.py) ─────── reads ─────▶ job storage (in-memory dictionary)
  │ put_nowait(job_id)                                  ▲
  ▼                                                     │ writes the status and results
asyncio.Queue (100 jobs at most)                        │
  │ up to BATCH_SIZE jobs, or those that arrived within MAX_WAIT_MS
  ▼                                                     │
worker (src/worker.py) ─────────────────────────────────┘
  │ one model call per batch, in asyncio.to_thread (the API stays responsive)
  ▼
Toto or baseline (src/detectors.py), loaded only once at startup
```

The API contract (endpoints, JSON, status codes, error format) is in [`docs/api_spec.md`](docs/api_spec.md).

## Running the project

Setup (Python 3.12):

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/data.py           # downloads 200 BOOM series to data/series.npz
python src/check_toto.py     # checks that Toto loads and forecasts
```

Evaluation outside the API:

```
python src/evaluate.py                  # raw comparison, results/eval.csv (~10 min on CPU)
python src/tuning.py                    # tuning on the validation set, results on the test set (~15 min)
python src/plot_detectors.py --index 42 # figure of one series, results/series_42_detectors.png
```

API, in two terminals:

```
uvicorn api:app --app-dir src
python src/client.py --index 42 --method toto
```

The interactive documentation is at `http://localhost:8000/docs`. The environment variables `BATCH_SIZE` (16), `MAX_WAIT_MS` (50) and `TOTO_SAMPLES` (256) configure the worker. For the load test, see [below](#load-test).

| File | Role |
|---|---|
| `src/data.py` | BOOM loading, history / future split, spike injection |
| `src/detectors.py` | baseline and Toto, with the same `predict_interval(_batch)` interface |
| `src/evaluate.py`, `src/tuning.py` | metrics, tuning on the validation set, results on the test set |
| `src/api.py`, `src/schemas.py`, `src/worker.py` | API, request validation, worker with batching |
| `src/client.py`, `loadtest/locustfile.py` | demo client, load test |

## Evaluation method

**Data.** 200 real series from the [BOOM](https://huggingface.co/datasets/Datadog/BOOM) dataset (Datadog observability metrics), with a single variable per series. Each series is split into a history of 1024 points and a future of 64 points. Incidents are simulated: 3 spikes are added to the future, of height 5 × the standard deviation of the history, at random positions (seed `i` for series `i`).

**Detectors.**
- **Baseline**: median ± k × 1.4826 × MAD of the last 200 points, i.e. a constant interval over the 64 points.
- **Toto** (`Datadog/Toto-Open-Base-1.0`): quantiles of a given level over `num_samples` forecast trajectories.

A future point outside the interval is flagged.

**Metrics.** A flagged point is a true positive (TP) if it is an injected spike, and a false alarm (FP) otherwise.

| Metric | Definition |
|---|---|
| Recall | TP / injected spikes |
| Precision | TP / (TP + FP), on the totals over all series |
| F1 | 2 × precision × recall / (precision + recall) |
| False alarms / series | FP / number of series |
| ms / series | mean computation time, after a warm-up call |

**Validation and test.** The 200 series are split into two random halves of 100, with a fixed seed. Settings are chosen **only on the validation set**, and results are reported **only on the test set**:
- **Toto interval level**: 99%, which maximizes the validation F1 (0.626, against 0.616 at 98% and 0.618 at 99.5%);
- **number of samples**: 256, the smallest whose F1 is within 0.01 of the best (128 samples is 0.02 below);
- **baseline `k`**: 18.5, chosen to get as many false alarms as Toto on the validation set (2.62 per series).

## Results

### Detection quality, on the test set (100 series, 300 spikes)

| Detector | Recall | Precision | F1 | False alarms / series | ms / series |
|---|---|---|---|---|---|
| Baseline, k = 3 (default setting) | 98.0% | 20.9% | 0.344 | 11.2 | 0.03 |
| Baseline, k = 18.5 (equal false alarms on the validation set) | 44.7% | 44.5% | 0.446 | 1.7 | 0.03 |
| Tuned Toto (99% level, 256 samples) | **85.3%** | **46.5%** | **0.602** | 3.0 | 2269 |

Source: [`results/eval_test.csv`](results/eval_test.csv). Since `k` is tuned on the validation set, the baseline makes slightly fewer false alarms than Toto on the test set (1.7 against 3.0 per series).

### Recall as a function of false alarms (validation)

![Recall as a function of false alarms per series](results/recall_vs_false_alarms.png)

Each point is a setting: a value of `k` for the baseline, an interval level for Toto. At an equal number of false alarms, the highest curve is the best. Toto's curve is above over the whole common range. At the chosen setting (2.62 false alarms per series), **Toto detects 82% of the spikes, against 41% for the baseline**.

### Example series

These figures use the default settings (k = 3, 256 samples), not the tuned `k`.

**Cyclic series** (`ds-277-T`). Toto's interval follows the cycle. The baseline interval has to cover the whole cycle, and only detects the spikes here because they go above the top of the cycle.

![Series ds-277-T](results/series_0_detectors.png)

**Real rise in the future** (`ds-1493-10S`). The baseline flags the whole rise (36 false alarms). Toto's interval widens when the rise happens, which gives 0 false alarms, but it misses one of the three spikes.

![Series ds-1493-10S](results/series_42_detectors.png)

**A case where Toto fails** (`ds-1419-10S`). Toto forecasts a very wide interval and misses all three spikes, which the baseline detects.

![Series ds-1419-10S](results/series_150_detectors.png)

More series are in [`results/`](results/).

### Load test

**Choosing the number of samples.** The worker groups up to `BATCH_SIZE` jobs and sends them to Toto in a single call. Toto duplicates each series `num_samples` times in its internal batch. For a batch of 16 jobs to really go through in a single call, `16 × num_samples ≤ max_samples_per_call` is therefore needed.

With 256 samples, that would be 4096 rows per call: on CPU (16 GB), the machine swaps and the call becomes about 10 times slower. With `max_samples_per_call = 512`, a batch of 16 series is split into 8 calls of 2 series, and batching no longer helps. Measured on 16 series, in steady state:

| Samples | One series per call | 16 series at once | Calls for 16 series |
|---|---|---|---|
| 32 | 440 ms / series | 318 ms / series | 1 |
| 256 | 2865 ms / series | 2956 ms / series | 8 |

**The load test therefore uses `TOTO_SAMPLES=32`.** That gives 16 × 32 = 512 samples, i.e. a single call per batch. The price is quality: on the validation set, F1 drops from 0.61 to 0.48 (`results/tuning_toto_samples.csv`). By default, the API keeps 256 samples. On GPU, we would keep 256 samples and raise `max_samples_per_call` to 4096.

**Running the test.** Terminal 1, the server (run again with `BATCH_SIZE=16`):

```
BATCH_SIZE=1 TOTO_SAMPLES=32 uvicorn api:app --app-dir src --no-access-log
```

Terminal 2, Locust with 32 users for 3 minutes:

```
locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 32 -r 32 -t 3m --csv loadtest/out/batch1
```

Once both tests are done, `python loadtest/summarize.py` writes [`results/loadtest.csv`](results/loadtest.csv).

**Results.** Toto (32 samples, 99% level) on CPU (MacBook Air, 16 GB). Each job contains one series, and each user sends jobs back to back without pausing.

| Batch | Series / s | p50 latency (end-to-end) | p95 latency (end-to-end) | Completed jobs |
|---|---|---|---|---|
| 1 | 2.67 | 12 s | 14 s | 478 |
| 16 | 3.04 | 10 s | 12 s | 544 |

- **Throughput: +14%.** With `BATCH_SIZE=16`, all 35 batches were full. The gain stays modest, because on CPU the model is already almost saturated by the samples of each series. On GPU, a bigger batch would use otherwise idle cores, and the gain would be much larger.
- **Latency: −2 s.** Here, latency is mostly waiting in the queue. With 32 users, Little's law gives latency ≈ users / throughput, i.e. 32 / 2.67 ≈ 12 s and 32 / 3.04 ≈ 10.5 s, which matches the measurements. Batching reduces latency because it increases throughput. Under low load, it would be the opposite: a job would wait up to `MAX_WAIT_MS`, then for its whole batch to finish.
- **No failures**, no `429`: the queue (100 jobs) was never full with 32 users.

## Conclusion

- **At equal false alarms, Toto is clearly better.** On the validation set, it detects 82% of the spikes against 41% for the baseline. On the test set, its F1 is 0.60, against 0.45 for the tuned baseline and 0.34 for the default baseline. The difference comes from the interval: Toto's follows the cycles and the level of the series, whereas the baseline's is flat.
- **But Toto is much more expensive.** It takes about 2.3 s per series on CPU with 256 samples, against 0.03 ms for the baseline, i.e. roughly 70,000 times more. Fewer samples reduce the cost but also the quality: with 32 samples, the computation is 7 times faster, but the validation F1 drops from 0.61 to 0.48. Batching only recovers 14% on CPU.
- **A cascade would combine both.** The baseline, with a low `k` (k = 3 detects 98% of the spikes), would filter all series at every point in time. Toto would only process the suspicious series, to remove their false alarms. Toto's cost would then be proportional to the share of suspicious series, which is small in production, where most series are normal most of the time. In this benchmark, on the contrary, every series contains spikes: the cascade would not save anything here, it would need to be measured on more realistic data. Its recall would also be capped by that of the filter.

## Limitations

- **Simulated incidents.** The spikes are injected into real series. They cover neither level shifts, nor slow drifts, nor drops to zero. A height of 5 standard deviations is a fairly easy case.
- **Unlabeled real anomalies.** A real anomaly in the future counts as a false alarm, like the rise in `ds-1493-10S`.
- **The two weaknesses of the baseline, kept on purpose:**
  - **Flat interval**: it does not follow cycles, can miss a spike placed in a trough, and flags regular jumps as anomalies.
  - **Zero MAD**: on an almost constant series (for example `ds-1525-5T`), the MAD is 0 and the interval would have zero width. It is then replaced by the standard deviation of the window.
- **Toto is random.** Its samples change from one run to the next, and so do the flagged points sometimes. It can also forecast a very wide interval and detect nothing (`ds-1419-10S`).
- **100 test series.** Differences of a few points of recall or precision are not significant.
- **Demo API.** In-memory jobs, a single worker, no authentication (see the limitations in [`docs/api_spec.md`](docs/api_spec.md)). The load test uses 32 samples, not 256.

## At Datadog scale

**Queue and storage.** The in-memory queue would be replaced by **Kafka**, which keeps the points even if a worker goes down and allows a stream to be replayed. The series would be **partitioned by identifier**, so that all the points of a series arrive in order at the same worker, which can keep its history at hand. The job dictionary would become **persistent storage** (Redis with a time to live, or a database), shared by all the API instances.

**Inference and workers.** Inference would run on **GPU with large batches**. This is where batching pays off: 256 samples × dozens of series per call, where the CPU is already saturated. **Workers would be stateless and replicated horizontally** (a Kafka consumer group), with more added automatically when the queue lag grows.

**Cost.** The cascade would reserve the GPU for the series flagged as suspicious by the baseline.
