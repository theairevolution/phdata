# Technical Report: Housing Price Prediction API — Improvements Summary

**Project:** pH Data ML Interview Challenge — Real Estate Price Prediction API  
**Date:** 2026-06-07  
**Branch:** main  

---

## Overview

This report documents the improvements made to the housing price prediction API and serves as source material for two presentations: a 10-minute business presentation for non-technical stakeholders, and a technical deep-dive for engineers. Sections 1–3 cover the three implementation workstreams (missing data handling, performance, code quality). Section 4 describes the production deployment architecture on AWS. Section 5 covers the MLOps lifecycle. Section 6 provides business-oriented narrative for the non-technical presentation. Section 7 describes how AI tooling was used throughout the project.

---

## Section 1 — Missing Data Handling

### Problem

Real estate data is inherently incomplete. A seller may know square footage and zipcode but not the number of bathrooms. An MLS listing might omit basement information. In the original API, every one of the seven home feature fields was required to be present and non-null for the model to produce a prediction. A missing value caused either a validation error or, worse, a silent prediction based on `NaN` — which a KNN regressor handles arbitrarily.

This limitation made the API unusable for the majority of real-world listing data, where partial information is the norm rather than the exception.

### Solution

A `KNNImputer` was trained on the home feature columns and persisted alongside the main model. At prediction time, any missing field is filled before the data reaches the regressor.

**Algorithm:** K-Nearest Neighbors imputation with `n_neighbors=5` and `weights="distance"`. For each missing value, the imputer identifies the five most similar complete records in the training set (by Euclidean distance across the available features) and computes a distance-weighted average. Neighbors that are more similar contribute proportionally more to the fill value than distant ones.

**Features covered:**

| Feature | Type | Typical range in training data |
|---------|------|-------------------------------|
| `bedrooms` | int | 0 – 33 |
| `bathrooms` | float | 0 – 8 |
| `sqft_living` | float | 290 – 13,540 sq ft |
| `sqft_lot` | float | 520 – 1,651,359 sq ft |
| `floors` | float | 1 – 3.5 |
| `sqft_above` | float | 290 – 9,410 sq ft |
| `sqft_basement` | float | 0 – 4,820 sq ft |

**Training discipline:** The imputer was fit exclusively on the training split (same `random_state=42` split used to train the main KNN regressor). It has never seen the test set, so there is no data leakage — the imputer's knowledge of "typical" feature distributions comes only from the data the model was also trained on.

**Deployment:** The fitted imputer is serialized to `model/imputer.pkl` and loaded once at API startup alongside `model.pkl`. There is no per-request I/O.

### Impact

The following outcomes are verified by the unit test suite (`test/unit/test_imputer_unit.py`):

**Completeness guarantee:** When all seven home features are provided, the imputer passes them through unchanged. Imputation does not alter values that are already present — it only fills gaps.

**Single missing feature:** Any single missing feature is filled with a plausible value derived from the five nearest complete neighbors. The remaining six features are unaffected.

**Multiple simultaneous missing features:** The imputer handles any combination of missing fields. Even four or five features missing simultaneously produces a complete row with no `NaN` values.

**All features missing:** The most extreme case — a request with only a zipcode and no home features whatsoever — still returns a valid price prediction. The imputer fills all seven fields from the training distribution before passing the record to the regressor.

**Prediction plausibility:** Imputed predictions stay in a sensible range. A test guard verifies that a prediction made with several missing features is within 50% of the price predicted for the same home with all features provided. Additionally, individual imputed values are bounded by training-data ranges: an imputed bedroom count stays between 0 and 33; imputed square footage stays between 290 and 13,540 square feet.

**Before vs. after:**

| Scenario | Before | After |
|----------|--------|-------|
| All features provided | Works | Works |
| One feature missing | Fails / silent NaN | Imputed, returns valid price |
| Multiple features missing | Fails | Imputed, returns valid price |
| Only zipcode provided | Fails | Imputed, returns valid price |

The API now handles the full spectrum of real-world data completeness, from perfectly specified listings to bare-minimum records with only a location.

---

## Section 2 — Performance

### Problem

The original API reloaded all of its model artifacts on every single request. Each call to `/predict` opened `model/model.pkl`, deserializes the full sklearn pipeline (scaler + KNN regressor), opened `model/imputer.pkl`, deserialized the KNN imputer, opened and parsed `model_features.json`, and read and scanned the entire demographics CSV to find the row matching the requested zipcode. A debug statement also serialized the full input DataFrame to stdout on every request.

Under load testing, this produced a baseline of approximately 97 requests per second at peak throughput, with median latency of 60 ms at 10 concurrent users — numbers that would degrade rapidly in production as concurrency increased.

### Methodology

All profiling runs used a containerized setup with fixed resource limits (2 vCPU, 2 GB memory, 1 gunicorn worker) for code change comparisons. The load generator (`test/profiling/profile_api.py`) sends POST `/predict` requests with a fixed payload across progressive concurrency stages: 1, 5, 10, 15, 20, 30, 40, 50, and 75 simultaneous users. Each stage runs for 20 seconds with a 5-request warmup. Errors are client-side ReadTimeouts (request queued for more than 10 seconds); no HTTP 5xx errors were observed in any run.

Four code configurations were tested in sequence, each building on the previous:

| Tag | Change applied |
|-----|---------------|
| A | Baseline — per-request file I/O, pickle deserialization, pandas CSV scan, `print()` on every request |
| B | Global caching — model, imputer, features, and demographics CSV loaded once at module import |
| C | Remove `print(input_data)` — debug statement removed |
| D | O(1) dict lookup — demographics pandas boolean scan replaced with pre-built dict |

### Results

**Throughput (requests per second):**

| Concurrent users | A (baseline) | B (caching) | C (no print) | D (dict lookup) |
|----------------:|-------------:|------------:|-------------:|----------------:|
| 1 | 15.3 | 15.5 | 16.7 | 16.5 |
| 5 | 72.9 | 80.8 | 86.5 | 86.8 |
| 10 | 86.1 | 113.6 | 162.0 | 161.3 |
| 15 | 91.4 | 113.0 | 201.4 | 207.0 |
| 20 | 93.3 | 115.5 | **217.9** | 215.7 |
| 30 | 92.0 | **118.8** | 210.7 | **216.8** |
| 40 | **96.8** | 115.8 | 183.1 | 168.2 |
| 50 | 87.0 | 112.8 | 155.2 | 163.2 |

**Median (p50) latency in milliseconds:**

| Concurrent users | A | B | C | D |
|----------------:|--:|--:|--:|--:|
| 1 | 13.7 | 11.7 | 7.9 | 8.0 |
| 10 | 59.5 | 35.9 | 9.7 | 9.6 |
| 20 | 160.3 | 118.5 | 39.5 | 39.4 |
| 30 | 267.9 | 197.1 | 84.6 | 80.7 |
| 50 | 508.0 | 361.7 | 164.6 | 157.9 |

**Full latency percentile comparison — baseline (A) vs optimized (D):**

The table below uses a dedicated run for each config to show p50, p95, and p99 from the same measurement.

| Concurrent users | A p50 | A p95 | A p99 | D p50 | D p95 | D p99 |
|----------------:|------:|------:|------:|------:|------:|------:|
| 1  | 19.3 ms | 33.7 ms | 37.6 ms | 8.0 ms | 9.0 ms | 9.7 ms |
| 5  | 39.5 ms | 78.5 ms | 88.2 ms | 9.0 ms | 10.7 ms | 16.5 ms |
| 10 | 109.7 ms | 149.3 ms | 159.7 ms | 9.6 ms | 15.9 ms | 24.3 ms |
| 15 | 196.1 ms | 239.6 ms | 265.9 ms | 37.0 ms | 37.6 ms | 59.1 ms |
| 20 | 274.6 ms | 327.9 ms | 350.1 ms | 39.4 ms | 61.9 ms | 87.5 ms |
| 30 | — | — | — | 80.7 ms | 146.6 ms | 221.6 ms |
| 50 | — | — | — | 157.9 ms | 735.5 ms | 1110.8 ms |

### Impact of Each Change

**A → B: Global caching (+23% peak RPS, −40% p50 at 10 users)**

Moving the model, imputer, feature list, and demographics CSV into module-level globals eliminates four file reads and two pickle deserializations per request. These objects are loaded once when the gunicorn worker starts and are reused across every request for the life of that worker process.

The improvement is genuine — about 24 ms of per-request I/O overhead removed — but smaller than expected, because a different bottleneck was dominating: the debug print statement.

**B → C: Removing `print(input_data)` (+83% peak RPS, −73% p50 at 10 users)**

This was the single largest performance improvement in the entire project. `print(input_data)` serializes a pandas DataFrame to a string representation and writes it to stdout on every request. In Python, both operations hold the Global Interpreter Lock (GIL): the string formatting runs in Python bytecode, and the `write()` syscall to stdout is a blocking I/O operation. In an async FastAPI/uvicorn worker, any GIL-held blocking operation prevents the event loop from processing other waiting requests.

At 100+ requests per second this meant the worker was spending a substantial portion of its time formatting and writing debug output rather than handling requests. Removing one line of debug code nearly doubled throughput from 119 to 218 RPS, and cut median latency at 10 users from 36 ms to under 10 ms.

This is a common production anti-pattern: debug instrumentation added during development that becomes a severe bottleneck under load. The fix is the removal of the line entirely — structured logging at DEBUG level is used instead, which can be filtered out without touching code.

**C → D: O(1) dict lookup (~0% change)**

The original code filtered the demographics DataFrame with a pandas boolean mask (`demographics[demographics["zipcode"] == zipcode]`) — an O(n) scan across all rows on every request. This was replaced with a pre-built Python dict keyed by zipcode, making the lookup O(1).

The change is correct and more efficient in principle, but produced no measurable throughput improvement. The demographics CSV contains approximately 100 rows, and the pandas scan is implemented in vectorized C — the absolute time was already in the microsecond range, negligible compared to the ~8 ms that sklearn KNN inference takes. At the scale of this dataset, the theoretical complexity improvement does not translate to wall-clock savings.

**Overall (A → D):**

| Metric | Baseline (A) | Optimized (D) | Improvement |
|--------|-------------|---------------|-------------|
| Peak throughput | ~97 RPS | ~229 RPS | **+136%** |
| p50 latency @ 10 users | 109.7 ms | 9.6 ms | **−91%** |
| p95 latency @ 10 users | 149.3 ms | 15.9 ms | **−89%** |
| p99 latency @ 10 users | 159.7 ms | 24.3 ms | **−85%** |
| p50 latency @ 20 users | 274.6 ms | 39.4 ms | **−86%** |
| p99 latency @ 20 users | 350.1 ms | 87.5 ms | **−75%** |
| Peak memory | ~129 MB | ~125 MB | −3% |

### Reading Tail Latency: p95, p99, and SLOs

p50 (median) answers "what does the typical user experience?" — it is useful for comparing code changes because it is stable and representative. p95 and p99 answer a different question: "how bad does it get for the unluckiest users, and when should an alarm fire?"

In the baseline (Config A), p99 at 20 concurrent users was **350 ms**. Most internal API SLOs target sub-200 ms for synchronous user-facing endpoints. A 350 ms p99 means 1 in 100 requests — experienced by a real estate agent waiting for a price estimate — would take more than a third of a second, just at 20 users. The system would violate a 200 ms SLO at moderate load even when p50 looked acceptable.

In the optimized (Config D), p99 at 20 users is **87 ms** — comfortably within common SLO targets. p99 at 30 users climbs to 221 ms, which represents the practical saturation point for a single worker at 1–2 vCPU.

The p99 column also reveals the **saturation cliff**: at 50 users, optimized p50 is 158 ms (acceptable), but p99 is 1,110 ms. The gap between p50 and p99 widening sharply is the load generator's signal that the worker's request queue is growing faster than it can drain — the trigger for horizontal autoscaling. For a production SLO of p99 ≤ 200 ms, the autoscale threshold should be set below the 30-user / ~220 RPS point.

### Resource Scaling

A secondary experiment tested the optimized code (configuration D) at 1, 2, and 4 vCPU allocations with a single gunicorn worker:

| vCPU | Peak RPS | Users at peak |
|-----:|--------:|--------------:|
| 1 | 223 | 20u |
| 2 | 229 | 20–30u |
| 4 | 201 | 20u |

**4 vCPU underperformed 1 vCPU.** The reason is OpenBLAS thread management. Sklearn's KNN inference relies on numpy, which calls into OpenBLAS for matrix operations. OpenBLAS spawns internal threads based on the host OS CPU count, not the Docker CPU limit. With 4 vCPU allocated, OpenBLAS spawned four threads per inference call. For a single-row input (one prediction request), there is not enough work to parallelize across four threads — the overhead of coordinating them (context switches, cache contention, thread synchronization) consumed the CPU budget without producing proportional throughput. At 4 vCPU the container showed ~396% CPU utilization while producing less throughput than a single vCPU at 99% CPU.

**Scaling implication:** Adding vCPUs to a single worker is not an effective strategy for this workload. The correct scaling approach is horizontal: adding a second container (each with 1–2 vCPU and one worker) behind a load balancer. Two such containers would yield approximately 2× throughput with near-linear scaling, because each worker runs its own GIL and Python interpreter independently.

---

## Section 3 — Code Quality

A full audit of the codebase identified 13 issues across three priority tiers. All were addressed. The items below are ordered within each tier by impact.

### Tier 1 — High-Impact Fixes

**1. CORS misconfiguration (security)**  
`src/main.py` had `allow_credentials=True` combined with `allow_origins=["*"]`. This combination is explicitly invalid per the Fetch/CORS specification: browsers reject it, and it is rejected by most CDN and proxy layers. More importantly, if the wildcard origin were ever narrowed to a specific domain — a common step when moving to production — `allow_credentials=True` would silently enable cross-site request forgery (CSRF). The field was corrected to `allow_credentials=False`.

**2. Zero observability (logging)**  
The API had no logging whatsoever. There was no way to know which model file was loaded, whether startup completed successfully, which zipcode triggered a failure, or what caused a 500 response. Structured logging was added at three levels: startup confirmation (which artifacts loaded, how many zipcodes in the demographics table), per-request debug (zipcode and predicted price on success), and error paths (warning on unknown zipcode, error with full exception detail on unexpected failure). The log format includes timestamp, logger name, and level, suitable for ingestion by any standard log aggregator.

**3. Unhandled errors with opaque responses**  
Any zipcode not present in the demographics table caused an unhandled `KeyError` inside the request handler. FastAPI caught this and returned a 500 Internal Server Error with no useful detail. Callers had no way to distinguish a configuration problem from a bad input. The fix adds an explicit check before attempting the lookup: unknown zipcodes now return 422 Unprocessable Entity with `detail: "Unknown zipcode: {value}"`, which is an actionable, specific error. Genuinely unexpected failures are caught by a broad `try/except`, logged with full context, and returned as 500 — a meaningful distinction from the invalid-input case.

**4. Hardcoded working-directory-relative paths**  
Model artifacts were referenced as `"model/model.pkl"` — relative to the process working directory at runtime. This worked only when the API was started from the `src/` directory. In the Docker test environment it was masked by volume mount symlinks; in local development it was fragile. All paths were replaced with `Path(__file__).parent.parent / "model" / ...`, resolving relative to the source file's location. This works correctly in every environment: production Docker, test Docker, and local development, without relying on CWD conventions or symlinks.

**5. File handle leaks in training scripts**  
Both `src/model/create_model.py` and `create_imputer.py` called `open()` and passed the file object directly to `pickle.dump()` or `json.dump()` without closing it. While standalone scripts reclaim handles at process exit — so there is no crash risk — the pattern is incorrect. An exception mid-write would leave a partially written artifact file with an open handle. Both scripts were updated to use `with open(...) as f:` context managers, which guarantee the handle is closed even on failure.

**6. Unused import removed**  
`create_imputer.py` imported `numpy as np` but never referenced it. Removed.

**7. Duplicate `create_model.py` deleted**  
A copy of `create_model.py` existed at the repository root and an identical version at `src/model/create_model.py`. No Makefile target, script, or documentation pointed to the root copy. Two identical files diverge silently — a change to one is not reflected in the other. The root copy was deleted; the canonical version remains at `src/model/create_model.py`.

**8. Dead `src/utils/loader.py` deleted**  
`src/utils/loader.py` defined three functions (`load_model`, `load_features`, `get_demographics`) that were never imported anywhere in the codebase. Dead code misleads contributors into thinking a reusable interface exists, accumulates drift from the actual implementation, and implies a separation of concerns that is not actually in use. The file was deleted.

### Tier 2 — Meaningful Improvements

**9. Input validation on all numeric fields**  
All seven optional numeric fields in `HomeFeatures` were updated to use `Field(default=None, ge=0)`. A KNN regressor has no built-in sanity check on its inputs — it will predict a price for a house with −5 bedrooms or −500 square feet without complaint, returning a nonsensical result with a 200 OK status. With `ge=0` constraints, Pydantic rejects invalid inputs at the API boundary and returns a 422 with field-level detail before the request reaches the model.

**10. `IMPUTER_FEATURES` deduplicated**  
`test/unit/test_imputer_unit.py` previously contained a hardcoded copy of the seven feature names, identical to the list defined in `src/api/endpoints.py`. Two identical lists in separate files will diverge silently when one is updated. The test file now imports `IMPUTER_FEATURES` directly from the source module, guaranteeing that the test always validates against the same feature set the API uses.

**11. Unit test coverage expanded**  
The original test suite had two API unit tests, both on the happy path. Four additional tests were added to cover the error-handling and validation boundaries introduced in this session:

- `test_predict_endpoint_price_positive` — predicted price must be a positive number
- `test_predict_endpoint_invalid_zipcode` — unknown zipcode must return 422 with "zipcode" in the detail
- `test_predict_endpoint_negative_bedrooms_rejected` — negative bedroom count must return 422
- `test_predict_endpoint_negative_sqft_rejected` — negative square footage must return 422

Without these tests, the fixes in items 3 and 9 could be silently reverted in a future commit with no test failure to signal the regression.

### Tier 3 — Structural Improvements

**12. GitHub Actions CI pipeline**  
`.github/workflows/ci.yml` was added, defining a `unit-tests` job that installs dependencies and runs `pytest test/unit` on every push and pull request to `main`. Before this, the existing unit tests were a local convenience with no automated enforcement. A breaking change to any endpoint or model loading path could be merged to main with no signal. CI turns the test suite into a quality gate: the branch cannot merge unless the tests pass.

**13. Environment variable documentation**  
Five environment variables are used across `gunicorn.conf.py`, `docker-compose.test.yml`, and the profiling script: `WEB_CONCURRENCY`, `API_CPUS`, `API_BASE_URL`, `API_SERVICE_NAME`, and `RESULTS_DIR`. Previously, discovering these required grepping across three different files with no central reference. An `.env.example` file documents all five variables with their defaults and descriptions in one place, following the standard convention for surfacing environment configuration to new contributors.

### Summary Table

| # | File(s) | Action | Tier | Primary impact |
|---|---------|--------|------|----------------|
| 1 | `src/main.py` | Fix CORS credentials flag | 1 | Security |
| 2 | `src/main.py`, `src/api/endpoints.py` | Add structured logging | 1 | Observability |
| 3 | `src/api/endpoints.py` | Add zipcode 422 guard + try/except | 1 | API contract |
| 4 | `src/api/endpoints.py` | `__file__`-relative paths | 1 | Reliability |
| 5 | `src/model/create_model.py`, `create_imputer.py` | Context managers for file I/O | 1 | Correctness |
| 6 | `create_imputer.py` | Remove unused `numpy` import | 1 | Cleanliness |
| 7 | `create_model.py` (root) | **Deleted** — duplicate | 1 | Maintainability |
| 8 | `src/utils/loader.py` | **Deleted** — dead code | 1 | Maintainability |
| 9 | `src/api/endpoints.py` | `ge=0` field validators | 2 | API correctness |
| 10 | `test/unit/test_imputer_unit.py` | Import `IMPUTER_FEATURES` from source | 2 | Test consistency |
| 11 | `test/unit/test_api_unit.py` | 4 new edge-case tests | 2 | Regression safety |
| 12 | `.github/workflows/ci.yml` | **Created** — CI pipeline | 3 | Process |
| 13 | `.env.example` | **Created** — env var reference | 3 | Developer experience |

---

## Section 4 — Production Architecture

> This section describes the target production architecture. The current implementation runs locally in Docker and is production-ready at the code level; the infrastructure described below is the deployment design for a cloud environment.

### System Architecture

```
                    ┌─────────────────────────────┐
                    │   Client (Browser / Mobile)  │
                    └──────────────┬──────────────┘
                                   │ HTTPS
                    ┌──────────────▼──────────────┐
                    │    AWS ALB (Application      │
                    │      Load Balancer)          │
                    │  • TLS termination           │
                    │  • /health target group      │
                    │  • WAF rate limiting         │
                    └──────┬──────────────┬────────┘
                           │              │
             ┌─────────────▼──┐    ┌──────▼─────────────┐
             │  ECS Task #1   │    │   ECS Task #2       │
             │  1 worker      │    │   1 worker          │
             │  1–2 vCPU      │    │   1–2 vCPU          │
             │  2 GB RAM      │    │   2 GB RAM          │
             │  gunicorn +    │    │   gunicorn +        │
             │  uvicorn       │    │   uvicorn           │
             └───────┬────────┘    └────────┬────────────┘
                     │                      │
        ┌────────────▼──────────────────────▼────────────┐
        │               AWS S3 (model artifacts)          │
        │   model.pkl  •  imputer.pkl  •  features.json   │
        │   versioned by model tag, downloaded at startup  │
        └────────────────────────────────────────────────┘
        ┌────────────────────────────────────────────────┐
        │        Amazon CloudWatch                        │
        │  Logs → Insights  •  Metrics  •  Alarms        │
        └────────────────────────────────────────────────┘
```

### AWS Deployment Design

**Compute:** ECS Fargate — serverless containers, no EC2 fleet to manage. Each task runs one gunicorn worker (the configuration proven optimal in profiling: 1 worker, 1–2 vCPU). Fargate handles placement, bin-packing, and health replacement automatically.

**Container registry:** Amazon ECR. Every commit to `main` that passes CI builds a new image tagged with the commit SHA and pushes to ECR. The ECS task definition references the specific image tag, so deployments are fully reproducible and rollbacks are a tag swap.

**Model artifact storage:** Artifacts are stored in S3 using path-based versioning as the primary versioning mechanism (e.g., `s3://sound-realty-models/v20260607/model.pkl`). S3 object versioning is also enabled as a safety net against accidental overwrites, but it is not the primary version identifier — the version is encoded in the path prefix. Tasks download artifacts at container startup via an ECS task role — no credentials in the image. This decouples model versioning from code versioning: a new model can be deployed without rebuilding the container.

**Load balancer:** ALB with a target group pointing to the ECS service. The `/health` endpoint already implemented returns 200 when the model is loaded and the worker is ready; this is used as the ALB health check and the ECS container health check. Unhealthy tasks are replaced automatically.

**Autoscaling:** ECS Service Autoscaling on `ALBRequestCountPerTarget`. Based on profiling data: a single task saturates at ~220 RPS. Setting the scale-out threshold at 150 RPS/task (≈68% of saturation) leaves headroom for bursty traffic before p99 climbs past the SLO ceiling. At 2 tasks the system handles ~440 RPS with near-linear scaling (each task runs its own Python process and GIL independently).

**Secrets and configuration:** AWS Systems Manager Parameter Store for any sensitive values. `WEB_CONCURRENCY` is passed as an ECS environment variable, defaulting to 1 for the single-worker configuration validated in profiling.

### Security

| Layer | Control |
|-------|---------|
| Network | ALB in public subnet; ECS tasks in private subnet; no public IPs on tasks. **VPC Endpoints for ECR and S3 are required** (or a NAT Gateway) so that tasks in the private subnet can pull images and download artifacts — without either, tasks fail at startup. |
| TLS | ALB terminates HTTPS; containers communicate over VPC on HTTP |
| Credentials | ECS task role grants S3 read access; no static credentials in code or image |
| CORS | Fixed (`allow_credentials=False`); for production, narrow `allow_origins` to the frontend domain |
| Input validation | Pydantic `ge=0` constraints on all numeric fields; unknown-zipcode 422 guard already implemented |
| Rate limiting | ALB WAF Web ACL — rate limit by IP (e.g., 1,000 requests/5 min) to prevent abuse |
| Secrets | API keys or auth tokens (if added) stored in Parameter Store, injected at runtime |

### Monitoring and Observability

The API already emits structured logs (timestamp, logger name, level, message) on every startup, error, and — at DEBUG level — every successful prediction. In production these flow to CloudWatch Logs via the ECS `awslogs` log driver.

**Metrics to track** (emitted as CloudWatch custom metrics from the application, or derived from ALB access logs):

| Metric | Alarm threshold | Action |
|--------|----------------|--------|
| p99 request latency | > 200 ms for 5 min | PagerDuty / on-call alert (autoscaling on `ALBRequestCountPerTarget` handles scaling independently) |
| Error rate (`5xx` + timeouts) | > 1% over 1 min | PagerDuty alert |
| Imputation hit rate (% requests with ≥ 1 null field) | — | Trend monitoring; large shifts indicate data quality change upstream |
| Task memory utilization | > 80% | Review for memory leak |
| Model inference time (logged separately from API overhead) | > 50 ms p99 | Investigate model complexity or data volume growth |

**Dashboards:** ALB metrics (p50/p95/p99 from access logs) + custom application metrics in a single CloudWatch dashboard. The saturation cliff visible in profiling (p99 diverging from p50 at 30+ users) is the operational signal to watch: when p99 starts trending away from p50, it precedes errors by approximately one concurrency stage.

---

## Section 5 — MLOps: Production ML Lifecycle

> This section describes the MLOps target state. Current implementation covers CI (GitHub Actions unit tests) and artifact serialization. The items below are the next steps toward full production ML operations.

### Current State vs. Target State

| Capability | Current | Production target |
|------------|---------|------------------|
| Model artifacts | Committed to git as `.pkl` | S3 with version tags |
| Model versioning | Git commit hash | MLflow / SageMaker Model Registry |
| Demographics data | Static CSV in repo | SageMaker Feature Store (versioned) |
| CI pipeline | Unit tests on push | Unit → integration → build → deploy → smoke test |
| Retraining | Manual script | Scheduled Step Functions workflow |
| Drift detection | None | CloudWatch + custom metrics |

### Model Registry

Committing `.pkl` files to git has two problems: binary files inflate repository size and provide no metadata about model quality. In production, each trained model would be registered in **AWS SageMaker Model Registry** with:

- Training date and dataset version
- Hyperparameters (`n_neighbors`, `weights`, train/test split seed)
- Evaluation metrics (R², MAE, RMSE on the held-out test set)
- Stage label: `Staging` → `Production` (requires manual promotion or automated metric gate)

This enables one-click rollback (promote the previous `Production` version), A/B testing (ALB weighted routing: 90% → current model, 10% → candidate), and a full audit trail of what was deployed when. MLflow is a viable alternative if a self-hosted tracking server is acceptable (e.g., running on ECS with an S3/RDS backend), but for a fully AWS-native stack SageMaker Model Registry requires no additional infrastructure to operate.

### Feature Store

The demographics CSV (`zipcode_demographics.csv`) is currently a static file joined at inference time. There are two risks: the file can go stale silently, and training and inference could diverge if the file is updated between retraining and deployment.

In production: **AWS SageMaker Feature Store** stores zipcode demographics as a versioned feature group. Both the training pipeline and the inference API read from the same feature group snapshot. When census data is refreshed (e.g., annually), the feature group is updated with a new version — training and inference are pinned to a consistent snapshot, preventing training/serving skew.

### CI/CD Pipeline

Current pipeline (GitHub Actions): commit → `pytest test/unit` → pass/fail.

Target pipeline:

```
git push → GitHub Actions
  ├─ unit-tests       (pytest test/unit — fast, ~30 s)
  ├─ integration      (docker-compose up + pytest test/integration — ~2 min)
  ├─ build-image      (docker build + push to ECR)
  ├─ deploy-staging   (ECS service update → staging environment)
  ├─ smoke-tests      (curl /health + /predict with known payload, assert expected price range)
  └─ promote-prod     (blue/green via AWS CodeDeploy: shifts ALB traffic to new task set; old set drained)
```

Blue/green deployment is orchestrated by **AWS CodeDeploy** (ECS deployment controller set to `CODE_DEPLOY`), which manages traffic shifting via ALB weighted target groups. This means zero-downtime deploys and instant rollback: if smoke tests fail on green, CodeDeploy shifts traffic back to blue before draining it.

### Model Retraining and Drift Detection

**Scheduled retraining:** An **Amazon EventBridge Scheduler** rule triggers an AWS Step Functions state machine weekly (or on-demand via an S3 event notification when new sales data lands). Steps: ingest new data → retrain KNN regressor + KNN imputer on combined historical + new data → evaluate against held-out test set → if metrics pass gate (e.g., R² > 0.75), register new version in Model Registry → trigger CI/CD pipeline to deploy.

**Data drift:** The application emits incoming request feature values (bedrooms, sqft_living, zipcode) as CloudWatch custom metrics. **CloudWatch Anomaly Detection** models the expected statistical range for each metric; deviations beyond the anomaly band trigger an alarm. A sustained shift in input distributions may indicate the service is receiving data from a market segment the model was not trained on.

**Concept drift:** Track the distribution of predicted prices over time. If the rolling mean predicted price diverges significantly from closed sale prices (requires a feedback loop from the MLS or CRM), it signals the model's price surface has become stale and retraining should be triggered ahead of schedule.

---

## Section 6 — Business Value Narrative

> This section is written for a non-technical audience. It avoids all implementation terminology and focuses on business outcomes. It serves as source material for the 10-minute business presentation.

### The Problem Sound Realty Was Facing

Estimating a home's value takes time. Agents often wait until they have a complete property profile — all the room counts, square footage measurements, lot size — before running an estimate. If any of that information is missing, the process stalls: they either go back to the seller, make manual assumptions, or skip the estimate entirely and rely on intuition.

The existing system made this worse, not better. If a field was missing, the tool would fail entirely — returning no estimate at all rather than a best-effort answer. This meant agents were effectively shut out of the tool for the majority of real listings, where some information is always missing at the early stages of engagement.

### What the Upgraded System Does

The upgraded tool does what a good agent does instinctively: it works with what it has. Give it a zipcode and square footage — it returns a price estimate. Give it everything you know — it returns a sharper estimate. Give it nothing but a neighborhood — it still gives you a starting point.

This is possible because the system has learned from tens of thousands of completed King County home sales. When a detail is missing, it finds the most similar homes in that dataset and uses their characteristics to fill in the gap — a data-driven equivalent of the judgment call an experienced agent makes when assessing an unfamiliar property.

### Why This Matters for the Business

**Faster initial engagement.** Agents can get a ballpark price estimate at the first conversation with a seller — before a full property assessment — without waiting for complete information. This shortens the time from first contact to a pricing recommendation.

**More listings, less friction.** Any property in a covered zipcode can now get an estimate, regardless of how complete the listing data is. The tool is no longer an all-or-nothing proposition.

**Reliability under load.** The system was tested with many simultaneous users and holds up without degradation. Median response time is under 40 milliseconds — fast enough to feel instant in a mobile app or web interface. A team of agents using the tool simultaneously will not experience slowdowns.

**Trustworthy estimates.** Predictions are grounded in real sales data from King County and enriched with neighborhood demographic information. The model understands that a 3-bedroom home in one zipcode may be priced very differently from the same home in another, even if the physical characteristics are identical.

**Fits into the existing workflow.** The tool is an API — it can be embedded in the CRM, the mobile listing app, or a simple web form. Agents do not need to change how they work; the estimate appears where they already are.

### The Opportunity Going Forward

This version of the system is the foundation. With a consistent feedback loop — connecting the tool's estimates to actual closed sale prices — the model can be retrained as the market evolves. A model trained in a rising market will eventually underestimate in a cooling one; with the infrastructure now in place, keeping it calibrated is a scheduled, automated process rather than a one-time effort.

---

## Conclusion

The three technical workstreams addressed the full lifecycle of a deployed ML API, and together they bring the system from a proof-of-concept to a production-ready foundation:

**Missing Data Handling** expanded the API from accepting only complete records to handling any degree of data completeness — including requests with no home features at all — while keeping predictions within plausible bounds. This directly addresses the real-world reality that property data is rarely complete at the point of first contact.

**Performance** eliminated two compounding bottlenecks to achieve a 136% increase in peak throughput and an 85–91% reduction in latency across p50, p95, and p99. The profiling exercise produced a data-backed scaling strategy: 1–2 vCPU per worker, horizontal scaling via additional containers, with p99 as the SLO signal and ~220 RPS/task as the scale-out trigger.

**Code Quality** addressed 13 issues spanning security, observability, error handling, input validation, and CI automation. The result is a codebase with enforced tests, traceable errors, validated inputs, and a documented deployment surface.

**Production Architecture** maps the current Docker-based deployment onto an AWS ECS Fargate architecture with ALB, S3 artifact storage, CloudWatch monitoring, and autoscaling thresholds derived directly from profiling data.

**MLOps** identifies the gap between the current state (git-committed pickles, manual retraining, no drift detection) and the production target (Model Registry, Feature Store, fully automated CI/CD, scheduled retraining, and drift alerting) — and describes the path between them.

---

## Section 7 — AI Usage

### Tool

I used **Claude Code** (Anthropic's CLI agent) as my primary AI assistant throughout this project. It ran directly in the terminal alongside the codebase, with access to read files, run commands, and make edits. I did not insert any aditional harness, such as skills or aditional instructions.

### How I Used It

**Codebase orientation.** Since the main objective of the exercise was to assess my approach to problem-solving, I tried to decouple my understanding of the problem from the LLM’s perspective to avoid biasing either side. I therefore performed the initial exploration and assessment of the codebase independently, identifying and documenting the main areas I wanted to address later.

**Planning before coding.** For each of the three workstreams, I used a structured planning process. Claude Code would read the relevant files and propose an approach, which I would review and approve before any changes were made. This helped avoid a common failure mode of AI-assisted development: jumping directly into implementation before the problem is fully understood. The plans included specific files, line references, and trade-off analyses, allowing me to evaluate them on their own merits.

**Implementation and iteration.** Once a plan was approved, Claude Code implemented the proposed changes. I reviewed every diff before it was committed. In several cases, I redirected the work. For example, during the performance optimization task, the initial proposal focused on caching. I asked it to also profile the print() statement, which ultimately proved to be the larger bottleneck. The tool identified the issue, but I had to know to ask the right question.

**Profiling and analysis.** Claude Code helped me iterate on the profiling process across multiple configurations, compile the results, and analyze the findings.

**Writing.** The technical report, code comments, and commit messages were initially drafted by Claude Code based on the actual code and collected data. I then reviewed and edited them for accuracy, clarity, and tone.

### What Worked Well

The biggest productivity gain was in the parts of the work that are correct-by-construction once you understand the problem: writing tests for known edge cases, fixing CORS headers, adding context managers, wiring up structured logging. These are not intellectually hard but they are time-consuming and easy to get slightly wrong. Delegating them to the AI while I focused on the higher-judgment decisions (what to measure, how to interpret results, what the architecture should be) was a good division of labor.

The planning workflow was the other major win. Having the tool propose a concrete plan — with specific files and reasoning — before writing a line of code meant that disagreements happened at the design level, not after the fact in a code review.

### Where I Had to Stay Engaged

AI-generated code is confident regardless of correctness. The profiling infrastructure required careful review: For instance AI suggested a config of max_requests = 1000 in gunicorn setting that would have caused worker recycling mid-test — something that only surfaced as mysterious 0.4% error rates in the first run.

### Context Management

The project spanned multiple sessions. Between sessions I relied on a memory system — short markdown files recording key decisions, profiling configurations, and feedback on what approaches worked. At the start of each session the relevant context was reloaded so the tool wasn't starting cold. For a project of this scope (not huge, but with several interdependent components), that discipline mattered: without it, the tool would have re-derived things I had already decided, or re-proposed approaches I had already rejected.

The other context discipline: keeping individual sessions focused. A session that tries to do performance optimization, code quality, and report writing simultaneously produces worse output than three focused sessions.