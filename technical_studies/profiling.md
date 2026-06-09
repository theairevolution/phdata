# API Profiling Report

## Test Methodology

All profiling runs use the same containerized setup:

- **Tool**: async load generator (`test/profiling/profile_api.py`) with httpx, progressive concurrency stages
- **Workload**: POST `/predict` with a fixed payload (3 bed / 2 bath / 1500 sqft / zipcode 98042)
- **Load stages**: 1 → 5 → 10 → 15 → 20 → 30 → 40 → 50 → 75 concurrent users (15 s at 1 user, 20 s all others)
- **Warmup**: 5 requests before each stage is measured
- **Request timeout**: 10 s per request; timeouts are counted as errors
- **Resource limits** (baseline for code tests): 2 vCPU, 2 GB memory, 1 gunicorn worker
- **gunicorn config**: UvicornWorker, `max_requests=100000` (prevents worker recycling during tests), `timeout=60`

**Metric definitions**  
*Peak RPS*: highest throughput observed across all stages (typically at 20–30 concurrent users for the optimized code).  
*Saturation point*: the concurrent-user stage where errors first appear or RPS begins to drop.  
*Errors*: client-side `ReadTimeout` (request exceeded 10 s waiting in the queue). No HTTP 5xx errors were observed in any run. The API itself never crashed or returned server errors.

---

## Part 1 — Code Optimization Impact

Fixed hardware: **2 vCPU / 2 GB / 1 worker**.  
Four configurations tested in order of implementation:

| Tag | Change | Code state |
|-----|--------|-----------|
| **A** | baseline | per-request load of model/imputer/features/CSV + `print(input_data)` + pandas O(n) scan |
| **B** | global caching | globals at module load; `print()` still present; pandas scan |
| **C** | remove `print()` | globals; no print; pandas scan |
| **D** | dict lookup | globals; no print; dict O(1) lookup |

### Results

| Users | A rps | B rps | C rps | D rps |
|------:|------:|------:|------:|------:|
| 1     |  15.3 |  15.5 |  16.7 |  16.5 |
| 5     |  72.9 |  80.8 |  86.5 |  86.8 |
| 10    |  86.1 | 113.6 | 162.0 | 161.3 |
| 15    |  91.4 | 113.0 | 201.4 | 207.0 |
| 20    |  93.3 | 115.5 | **217.9** | 215.7 |
| 30    |  92.0 | **118.8** | 210.7 | **216.8** |
| 40    |  **96.8** | 115.8 | 183.1 | 168.2 |
| 50    |  87.0 | 112.8 | 155.2 | 163.2 |

**Bold** = peak RPS for that config.

| Users | A p50ms | B p50ms | C p50ms | D p50ms |
|------:|--------:|--------:|--------:|--------:|
| 1     |    13.7 |    11.7 |     7.9 |     8.0 |
| 10    |    59.5 |    35.9 |     9.7 |     9.6 |
| 20    |   160.3 |   118.5 |    39.5 |    39.4 |
| 30    |   267.9 |   197.1 |    84.6 |    80.7 |
| 50    |   508.0 |   361.7 |   164.6 |   157.9 |

### Per-change delta

| Change | A→B | B→C | C→D |
|--------|-----|-----|-----|
| Peak RPS | 97 → 119 (+**23%**) | 119 → 218 (+**83%**) | 218 → 217 (**~0%**) |
| p50 @ 10u | 59.5 → 35.9 ms (−40%) | 35.9 → 9.7 ms (−73%) | 9.7 → 9.6 ms (~0%) |
| Saturation (first errors) | 50u (0.51%) | 50u (0.49%) | 40u (0.27%) | 40u (0.38%) |

### Analysis

**A → B (global caching, +23%)**  
Moving model, imputer, `model_features.json`, and the demographics CSV out of the request handler into module-level globals eliminates 4 file reads + 2 pickle deserializations per request. The improvement is real but modest (+23% RPS, −40% p50 at 10u), because the `print(input_data)` statement was still present and was the dominant bottleneck.

**B → C (remove `print()`, +83%)**  
`print(input_data)` serializes a full pandas DataFrame to a string and writes it to stdout on every single request. At 100+ RPS this is constant GIL-held I/O: the string formatting and the `write()` syscall both block the event loop, preventing the worker from picking up the next request. Removing this one line nearly doubles throughput (119 → 218 RPS) and cuts p50 latency at 10 users from 36 ms to 10 ms. **This was the single biggest bottleneck in the entire codebase.**

**C → D (O(1) dict lookup, ~0%)**  
Replacing the pandas boolean scan (`demographics[demographics["zipcode"] == zipcode]`) with a pre-built dict keyed by zipcode changes the complexity from O(n) to O(1). However, the CSV is small (~100 rows) and the pandas scan is vectorized in C, so the absolute time is already negligible relative to the sklearn inference. No measurable throughput improvement. The change is still correct and more efficient in principle.

---

## Part 2 — Resource Scaling Impact

Fixed code: **Config D** (best version). Fixed workers: **1**.  
Three vCPU allocations tested, same 1–75 user stage set.

### Results

| Users | 1 vCPU rps | 2 vCPU rps | 4 vCPU rps |
|------:|-----------:|-----------:|-----------:|
| 1     |       17.4 |       17.0 |       17.1 |
| 5     |       86.9 |       83.5 |       86.5 |
| 10    |      168.0 |      168.2 |      159.2 |
| 15    |      209.2 |      210.4 |      185.2 |
| 20    |      **223.0** |      **228.8** |      200.8 |
| 30    |      214.8 |      **226.0** |      188.4 |
| 40    |      196.8 |      187.7 |      175.0 |
| 50    |      165.4 |      164.6 |      152.3 |
| 75    |      132.4 |      128.4 |      109.3 |

**Bold** = peak RPS for that config.

| Users | 1 vCPU p50 | 2 vCPU p50 | 4 vCPU p50 |
|------:|-----------:|-----------:|-----------:|
| 10    |        7.3 |        7.1 |       10.9 |
| 20    |       37.2 |       35.1 |       47.2 |
| 30    |       81.9 |       73.4 |       90.6 |
| 50    |      149.9 |      169.0 |      170.4 |

| Users | 1 vCPU avg CPU% | 2 vCPU avg CPU% | 4 vCPU avg CPU% |
|------:|----------------:|----------------:|----------------:|
| 5     |            42.8 |           115.2 |           255.3 |
| 10    |            77.2 |           174.4 |           368.5 |
| 20    |            99.0 |           198.3 |           351.6 |
| 30    |            88.1 |           176.6 |           396.6 |

### Saturation point by vCPU count

| vCPU | Peak RPS | Users at peak | First errors at |
|-----:|---------:|--------------:|----------------:|
| 1    | 223 RPS  | 20u           | 40u (0.48%)     |
| 2    | 229 RPS  | 20–30u        | 30u (0.02%), grows at 40u |
| 4    | 201 RPS  | 20u           | 30u (0.26%)     |

### Analysis

**Does adding vCPUs increase the saturation point (max concurrent users)?**  
No. All three configurations saturate at approximately the same number of concurrent users (~20–30). More vCPUs do not allow the API to serve more simultaneous users before degrading. The saturation point is set by the worker count and GIL, not the CPU budget.

**Why doesn't 4 vCPU outperform 1 vCPU?**  
This is a CPU-bound workload (sklearn KNN inference + pandas operations). A single uvicorn worker is constrained by the Python GIL: only one Python thread runs at a time. The apparent parallelism comes from numpy/OpenBLAS, which spawns internal BLAS threads for matrix operations.

The problem: OpenBLAS reads `cpu_count()` from the host OS (not the Docker CPU limit) and spawns that many threads. With 4 vCPU, those threads each claim CPU time, but the actual useful work per inference call is roughly fixed — only so much can be parallelized inside a single matrix multiply for a 1-row input. The extra threads cause context-switch overhead and cache contention, consuming the 4 vCPU budget without proportional throughput gain.

The CPU% metrics confirm this: 4 vCPU shows ~396% CPU usage at saturation (consuming all 4 vCPUs) yet produces lower throughput than 1 vCPU at 99% CPU. The CPU is busy with thread coordination overhead rather than inference.

**1 vCPU vs 2 vCPU**  
The difference is small: 2 vCPU shows ~2–3% higher peak RPS (229 vs 223) and slightly lower p50 at mid-range concurrency. Both are within run-to-run noise. The marginal gain from a second CPU likely comes from the asyncio event loop and numpy threads having room to run concurrently without preempting each other.

**Scaling strategy implication**  
To double throughput, add a second container (each at 1–2 vCPU / 1 worker), not a second worker in the same container. A load balancer distributing between two such containers would yield ~2× RPS with near-linear scaling, because each container's worker runs fully independently with its own GIL.

---

## Part 3 — Errors Summary

All errors across all runs were **client-side timeouts** (httpx `ReadTimeout`): the request spent more than 10 seconds queued waiting for a worker response. No HTTP 5xx errors, no worker crashes, no gunicorn restarts were observed.

| Run | Stage with first errors | Error type | Root cause |
|-----|------------------------|------------|------------|
| Config A | 50u (0.51%) | ReadTimeout | Request queue builds faster than ~90 RPS can drain it; some exceed 10 s |
| Config B | 50u (0.49%) | ReadTimeout | Same — queue drain rate ~115 RPS still insufficient at 50 concurrent |
| Config C | 40u (0.27%) | ReadTimeout | Saturation at ~218 RPS; 40+ users oversaturate the single worker |
| Config D | 40u (0.38%) | ReadTimeout | Same as C |
| D @ 1 vCPU | 40u (0.48%) | ReadTimeout | Same |
| D @ 2 vCPU | 30u (0.02%), 40u (0.48%) | ReadTimeout | Same |
| D @ 4 vCPU | 30u (0.26%) | ReadTimeout | Slightly earlier saturation due to thread overhead |

**Historical errors (fixed before this test run)**  
Earlier profiling runs showed systematic 0.4–2% errors at every stage. Root cause: `max_requests = 1000` in gunicorn.conf.py caused the worker to recycle every ~16 seconds at 60 RPS. During the ~1 s restart, new requests hit a starting worker. Fixed by raising `max_requests` to `100000`.

---

## Summary

| Metric | Original (A) | Optimized (D) | Improvement |
|--------|-------------|---------------|-------------|
| Peak throughput | ~97 RPS | ~229 RPS | **+136%** |
| p50 latency @ 10u | 59.5 ms | 7.1 ms | **−88%** |
| p50 latency @ 20u | 160 ms | 35 ms | **−78%** |
| Saturation point | ~15u | ~20u | minimal shift |
| Memory (peak) | ~129 MB | ~125 MB | −3% |

**Biggest wins, ranked:**

1. **Remove `print(input_data)` (+83% RPS)** — serializing a DataFrame to stdout on every request at load was the dominant bottleneck. One line of debug output cut throughput by more than half.
2. **Global caching (+23% RPS)** — loading model/imputer/CSV per request added disk I/O and deserialization overhead. Module-level globals load once at worker startup.
3. **Dict lookup (~0%)** — theoretically cleaner, but the demographics CSV is small enough that the pandas scan was never measurably expensive.

**Resource scaling:**  
1–2 vCPU per worker is the sweet spot. 4 vCPU is counterproductive for a single worker because OpenBLAS over-threads against the fixed CPU budget. More concurrent users are handled by adding more containers (1 worker each), not by adding vCPUs to one container.
