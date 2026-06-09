# Code Quality Improvements Report

**Date:** 2026-06-07  
**Branch:** main  
**Scope:** Full codebase audit and remediation

---

## Summary

13 issues were identified and addressed across three priority tiers. The changes below are grouped by tier (quick wins → meaningful improvements → structural). Each entry shows what changed, why it matters, and the estimated impact.

---

## Tier 1 — Quick Wins

### 1. CORS misconfiguration fixed
**File:** `src/main.py`  
**Change:** `allow_credentials=True` → `allow_credentials=False`

**Why it matters — High security impact.**  
Combining `allow_origins=["*"]` with `allow_credentials=True` is explicitly invalid per the Fetch/CORS specification. Browsers reject this combination, but the server was still advertising it, which could mislead proxy/CDN layers. More importantly, if origins were ever narrowed to a specific domain, `allow_credentials=True` would silently enable cross-site request forgery. Fixing it now prevents a footgun from surviving into a production deployment.

---

### 2. Application logging added
**Files:** `src/main.py`, `src/api/endpoints.py`  
**Change:** Added `logging.basicConfig` at startup; `logger = logging.getLogger(__name__)` in endpoints; startup confirmation lines for each artifact loaded; warning on unknown zipcode; error on prediction failure; debug line per successful prediction.

**Why it matters — High observability impact.**  
The API had zero logging. Errors returned generic 500s with no server-side trace. In production (or even local debugging), there was no way to know which zipcode triggered a failure, which model file was actually loaded, or whether startup completed successfully. Logging is the minimum viable observability layer for a deployed service.

---

### 3. Error handling added to `/predict`
**File:** `src/api/endpoints.py`  
**Change:** Added an explicit 422 guard for unknown zipcodes; wrapped the full prediction pipeline in try/except returning 500 with a log line on unexpected failures.

**Why it matters — High reliability + API contract impact.**  
Previously, any zipcode not in `_DEMOGRAPHICS` caused an unhandled `KeyError`, which FastAPI converted to an opaque 500 Internal Server Error. Callers had no way to distinguish "zipcode not found" from "server crashed." The fix returns a 422 with a clear `detail` message for unknown zipcodes, and 500 only for genuinely unexpected errors — a meaningful distinction for API consumers.

---

### 4. Model/data paths made robust
**File:** `src/api/endpoints.py`  
**Change:** Replaced hardcoded `"model/model.pkl"` (CWD-relative) with `Path(__file__).parent.parent / "model" / "model.pkl"` (__file__-relative).

**Why it matters — Medium reliability impact.**  
The previous paths worked only when the process was started from a specific working directory. In the Docker test environment this was masked by symlinks; in local development it was silently fragile. The `__file__`-relative approach resolves correctly in every environment: production Docker, test Docker, and local dev — without relying on symlinks or CWD conventions.

---

### 5. File-handle leaks fixed in training scripts
**Files:** `src/model/create_model.py`, `create_imputer.py`  
**Change:** Both scripts called `open()` and passed the file object directly to `pickle.dump()` / `json.dump()` without closing it. Replaced with `with open(...) as f:` context managers.

**Why it matters — Low runtime impact, good hygiene.**  
Standalone scripts running to completion reclaim file handles at process exit, so there's no practical crash risk. However, the pattern is incorrect and teaches bad habits. Using context managers is the Python-idiomatic way to guarantee file handles are closed even if an exception occurs mid-write (preventing partial/corrupt artifact files).

---

### 6. Unused `numpy` import removed
**File:** `create_imputer.py`  
**Change:** Deleted `import numpy as np` (never referenced in the file).

**Why it matters — Low impact, code cleanliness.**  
Unused imports add noise that makes readers wonder if `np` is used somewhere they missed. Removing it makes the dependency surface accurate.

---

### 7. Duplicate `create_model.py` deleted
**File:** `create_model.py` (project root)  
**Change:** Deleted. Canonical version remains at `src/model/create_model.py`.

**Why it matters — Medium maintainability impact.**  
Two identical files diverge silently: a change made to one is not reflected in the other. The root-level copy had no clear purpose (no Makefile target, no documentation pointing to it) and posed an ongoing maintenance risk.

---

### 8. Dead `loader.py` deleted
**File:** `src/utils/loader.py`  
**Change:** Deleted. The three functions (`load_model`, `load_features`, `get_demographics`) were never imported anywhere in the codebase.

**Why it matters — Medium maintainability impact.**  
Dead code is actively harmful: it misleads new contributors into thinking a module provides a reusable interface, it accumulates drift from the actual implementation, and it creates a false impression of separation of concerns. Deleting it removes ambiguity.

---

## Tier 2 — Meaningful Improvements

### 9. Pydantic field validation added to `HomeFeatures`
**File:** `src/api/endpoints.py`  
**Change:** All numeric optional fields now use `Field(default=None, ge=0)`, rejecting negative values with a 422 before they reach the model.

**Why it matters — High API correctness impact.**  
A KNN regressor will gladly predict a price for a house with −5 bedrooms — the model has no built-in guard. Without validation, the API would silently return nonsensical predictions for invalid inputs. With `ge=0`, the API clearly rejects the input at the boundary, and clients receive an actionable 422 instead of a wrong 200.

---

### 10. `IMPUTER_FEATURES` deduplicated
**Files:** `src/api/endpoints.py` (canonical definition), `test/unit/test_imputer_unit.py` (now imports it)  
**Change:** `test_imputer_unit.py` previously copy-pasted the full feature list. It now does `from src.api.endpoints import IMPUTER_FEATURES`.

**Why it matters — Medium maintainability impact.**  
Two identical lists in separate files will drift. If a feature is added to the imputer but only one list is updated, tests will silently pass against stale data. A single source of truth guarantees consistency at import time.

---

### 11. Unit test coverage expanded
**File:** `test/unit/test_api_unit.py`  
**Change:** Added four new tests: `test_predict_endpoint_price_positive` (price must be > 0), `test_predict_endpoint_invalid_zipcode` (unknown zipcode → 422), `test_predict_endpoint_negative_bedrooms_rejected` (−1 bedrooms → 422), `test_predict_endpoint_negative_sqft_rejected` (−500 sqft → 422).

**Why it matters — High regression safety impact.**  
The original test suite had 2 API unit tests, both on the happy path. There was zero coverage of the error cases added in this session, meaning those fixes could be silently reverted with no test failure. The new tests pin the contract for the most important edge cases: unknown zipcode handling and field validation boundaries.

---

## Tier 3 — Structural Improvements

### 12. GitHub Actions CI pipeline added
**File:** `.github/workflows/ci.yml`  
**Change:** Created a `unit-tests` job that installs dependencies and runs `pytest test/unit` on every push and pull request to `main`.

**Why it matters — Critical process impact.**  
The project had no automated gate on commits. A breaking change to any endpoint or model loading could be merged to main undetected. CI is the minimum viable safety net: it turns the existing unit tests from a local convenience into an enforced quality gate. The pipeline targets `test/unit` only (no running API needed), keeping it fast and dependency-free.

---

### 13. `.env.example` created
**File:** `.env.example`  
**Change:** Documented all five environment variables used across the project (`WEB_CONCURRENCY`, `API_CPUS`, `API_BASE_URL`, `API_SERVICE_NAME`, `RESULTS_DIR`) with defaults and descriptions.

**Why it matters — Medium developer experience impact.**  
The variables were scattered across `gunicorn.conf.py`, `docker-compose.test.yml`, and the profiling script with no single reference. A new contributor setting up a local environment had to grep across files to discover them. `.env.example` is the standard convention for surfacing this configuration in one place.

---

## Files Changed

| File | Action | Tier |
|------|--------|------|
| `src/main.py` | Edited — fix CORS, add logging | 1 |
| `src/api/endpoints.py` | Edited — paths, error handling, logging, validators | 1, 2 |
| `src/model/create_model.py` | Edited — context managers for file I/O | 1 |
| `create_imputer.py` | Edited — remove unused import, context manager | 1 |
| `test/unit/test_api_unit.py` | Edited — 4 new tests | 2 |
| `test/unit/test_imputer_unit.py` | Edited — import IMPUTER_FEATURES from source | 2 |
| `create_model.py` (root) | **Deleted** — duplicate of src/model/create_model.py | 1 |
| `src/utils/loader.py` | **Deleted** — dead code, never imported | 1 |
| `.github/workflows/ci.yml` | **Created** — CI pipeline | 3 |
| `.env.example` | **Created** — environment variable reference | 3 |

---

## What Was Not Changed (and Why)

| Item | Rationale |
|------|-----------|
| Missing `__init__.py` files | The import system already works via PYTHONPATH in all environments. Adding `__init__.py` would be a larger refactor (changes all import statements across the project) with minimal benefit for a single-service app. |
| `model.pkl` / `imputer.pkl` in version control | Removing them requires a model registry or download step in CI, which is a larger infrastructure change beyond this audit's scope. |
| Pydantic `.dict()` → `.model_dump()` migration | The codebase may target Pydantic v1 (`.dict()` is v1 API). A compatibility shim was added (`model_dump()` with fallback to `.dict()`), but a full migration requires knowing the pinned Pydantic version. |
| Centralized `config.py` for all paths and constants | Import path complexity across Docker environments made this higher-risk than the direct `Path(__file__)`-relative approach applied to endpoints.py. |
