# FastAPI Machine Learning Model Deployment

This project implements a RESTful API using FastAPI to deploy a machine learning model for predicting home prices based on various features. The model is trained on real estate data and can provide predictions based on user input.

## Project Structure

```
phdata-cleanrepo/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI pipeline (unit tests on push/PR)
├── src/
│   ├── main.py                       # FastAPI application entry point
│   ├── api/
│   │   └── endpoints.py              # /predict and /health endpoints
│   ├── model/
│   │   ├── Dockerfile                # Docker image for model training
│   │   ├── create_model.py           # Trains the KNN regressor and saves model.pkl
│   │   ├── model.pkl                 # Serialized model (generated — not committed)
│   │   ├── imputer.pkl               # Serialized KNN imputer (generated — not committed)
│   │   └── model_features.json       # Feature order expected by the model
│   └── data/
│       ├── kc_house_data.csv         # Training data
│       ├── zipcode_demographics.csv  # Demographic features joined at inference time
│       └── future_unseen_examples.csv
├── test/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_api_unit.py          # API unit tests (happy path + edge cases)
│   │   └── test_imputer_unit.py      # Imputer behaviour and bounds tests
│   ├── integration/
│   │   └── test_api_integration.py   # End-to-end tests against a running container
│   └── profiling/
│       └── profile_api.py            # Progressive load-stage profiler
├── technical_studies/
│   ├── code_quality.md               # Code quality audit notes
│   └── profiling.md                  # Profiling methodology and findings
├── notebooks/
│   └── imputation_experiment.ipynb   # Exploratory analysis for imputation approach
├── presentation_resources/           # Charts from profiling runs for presentations
├── profiling-baselines/
│   └── baseline_original_api.json    # Profiling results for the unmodified API
├── create_imputer.py                 # Trains the KNN imputer and saves imputer.pkl
├── Dockerfile                        # API production image
├── Dockerfile.test                   # Test runner image
├── Dockerfile.profiling              # Load profiler image
├── Makefile                          # Build, test, and profiling automation
├── docker-compose.test.yml           # Docker Compose for tests and profiling
├── gunicorn.conf.py                  # Gunicorn worker configuration
├── pytest.ini                        # Pytest configuration
├── requirements.txt                  # API dependencies
├── requirements-test.txt             # Test dependencies
├── requirements-profiling.txt        # Profiling dependencies
├── README.md
└── TECHNICAL_REPORT.md               # Detailed write-up of all improvements
```

## Setup Instructions

### Prerequisites

- Docker installed on your system
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd mle-project-challenge-2026
```

### Step 2: Generate Model Artifacts (First Time Only)

Before running the API, you need to generate the model files. This only needs to be done once, or when you want to retrain the model.

**Build the model creation Docker image:**
```bash
docker build -f src/model/Dockerfile -t create-model .
```

**Run the container to generate model artifacts:**
```bash
docker run --rm -v "$(pwd)/src/model:/app/model" create-model
```

This will create `model.pkl` and `model_features.json` in the `src/model/` directory.

**Generate the KNN imputer artifact:**
```bash
cd src && python ../create_imputer.py
```

This creates `imputer.pkl` in `src/model/`. The imputer is required by the API to fill in any missing home feature fields at prediction time (see [Missing Data Handling](#missing-data-handling) below).

### Step 3: Build and Run the API

**Build the API Docker image:**
```bash
docker build -t mle-project-challenge-2026 .
```

**Run the API container:**
```bash
docker run -d -p 8000:8000 --name housing-api mle-project-challenge-2026
```

### Step 4: Access the API

Open your browser and go to `http://127.0.0.1:8000/docs` to view the interactive API documentation.

### Managing the Container

**Stop the container:**
```bash
docker stop housing-api
```

**Start the container again:**
```bash
docker start housing-api
```

**Remove the container:**
```bash
docker rm housing-api
```

**View container logs:**
```bash
docker logs housing-api
```

## Usage

Send a POST request to `/predict` with a JSON body. Only `zipcode` is required — all home feature fields are optional. Missing fields are filled automatically by the KNN imputer (see [Missing Data Handling](#missing-data-handling)).

```json
{
  "zipcode": "98103",
  "bedrooms": 3,
  "bathrooms": 2.0,
  "sqft_living": 1800,
  "sqft_lot": 5000,
  "floors": 1.0,
  "sqft_above": 1800,
  "sqft_basement": 0
}
```

**Response:**
```json
{"predicted_price": 625000.0}
```

All numeric fields must be non-negative (`>= 0`). An unknown zipcode returns `422 Unprocessable Entity`.

## Missing Data Handling

All seven home feature fields (`bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `sqft_above`, `sqft_basement`) are optional. When one or more are absent, a KNN imputer fills them before passing the record to the model.

The imputer was trained with `n_neighbors=5, weights="distance"` on the same training split as the main model. For each missing value it finds the five most similar complete records in the training set and computes a distance-weighted average — more similar neighbours contribute proportionally more to the fill value.

| Scenario | Result |
|----------|--------|
| All fields provided | Fields passed through unchanged |
| One or more fields missing | Missing fields imputed; prediction returned |
| Only `zipcode` provided | All seven fields imputed; prediction returned |

This allows the API to return a best-effort estimate for listings at any stage of completeness, from a fully specified property to a bare-minimum record with only a location.

## Testing

This project uses Docker-based testing to ensure environment consistency between testing and production. All tests run inside Docker containers, eliminating "works on my machine" issues.

### Quick Start

Run unit tests (fast, recommended for development):
```bash
make test-unit
```

Run integration tests (full environment):
```bash
make test-integration
```

Run all tests:
```bash
make test-all
```

### Test Types

**Unit Tests** (`test/unit/`)
- Use FastAPI TestClient for in-process testing
- No external dependencies or containers required
- Fast execution (typically under 30 seconds)
- Ideal for rapid development iteration

**Integration Tests** (`test/integration/`)
- Test against a real running API container
- Verify end-to-end functionality via HTTP requests
- Ensure Docker networking and orchestration work correctly
- More comprehensive but slower execution

### Running Tests

#### Unit Tests Only

```bash
make test-unit
```

This command:
- Builds the test Docker image
- Runs only tests in `test/unit/` directory
- Generates coverage reports
- Completes quickly without starting the full API container

#### Integration Tests Only

```bash
make test-integration
```

This command:
- Builds both API and test containers using Docker Compose
- Starts the API container and waits for health check
- Runs tests in `test/integration/` directory
- Automatically stops and removes containers when complete

#### All Tests

```bash
make test-all
```

Runs both integration and unit tests for comprehensive validation.

### Viewing Coverage Reports

After running tests, coverage reports are generated in the `test-results/` directory:

**HTML Coverage Report:**
```bash
open test-results/coverage/index.html
```

**Terminal Coverage Summary:**
Coverage is automatically displayed in the terminal after test execution.

### Running Specific Tests

Run a specific test file:
```bash
docker run --rm ml-api-test pytest test/unit/test_api_unit.py -v
```

Run a specific test function:
```bash
docker run --rm ml-api-test pytest test/unit/test_api_unit.py::test_predict_endpoint -v
```

Run tests matching a pattern:
```bash
docker run --rm ml-api-test pytest -k "predict" -v
```

### Troubleshooting

**Issue: "Cannot connect to the Docker daemon"**

Solution: Ensure Docker is running on your system.
```bash
docker ps  # Should list running containers without error
```

**Issue: Integration tests fail with connection errors**

Solution: Check if the API container is healthy.
```bash
docker-compose -f docker-compose.test.yml up
# In another terminal:
docker-compose -f docker-compose.test.yml ps
docker-compose -f docker-compose.test.yml logs api
```

**Issue: Tests pass locally but fail in Docker**

Solution: This usually indicates environment differences. Check:
- Model artifacts exist in `model/` directory
- Data files exist in `data/` directory
- All dependencies are listed in `requirements.txt`

**Issue: "Port 8000 already in use"**

Solution: Stop any running containers or services using port 8000.
```bash
docker-compose -f docker-compose.test.yml down
docker stop housing-api  # If the main API is running
```

**Issue: Test results not appearing in `test-results/` directory**

Solution: Ensure the directory exists and has proper permissions.
```bash
mkdir -p test-results
chmod 755 test-results
```

**Issue: Tests are very slow**

Solution: Run unit tests only for faster feedback during development.
```bash
make test-unit  # Much faster than integration tests
```

**Issue: "Image not found" errors**

Solution: Build the test image explicitly.
```bash
make test-build
```

### Cleaning Up

Remove test containers and artifacts:
```bash
make clean
```

This removes:
- All Docker containers created by docker-compose.test.yml
- All test result files and coverage reports

### Development Workflow

For rapid development iteration:

1. Make code changes in `src/` or test changes in `test/`
2. Run unit tests: `make test-unit`
3. Fix any issues and repeat
4. Before committing, run full suite: `make test-all`

The Docker setup mounts source code as volumes, so you don't need to rebuild containers for every change during integration testing.

## Continuous Integration

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs the unit test suite automatically on every push and pull request to `main`. The `unit-tests` job installs dependencies and runs `pytest test/unit`. Pull requests cannot merge unless all unit tests pass.

## Load Profiling

The profiling module runs progressive load stages against the API and records
how CPU and memory evolve as concurrency increases.  It requires the Docker
socket to be accessible so it can query live container stats.

### How it works

Five stages are executed in sequence, each sending continuous POST requests to
`/predict` for a fixed duration:

| Stage | Concurrent users | Duration |
|-------|-----------------|----------|
| 1     | 1               | 15 s     |
| 2     | 5               | 20 s     |
| 3     | 10              | 20 s     |
| 4     | 15              | 20 s     |
| 5     | 20              | 20 s     |

During each stage the profiler polls the API container's cgroup stats every 1.5 s
(CPU % and memory MB), then generates three charts and a JSON summary in
`test-results/profiling/<run_id>/`.

### Quick start

```bash
make profile
```

This target:
1. Builds the API image and starts it (with health-check gating).
2. Builds `Dockerfile.profiling` and starts the `profiling` container.
3. Runs all five load stages and stops all containers when done.

### Output

```
test-results/profiling/<YYYYMMDDTHHMMSS>/
├── resource_timeline.png   — CPU % and memory MB over time (stage boundaries marked)
├── latency_by_stage.png    — p50 / p95 / p99 response latency per stage
├── throughput_by_stage.png — requests/second and error rate per stage
└── summary.json            — machine-readable results for all stages
```

A summary table is also printed to the terminal:

```
Users   Reqs     RPS    Err%    p50 ms   p95 ms   p99 ms  AvgCPU%  PeakMem MB
──────────────────────────────────────────────────────────────────────────────
    1     42    2.80     0.0     340.1    380.2    390.5      12.3       145.0
    5    180    9.00     0.0     545.3    620.1    640.0      38.7       147.2
   10    290   14.50     0.0     680.4    790.2    820.3      65.1       148.5
   15    380   19.00     1.5     920.0   1200.0   1300.0      78.4       150.0
   20    400   20.00     4.0    1150.0   1500.0   1600.0      85.0       151.2
```

> The profiler exits with a non-zero code if any stage exceeds 10% errors, making
> it usable as a gate in CI pipelines.

### Profiling without Docker Compose

You can run the profiler against any already-running API instance:

```bash
docker build -f Dockerfile.profiling -t ml-api-profiler .
mkdir -p test-results/profiling

docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$(pwd)/test-results/profiling":/app/results \
  -e API_BASE_URL=http://host.docker.internal:8000 \
  -e API_SERVICE_NAME=api \
  ml-api-profiler
```

### Customising the load stages

Edit the `LOAD_STAGES` list in `test/profiling/profile_api.py`:

```python
LOAD_STAGES = [
    (1,  15),   # (concurrent_users, duration_seconds)
    (5,  20),
    (10, 20),
    (15, 20),
    (20, 20),
]
```

## Feedback

We welcome any feedback regarding the project or the interview process. Your insights are valuable to us as we strive to improve the experience for future candidates.