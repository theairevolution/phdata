import multiprocessing

# Worker class — uvicorn workers give async FastAPI support under gunicorn's
# process manager, combining multi-process parallelism with async I/O.
worker_class = "uvicorn.workers.UvicornWorker"

# (2 × CPU count) + 1 keeps all cores busy even when individual workers block
# briefly on I/O. Override with the WEB_CONCURRENCY env var if needed.
workers = int(__import__("os").getenv("WEB_CONCURRENCY", multiprocessing.cpu_count() * 2 + 1))

bind = "0.0.0.0:8000"

# Restart workers after this many requests to reclaim any memory creep.
# Set high enough to not trigger during a profiling run (~9000 requests over 2.5 min).
max_requests = 100000
max_requests_jitter = 1000

# Seconds a worker may be silent before gunicorn kills and restarts it.
# KNN imputation on first request after startup can be slow, so give it room.
timeout = 60

# Forward worker stdout/stderr to gunicorn's logger.
accesslog = "-"
errorlog  = "-"
loglevel  = "info"
