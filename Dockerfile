FROM python:3.10-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src .
COPY gunicorn.conf.py .

CMD ["gunicorn", "main:app", "--config", "gunicorn.conf.py"]