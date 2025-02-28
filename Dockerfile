# Use a lightweight Python image
FROM python:3.9-slim

LABEL authors="fatemeh"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://mlflow-svc:5000

RUN apt-get update && apt-get install -y \
    git \
    nano \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/entrypoint.sh

# Expose both MLflow (5000) and FastAPI (8000)
EXPOSE 5000 8000 5001

# Start script with shell execution
ENTRYPOINT ["/bin/sh", "/app/entrypoint.sh"]
