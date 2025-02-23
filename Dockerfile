# Use a single base image (don't use multiple FROMs unless necessary)
FROM python:3.9-slim

LABEL authors="fatemeh"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://mlflow-svc:5000

RUN apt-get update && apt-get install -y \
    git \
    nano \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
