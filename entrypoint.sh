#!/bin/bash

set -e

# Default values (override in Kubernetes environment)
: "${MLFLOW_BACKEND_STORE_URI:=sqlite:///mlflow.db}"  # Default to SQLite if not set
: "${MLFLOW_DEFAULT_ARTIFACT_ROOT:=file:///mlflow_artifacts}"  # Default local path

echo "Starting MLflow server..."
echo "Backend Store URI: $MLFLOW_BACKEND_STORE_URI"
echo "Artifact Root: $MLFLOW_DEFAULT_ARTIFACT_ROOT"

mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_DEFAULT_ARTIFACT_ROOT" \
    --host 0.0.0.0 \
    --port 5000 \
    --serve-artifacts \
    --artifacts-destination "$MLFLOW_ARTIFACT_ROOT" \
    --registry-store-uri postgresql://mlflow:mlflowpassword@postgresql-svc:5432/mlflowdb \
    --gunicorn-opts "--log-level debug" &


# Wait a bit for MLflow to start
sleep 5

echo "Starting FastAPI server..."
exec uvicorn server:app --host 0.0.0.0 --port 8000 --workers 2

