#!/bin/bash

set -e

# Default values (override in Kubernetes environment)
: "${MLFLOW_BACKEND_STORE_URI:=sqlite:///mlflow.db}"  # Default to SQLite if not set
: "${MLFLOW_DEFAULT_ARTIFACT_ROOT:=file:///mlflow_artifacts}"  # Default local path

echo "Starting MLflow server..."
echo "Backend Store URI: $MLFLOW_BACKEND_STORE_URI"
echo "Artifact Root: $MLFLOW_DEFAULT_ARTIFACT_ROOT"

# Start MLflow server
exec mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_DEFAULT_ARTIFACT_ROOT" \
    --host 0.0.0.0 \
    --port 5000 &


# Wait a bit for MLflow to start
sleep 5

echo "Starting FastAPI server..."
exec uvicorn server:app --host 0.0.0.0 --port 8000 --reload
