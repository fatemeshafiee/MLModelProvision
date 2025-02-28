import requests
import os
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
import uuid
from schemes import  *
from schemes import NwdafMLModelProvSubsc
import subprocess
from fastapi import BackgroundTasks
import time
import mlflow.sklearn  
import pickle
import joblib

import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
app = FastAPI()


# ToDo (Update this based on k8s)
MLFLOW_TRACKING_URI = "http://mlflow-svc:5000"
MLFLOW_ARTIFACT_PATH = "/mlflow_artifacts"
MLFLOW_SERVE_URI = "http://mlflow-svc:5001/models/{model_name}/versions/{model_version}/invocations"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

os.makedirs(MLFLOW_ARTIFACT_PATH, exist_ok=True)

class ModelLogRequest(BaseModel):

    model_name: str
    model_url: HttpUrl
    mLEvent: NwdafEvent
    event_filter: Optional[Dict[str, Any]] = None
    inference_data: Optional[Dict[str, Any]] = None
    target_ue: Optional[Dict[str, Any]] = None
    is_update: bool = False  # Whether this is an update to an existing model


def flatten_dict(d_in, parent_key='', sep='_'):

    items = []
    for k, v in d_in.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())

        elif isinstance(v, list):
            # Sort list (convert to string for sorting consistency)
            sorted_list = sorted(v, key=lambda x: str(x))
            items.append((new_key, str(sorted_list)))  # Store sorted list as a string

        else:
            items.append((new_key, str(v)))

    return dict(items)


def register_existing_model(model_name: str) -> str:
    client = mlflow.tracking.MlflowClient()

    runs = client.search_runs(experiment_ids=["0"], order_by=["start_time DESC"])
    if not runs:
        raise Exception(f"No runs found for model {model_name}")

    latest_run_id = runs[0].info.run_id
    model_uri = f"runs:/{latest_run_id}/model"

    registered_model = mlflow.register_model(model_uri, model_name)

    time.sleep(5)

    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version

    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production"
    )
    model_url = MLFLOW_SERVE_URI.format(model_name=model_name, model_version=latest_version)
    return model_url

def generate_mlflow_tags(
    mLEvent: NwdafEvent,
    event_filter: Optional[Dict[str, Any]] = None,
    inference_data: Optional[Dict[str, Any]] = None,
    target_ue: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    tags = {"mLEvent": str(mLEvent)}  # Add mLEvent as a tag
    if event_filter:
        tags.update(flatten_dict(event_filter, "eventFilter"))
    if inference_data:
        tags.update(flatten_dict(inference_data, "inferenceData"))
    if target_ue:
        tags.update(flatten_dict(target_ue, "targetUe"))
    return tags

def generate_tags_for_model_log_request(request: ModelLogRequest) -> Dict[str, str]:
    return generate_mlflow_tags(
        mLEvent=request.mLEvent,
        event_filter=request.event_filter,
        inference_data=request.inference_data,
        target_ue=request.target_ue
    )

MLFLOW_TRACKING_URI = "http://mlflow-svc:5000"
MLFLOW_ARTIFACT_PATH = "/mlflow_artifacts"
MLFLOW_SERVE_URI = "http://mlflow-svc:5001/models/{model_name}/versions/{model_version}/invocations"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def download_model(model_url: str, model_name: str):
    local_path = os.path.join(MLFLOW_ARTIFACT_PATH, f"{model_name}_{uuid.uuid4().hex()}.pkl")
    response = requests.get(model_url, stream=True)
    if response.status_code == 200:
        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return local_path
    else:
        raise HTTPException(status_code=400, detail="Failed to download model from URL")

@app.post("/log_model/")
async def log_ml_model(request: ModelLogRequest):
    try:
        model_name = request.model_name
        tags = generate_tags_for_model_log_request(request)

        model_local_path = download_model(request.model_url, model_name)
        loaded_model = joblib.load(model_local_path)
        if not hasattr(loaded_model, "predict"):
            raise HTTPException(status_code=500, detail="❌ Model does not have a `predict()` method!")
        
        
        artifact_path = f"models/{model_name}"
        
        with mlflow.start_run():
            mlflow.log_param("original_model_url", request.model_url)
            # Log the model using MLflow's sklearn flavor; this creates the MLmodel file
            mlflow.set_tags(tags)
            mlflow.sklearn.log_model(
                sk_model=loaded_model,
                artifact_path=artifact_path,
                serialization_format="cloudpickle"
            )
            run_id = mlflow.active_run().info.run_id

            model_uri = f"runs:/{run_id}/{artifact_path}"
            client = mlflow.tracking.MlflowClient()
            registered_model = mlflow.register_model(model_uri, model_name)
            for tag_key, tag_value in tags.items():
                client.set_registered_model_tag(model_name, tag_key, tag_value)
        
        time.sleep(5)
        
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production"
        )
        
        # ★ Verify that the MLmodel file exists at the expected location.
        # This should be at: /mlflow_artifacts/0/<run_id>/artifacts/models/<model_name>/MLmodel
        mlmodel_path = os.path.join(MLFLOW_ARTIFACT_PATH, "0", run_id, "artifacts", artifact_path, "MLmodel")
        if not os.path.exists(mlmodel_path):
            raise HTTPException(status_code=500, detail=f"❌ MLmodel file missing at {mlmodel_path}")
        
        # Serve the model using the standard serving URI (models:/{model_name}/{latest_version})
        subprocess.Popen([
            "mlflow", "models", "serve",
            "--model-uri", f"models:/{model_name}/{latest_version}",
            "--host", "0.0.0.0",
            "--port", "5001",
            "--no-conda"
        ])
        
        inference_url = MLFLOW_SERVE_URI.format(model_name=model_name, model_version=latest_version)
        
        return {
            "message": f"✅ Model {model_name} logged, registered, and served successfully!",
            "model_uri": model_uri,
            "inference_url": inference_url,
            "model_version": latest_version
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_latest_model/{model_name}")
async def get_latest_model(model_name: str):
    """
    Retrieves the latest version of the specified ML model.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        latest_model = client.get_latest_versions(model_name, stages=["Production"])

        if latest_model:
            # Check if the model is currently served
            inference_url = None
            response = requests.get(f"{MLFLOW_TRACKING_URI}/models/{model_name}")
            if response.status_code == 200:
                inference_url = f"{MLFLOW_TRACKING_URI}/models/{model_name}/invocations"

            return {
                "model_uri": latest_model[0].source,
                "inference_url": inference_url  # Will be None if not served
            }

        else:
            raise HTTPException(status_code=404, detail="Model not found in Production stage")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query_models/")
async def query_models(filters: Dict[str, Any]):

    try:
        flattened_filters = flatten_dict(filters)

        query_str = " AND ".join([f"tag.{key}='{value}'" for key, value in flattened_filters.items()])

        client = mlflow.tracking.MlflowClient()
        models = client.search_model_versions(query_str)

        results = []
        for model in models:
            inference_url = None
            try:
                response = requests.get(f"{MLFLOW_TRACKING_URI}/models/{model.name}")
                if response.status_code == 200:
                    inference_url = f"{MLFLOW_TRACKING_URI}/models/{model.name}/invocations"
            except requests.RequestException:
                pass

            results.append({
                "model_uri": model.source,
                "version": model.version,
                "inference_url": inference_url  # Will be None if not served
            })

        return results if results else {"message": "No models match the provided filters."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.put("/update_model/{model_name}")
async def update_model(model_name: str, new_model_url: HttpUrl):
    """
    Updates an existing ML model by archiving the old version and making the new one available.
    """
    try:
        client = mlflow.tracking.MlflowClient()

        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        for version in latest_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )

        model_local_path = download_model(str(new_model_url), model_name)

        with mlflow.start_run():
            mlflow.log_param("updated_model_url", str(new_model_url))
            mlflow.log_artifact(model_local_path, artifact_path="model")
            result = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", model_name)

        # Step 4: Move the new version to Production
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Production"
        )

        return {
            "message": f"Model {model_name} updated successfully. Old version archived, new version in Production.",
            "model_uri": result.source,
            "new_version": result.version
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Run with Uvicorn
# uvicorn filename:app --host 0.0.0.0 --port 8000
