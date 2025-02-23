import requests
import os
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
from uuid import uuid4
from schemes import  *
from schemes import NwdafMLModelProvSubsc
import subprocess
from fastapi import BackgroundTasks
import time

app = FastAPI()



# ToDo (Update this based on k8s)
MLFLOW_TRACKING_URI = "http://mlflow-svc:5000"
MLFLOW_ARTIFACT_PATH = "/mlflow_artifacts"
MLFLOW_SERVE_URI = "http://mlflow-svc:5000/models/{model_name}/versions/{model_version}/invocations"
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


def download_model(model_url: str, model_name: str):

    local_path = os.path.join(MLFLOW_ARTIFACT_PATH, f"{model_name}_{uuid4().hex}.pkl")

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
        model_url = str(request.model_url)

        model_local_path = download_model(model_url, model_name)

        tags = generate_tags_for_model_log_request(request)

        with mlflow.start_run():
            mlflow.log_param("original_model_url", model_url)
            mlflow.set_tags(tags)

            mlflow.pyfunc.log_model(artifact_path="model", loader_module="mlflow.sklearn", code_path=None)

            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"


            registered_model = mlflow.register_model(model_uri, model_name)

        time.sleep(5)

        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version

        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production"
        )

        inference_url = f"http://mlflow-container:5001/models/{model_name}/versions/{latest_version}/invocations"

        return {
            "message": f"Model {model_name} logged and registered successfully",
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
