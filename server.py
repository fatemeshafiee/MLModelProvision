import requests
from fastapi import FastAPI, HTTPException, Response, Header
from pydantic import BaseModel
from typing import Dict
import uvicorn
import mlflow
from uuid import uuid4



# Import your Pydantic models
from types import NwdafMLModelProvSubsc
from mLFlowAPIs import generate_mlflow_tags
app = FastAPI()

# In-memory storage for subscriptions
subscriptions: Dict[str, NwdafMLModelProvSubsc] = {}

API_ROOT = "http://localhost:8000"  # TODO: Change it for k8s.
MLFLOW_MODEL_SERVE_URL = "http://mlflow-container:5001/invocations"   # TODO: Change it for k8s.


def search_mlflow_model(subscription: NwdafMLModelProvSubsc):

    filters = generate_mlflow_tags(subscription)
    query_str = " AND ".join([f"tag.{key}='{value}'" for key, value in filters.items()])

    try:
        client = mlflow.tracking.MlflowClient()
        models = client.search_model_versions(query_str)

        return models if models else None

    except Exception as e:
        print(f"MLflow query failed: {e}")
        return None


def serve_ml_model(model_name: str):

    try:
        requests.post(
            f"{MLFLOW_MODEL_SERVE_URL}/start",
            json={"model_name": model_name}
        )
        return f"{MLFLOW_MODEL_SERVE_URL}"
    except Exception as e:
        print(f"Error serving ML model: {e}")
        return None


@app.post("/subscribe/", status_code=201, response_model=Dict[str, Any])
async def subscribe(subscription: NwdafMLModelProvSubsc, response: Response):

    matched_models = search_mlflow_model(subscription)

    if not matched_models:
        raise HTTPException(status_code=503, detail="No matching ML model available in MLflow")

    latest_model = sorted(matched_models, key=lambda m: int(m.version), reverse=True)[0]
    model_uri = latest_model.source
    model_name = latest_model.name

    inference_url = serve_ml_model(model_name)
    if not inference_url:
        raise HTTPException(status_code=500, detail="Failed to start model serving")

    sub_id = f"sub-{uuid4().hex}"
    subscriptions[sub_id] = subscription

    location_url = f"{API_ROOT}/nnwdaf-mlmodelprovision/v1/subscriptions/{sub_id}"
    response.headers["Location"] = location_url

    imm_rep_flag = subscription.eventReq.immRep if subscription.eventReq else False

    response_payload = {
        "subscriptionId": sub_id,
        "model_uri": model_uri,
        "inference_url": inference_url,
    }

    if imm_rep_flag:
        # 🚀 Generate Immediate Report (Call MLflow Inference)
        try:
            inference_response = requests.post(
                inference_url, json={"inputs": [[1.0, 2.0, 3.0]]}  # Dummy Input Data
            )
            if inference_response.status_code == 200:
                response_payload["immediateReport"] = inference_response.json()
            else:
                response_payload["immediateReport"] = {
                    "status": "failed",
                    "error": "Inference failed"
                }
        except Exception as e:
            print(f"Error calling MLflow inference: {e}")
            response_payload["immediateReport"] = {
                "status": "failed",
                "error": str(e)
            }

    return response_payload

@app.put("/nnwdaf-mlmodelprovision/v1/subscriptions/{subscriptionId}", response_model=NwdafMLModelProvSubsc)
async def update_subscription(subscriptionId: str, updated_subscription: NwdafMLModelProvSubsc):
    if subscriptionId not in subscriptions:
        raise HTTPException(status_code=404, detail="Subscription not found")

    subscriptions[subscriptionId] = updated_subscription

    return updated_subscription

@app.delete("/nnwdaf-mlmodelprovision/v1/subscriptions/{subscriptionId}", status_code=204)
async def delete_subscription(subscriptionId: str, response: Response):
    if subscriptionId not in subscriptions:
        raise HTTPException(status_code=404, detail="Subscription not found")

    del subscriptions[subscriptionId]

    response.status_code = 204
    return

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)