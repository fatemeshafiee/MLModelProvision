import requests
from fastapi import FastAPI, HTTPException, Response, Header
from pydantic import BaseModel
from typing import Dict
import uvicorn
import mlflow
from uuid import uuid4
import time
from schemes import *
from mLFlowAPIs import *
from mLFlowAPIs import app

from fastapi_utils.tasks import repeat_every
# In-memory storage for subscriptions
import datetime 
subscriptions: Dict[str, NwdafMLModelProvSubsc] = {}
last_report_times: Dict[str, datetime.datetime] = {}

API_ROOT = "http://localhost:8000"  # TODO: Change it for k8s.
MLFLOW_TRACKING_URI = "http://mlflow-svc:5000"
MLFLOW_SERVE_URI = "http://mlflow-svc:5000/models/{model_name}/versions/{model_version}/invocations"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def process_scheduled_notifications():

    global last_report_times
    current_time = datetime.datetime.utcnow()

    for sub_id, subscription in subscriptions.items():
        notif_uri = subscription.notifUri
        notif_correlation_id = subscription.notifCorreId
        reporting_interval = subscription.eventReq.repPeriod if subscription.eventReq else None

        last_report_time = last_report_times.get(sub_id, None)

        if not reporting_interval:
            continue

        if last_report_time and (current_time - last_report_time).total_seconds() < reporting_interval:
            continue

        search_results = search_mlflow_models(subscription.mLEventSubscs)

        event_notifs = []
        for event_sub in subscription.mLEventSubscs:
            event = event_sub.mLEvent
            model_info = search_results.get(event)

            if model_info and model_info["status"] in ["found", "registered"]:
                model_url = model_info["mlflow_model_url"]
                model_version = model_info["model_version"]

                event_notifs.append(MLEventNotif(
                    event=event,
                    mLFileAddr=MLModelAddr(mLModelUrl=model_url),
                    modelUniqueId=int(model_version)
                ))

        if not event_notifs:
            continue  # No new models to notify about

        notification_payload = NwdafMLModelProvNotif(
            eventNotifs=event_notifs,
            subscriptionId=sub_id
        )

        try:
            response = requests.post(
                notif_uri, json=notification_payload.dict(), timeout=5
            )

            if response.status_code == 200:
                last_report_times[sub_id] = current_time
                print(f"✅ Notification sent successfully to {notif_uri}")
            else:
                print(f"⚠️ Failed to send notification to {notif_uri}, Status Code: {response.status_code}")

        except requests.RequestException as e:
            print(f"❌ Error sending notification to {notif_uri}: {e}")

def search_mlflow_models(event_subscriptions: List[MLEventSubscription]) -> Dict[str, Any]:
    results = {}
    client = mlflow.tracking.MlflowClient()

    for sub in event_subscriptions:
        tags = generate_mlflow_tags(
            mLEvent=sub.mLEvent,
            event_filter=sub.mLEventFilter,
            inference_data=sub.inferDataForModel,
            target_ue=sub.tgtUe
        )

        query_str = ""
        query_parts = []
        for tag_key, tag_value in tags.items():
            logger.info("Tag key: %s (type: %s), Tag value: %s (type: %s)",
                tag_key, type(tag_key), tag_value, type(tag_value))
        for key, value in tags.items():
            if value is not None and str(value) != "None":
                cleaned_value = str(value).replace("'", "")
                query_parts.append(f"tag.{key}='{cleaned_value}'")
        query_str = " AND ".join(query_parts)
        logger.info("the query_str is %s", query_str)

        try:
            models = client.search_model_versions(query_str)

            if models:
                latest_model = sorted(models, key=lambda m: int(m.version), reverse=True)[0]
                results[str(sub.mLEvent)] = {
                    "status": "found",
                    "mlflow_model_url": MLFLOW_SERVE_URI.format(
                        model_name=latest_model.name, model_version=latest_model.version
                    ),
                    "model_version": latest_model.version
                }
            else:
                # No registered model found, check for logged models
                runs = client.search_runs(experiment_ids=["0"], order_by=["start_time DESC"])
                run_id = None

                for run in runs:
                    run_tags = run.data.tags
                    if all(f"tag.{k}" in run_tags and run_tags[f"tag.{k}"] == v for k, v in tags.items()):
                        run_id = run.info.run_id
                        break

                if run_id:
                    model_uri = f"runs:/{run_id}/model"
                    registered_model = mlflow.register_model(model_uri, str(sub.mLEvent))

                    time.sleep(5)

                    latest_version = client.get_latest_versions(str(sub.mLEvent), stages=["None"])[0].version

                    client.transition_model_version_stage(
                        name=str(sub.mLEvent),
                        version=latest_version,
                        stage="Production"
                    )

                    results[str(sub.mLEvent)] = {
                        "status": "registered",
                        "mlflow_model_url": MLFLOW_SERVE_URI.format(
                            model_name=str(sub.mLEvent), model_version=latest_version
                        ),
                        "model_version": latest_version
                    }
                else:
                    results[str(sub.mLEvent)] = {"status": "not_found"}

        except Exception as e:
            print(f"MLflow query failed for event {sub.mLEvent}: {e}")
            results[sub.mLEvent] = {"status": "error", "error_message": str(e)}

    return results


@app.post("/subscribe/", status_code=201, response_model=NwdafMLModelProvSubsc)
async def subscribe(subscription: NwdafMLModelProvSubsc, response: Response):
    sub_id = f"sub-{uuid4().hex}"
    subscriptions[sub_id] = subscription

    location_url = f"http://localhost:8000/nnwdaf-mlmodelprovision/v1/subscriptions/{sub_id}"
    response.headers["Location"] = location_url

    fail_event_reports = []
    ml_event_notifs = []

    imm_rep_flag = subscription.eventReq.immRep if subscription.eventReq else False

    search_results = search_mlflow_models(subscription.mLEventSubscs)
    logger.info("having the search_results")

    for event_sub in subscription.mLEventSubscs:
        event = event_sub.mLEvent
        model_info = search_results.get(str(event))
        logger.info("in the loop line 178")
        if model_info and model_info["status"] in ["found", "registered"]:
            model_url = model_info["mlflow_model_url"]
            model_version = model_info["model_version"]
            logger.info("notif is created")
            if imm_rep_flag:
                ml_event_notifs.append(MLEventNotif(
                    event=event,
                    mLFileAddr=MLModelAddr(mLModelUrl=model_url),
                    modelUniqueId=int(model_version)
                ))
                current_time = datetime.datetime.utcnow()
                last_report_times[sub_id] = current_time

        elif model_info and model_info["status"] == "not_found":
            fail_event_reports.append(FailureEventInfoForMLModel(
                event=event,
                failureCode=FailureCode(failure_code="UNAVAILABLE_ML_MODEL")
            ))

    subscription.failEventReports = fail_event_reports if fail_event_reports else None
    subscription.mLEventNotifs = ml_event_notifs if ml_event_notifs else None

    return subscription

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



@app.on_event("startup")
@repeat_every(seconds=1)  # Runs every 1 second
def scheduled_notification_task():
    process_scheduled_notifications()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
