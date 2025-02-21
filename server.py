from fastapi import FastAPI, HTTPException, Response, Header
from pydantic import BaseModel
from typing import Dict
import uvicorn

# Import your Pydantic models
from types import NwdafMLModelProvSubsc

app = FastAPI()

# In-memory storage for subscriptions
subscriptions: Dict[str, NwdafMLModelProvSubsc] = {}

API_ROOT = "http://localhost:8000"  # TODO: Change it for k8s.


@app.post("/subscribe/", status_code=201, response_model=NwdafMLModelProvSubsc)
async def subscribe(subscription: NwdafMLModelProvSubsc, response: Response):
    sub_id = f"sub-{len(subscriptions) + 1}"
    subscriptions[sub_id] = subscription

    location_url = f"{API_ROOT}/nnwdaf-mlmodelprovision/v1/subscriptions/{sub_id}"

    response.headers["Location"] = location_url

    return subscription  # Return the received subscription object as response

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