import logging
import os
from typing import Dict

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from model_utils import DEFAULT_FEATURE_ORDER, predict_anomaly

API_KEY = os.getenv("API_KEY", "dev-key")
LOGGER = logging.getLogger("anomaly-backend")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

app = FastAPI(title="Anomaly Detection Backend", version="1.0.0")


class Report(BaseModel):
    cpu: float = Field(..., description="CPU percent")
    memory: float = Field(..., description="Memory percent")
    connections: int = Field(..., description="Number of network connections")
    bytes_sent: float = Field(..., description="Bytes sent")
    bytes_recv: float = Field(..., description="Bytes received")


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/report")
async def receive_report(report: Report, _: None = Depends(verify_api_key)) -> Dict[str, float | bool]:
    features = report.model_dump()
    for key in DEFAULT_FEATURE_ORDER:
        if key not in features:
            raise HTTPException(status_code=400, detail=f"Missing feature: {key}")

    result = predict_anomaly(features)
    LOGGER.info(
        "Processed report | features=%s | decision=%.4f | threshold=%.4f | anomaly=%s",
        features,
        result["decision_value"],
        result["threshold"],
        result["is_anomaly"],
    )
    return {
        "is_anomaly": result["is_anomaly"],
        "anomaly_score": result["anomaly_score"],
        "decision_value": result["decision_value"],
        "threshold": result["threshold"],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
