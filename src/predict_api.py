"""FastAPI endpoint for real-time churn predictions."""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.feature_store import get_customer_features
from src.utils import load_config

config = load_config()
app = FastAPI(title="Churn Prediction API")
model = joblib.load(config["model"]["path"])


class PredictionRequest(BaseModel):
    customer_id: str


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    will_churn: bool
    risk_level: str


class FeatureRequest(BaseModel):
    tenure_months: int
    monthly_charges: float
    total_charges: float
    num_support_tickets: int
    num_referrals: int
    avg_charge_per_month: float
    charge_deviation: float
    ticket_rate: float
    total_services: int
    is_high_value: int
    has_premium: int
    contract_month_to_month: int
    contract_one_year: int
    contract_two_year: int
    internet_fiber_optic: int
    internet_dsl: int
    internet_none: int
    online_security: int
    tech_support: int
    streaming_tv: int
    streaming_movies: int
    referral_ticket_ratio: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/customer", response_model=PredictionResponse)
def predict_by_customer(req: PredictionRequest):
    features = get_customer_features(req.customer_id)
    if features is None:
        raise HTTPException(404, f"Customer {req.customer_id} not found")

    df = pd.DataFrame([features])
    prob = float(model.predict_proba(df)[:, 1][0])

    return PredictionResponse(
        customer_id=req.customer_id,
        churn_probability=round(prob, 4),
        will_churn=prob > 0.5,
        risk_level="high" if prob > 0.7 else "medium" if prob > 0.4 else "low",
    )


@app.post("/predict/features", response_model=PredictionResponse)
def predict_by_features(req: FeatureRequest):
    df = pd.DataFrame([req.model_dump()])
    prob = float(model.predict_proba(df)[:, 1][0])

    return PredictionResponse(
        customer_id="ad-hoc",
        churn_probability=round(prob, 4),
        will_churn=prob > 0.5,
        risk_level="high" if prob > 0.7 else "medium" if prob > 0.4 else "low",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config["api"]["host"], port=config["api"]["port"])
