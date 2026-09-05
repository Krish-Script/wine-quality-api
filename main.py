# Wine Quality ML API - v2

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from xgboost import XGBClassifier
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import joblib
import numpy as np

from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

DATABASE_URL = os.getenv("DATABASE_URL")
assert DATABASE_URL is not None, "DATABASE_URL environment variable not set"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Database model
class PredictionLog(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    fixed_acidity = Column(Float)
    volatile_acidity = Column(Float)
    citric_acid = Column(Float)
    chlorides = Column(Float)
    total_sulfur_dioxide = Column(Float)
    density = Column(Float)
    sulphates = Column(Float)
    alcohol = Column(Float)
    prediction = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

class DriftLog(Base):
    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer)
    drifted_features = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Load model
app = FastAPI()
model = XGBClassifier()
model.load_model('wine_model.json')
scaler = joblib.load('wine_scaler.pkl')
features = joblib.load('wine_features.pkl')

class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    chlorides: float
    total_sulfur_dioxide: float
    density: float
    sulphates: float
    alcohol: float

TRAINING_STATS = {
    "fixed_acidity":        {"mean": 8.319637,  "std": 1.741096},
    "volatile_acidity":     {"mean": 0.527821,  "std": 0.179060},
    "citric_acid":          {"mean": 0.270976,  "std": 0.194801},
    "chlorides":            {"mean": 0.087467,  "std": 0.047065},
    "total_sulfur_dioxide": {"mean": 46.467792, "std": 32.895324},
    "density":              {"mean": 0.996747,  "std": 0.001887},
    "sulphates":            {"mean": 0.658149,  "std": 0.169507},
    "alcohol":              {"mean": 10.422983, "std": 1.065668}
}

def check_drift(wine_input: dict) -> list:
    drifted = []
    for feature, value in wine_input.items():
        stats = TRAINING_STATS.get(feature)
        if stats:
            lower = stats["mean"] - 3 * stats["std"]
            upper = stats["mean"] + 3 * stats["std"]
            if value < lower or value > upper:
                drifted.append(feature)
    return drifted

@app.get("/")
def root():
    return {"status": "Wine Quality API is running"}

@app.post("/predict")
def predict(wine: WineInput):
    input_data = np.array([[
        wine.fixed_acidity,
        wine.volatile_acidity,
        wine.citric_acid,
        wine.chlorides,
        wine.total_sulfur_dioxide,
        wine.density,
        wine.sulphates,
        wine.alcohol
    ]])

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0]

    result = "High Quality" if prediction == 1 else "Low Quality"
    confidence = round(float(max(probability)) * 100, 2)

    # Check for drift
    wine_dict = wine.model_dump()
    drifted_features = check_drift(wine_dict)
    drift_detected = len(drifted_features) > 0

    db = SessionLocal()

    # Log prediction
    log = PredictionLog(
        fixed_acidity=wine.fixed_acidity,
        volatile_acidity=wine.volatile_acidity,
        citric_acid=wine.citric_acid,
        chlorides=wine.chlorides,
        total_sulfur_dioxide=wine.total_sulfur_dioxide,
        density=wine.density,
        sulphates=wine.sulphates,
        alcohol=wine.alcohol,
        prediction=result,
        confidence=confidence
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    # Log drift if detected
    if drift_detected:
        drift_log = DriftLog(
            prediction_id=log.id,
            drifted_features=", ".join(drifted_features)
        )
        db.add(drift_log)
        db.commit()

    db.close()

    return {
        "prediction": result,
        "confidence": confidence,
        "drift_detected": drift_detected,
        "drifted_features": drifted_features
    }

@app.get("/predictions/history")
def history():
    db = SessionLocal()
    logs = db.query(PredictionLog).order_by(
        PredictionLog.timestamp.desc()
    ).limit(20).all()
    db.close()

    return [
        {
            "id": log.id,
            "prediction": log.prediction,
            "confidence": log.confidence,
            "alcohol": log.alcohol,
            "volatile_acidity": log.volatile_acidity,
            "timestamp": log.timestamp
        }
        for log in logs
    ]

@app.get("/drift/history")
def drift_history():
    db = SessionLocal()
    logs = db.query(DriftLog).order_by(
        DriftLog.timestamp.desc()
    ).limit(20).all()
    db.close()

    return [
        {
            "id": log.id,
            "prediction_id": log.prediction_id,
            "drifted_features": log.drifted_features,
            "timestamp": log.timestamp
        }
        for log in logs
    ]