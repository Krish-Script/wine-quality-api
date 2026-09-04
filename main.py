from fastapi import FastAPI
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
    allow_methods=["*"],
    allow_headers=["*"],
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

    # Log to database
    db = SessionLocal()
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
    db.close()

    return {
        "prediction": result,
        "confidence": confidence
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