
import sqlite3
import joblib
import numpy as np

from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


# ==========================================
# 1. APPLICATION SETUP
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
import os

DATABASE = Path(
    os.environ.get("DATABASE_PATH", "/tmp/heartcare.db")
)
MODEL_PATH = BASE_DIR / "heart_disease_model.pkl"

app = FastAPI(title="HeartCare - Heart Disease Prediction")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)

# Load the already-trained model
model = joblib.load(MODEL_PATH)


# ==========================================
# 2. DATABASE CONNECTION
# ==========================================

def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def create_tables():
    connection = get_db()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            administrator TEXT,
            patient_id TEXT,
            patient_name TEXT,
            age REAL,
            sex INTEGER,
            blood_pressure REAL,
            cholesterol REAL,
            bmi REAL,
            smoker INTEGER,
            diabetes INTEGER,
            physical_activity REAL,
            family_history INTEGER,
            stress_level REAL,
            prediction INTEGER,
            probability REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


create_tables()


# ==========================================
# 3. PATIENT INPUT STRUCTURE
# ==========================================

class PatientData(BaseModel):
    patient_id: str = ""
    patient_name: str = ""

    age: float = Field(gt=0)
    sex: int
    blood_pressure: float = Field(gt=0)
    cholesterol: float = Field(gt=0)
    bmi: float = Field(gt=0)

    smoker: int
    diabetes: int

    physical_activity: float
    family_history: int
    stress_level: float


# ==========================================
# 4. DASHBOARD STATISTICS
# ==========================================

def get_dashboard_data():
    connection = get_db()

    total = connection.execute(
        "SELECT COUNT(*) FROM predictions"
    ).fetchone()[0]

    positive = connection.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction = 1"
    ).fetchone()[0]

    negative = connection.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction = 0"
    ).fetchone()[0]

    recent = connection.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
        LIMIT 5
    """).fetchall()

    connection.close()

    return {
        "total_predictions": total,
        "positive_predictions": positive,
        "negative_predictions": negative,
        "recent_predictions": [dict(row) for row in recent]
    }


# ==========================================
# 5. HOME / DASHBOARD
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )


# ==========================================
# 6. DASHBOARD STATISTICS API
# ==========================================

@app.get("/api/stats")
async def dashboard_statistics():
    return get_dashboard_data()


# ==========================================
# 7. PATIENT PREDICTION
# ==========================================

@app.post("/api/predict")
async def predict_heart_disease(patient: PatientData):

    if patient.sex not in [0, 1]:
        raise HTTPException(400, "Invalid sex selection.")

    if patient.smoker not in [0, 1]:
        raise HTTPException(400, "Invalid smoking selection.")

    if patient.diabetes not in [0, 1]:
        raise HTTPException(400, "Invalid diabetes selection.")

    if patient.family_history not in [0, 1]:
        raise HTTPException(400, "Invalid family history selection.")

    # Keep the same feature order as the previous application
    patient_data = np.array([[
        patient.age,
        patient.sex,
        patient.blood_pressure,
        patient.cholesterol,
        patient.bmi,
        patient.smoker,
        patient.diabetes,
        patient.physical_activity,
        patient.family_history,
        patient.stress_level
    ]], dtype=float)

    # Predict using the existing trained model
    prediction = int(model.predict(patient_data)[0])

    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(patient_data)[0]
        classes = list(model.classes_)

        if 1 in classes:
            probability = float(
                probabilities[classes.index(1)] * 100
            )

    result = (
        "Heart Disease Detected"
        if prediction == 1
        else "No Heart Disease Detected"
    )

    # Save patient information and prediction
    connection = get_db()

    cursor = connection.execute("""
        INSERT INTO predictions (
            patient_id,
            patient_name,
            age,
            sex,
            blood_pressure,
            cholesterol,
            bmi,
            smoker,
            diabetes,
            physical_activity,
            family_history,
            stress_level,
            prediction,
            probability
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient.patient_id.strip(),
        patient.patient_name.strip(),
        patient.age,
        patient.sex,
        patient.blood_pressure,
        patient.cholesterol,
        patient.bmi,
        patient.smoker,
        patient.diabetes,
        patient.physical_activity,
        patient.family_history,
        patient.stress_level,
        prediction,
        probability
    ))

    record_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return {
        "success": True,
        "record_id": record_id,
        "patient_name": patient.patient_name,
        "prediction": prediction,
        "result": result,
        "probability": probability,
        "message": "Prediction completed and saved successfully."
    }


# ==========================================
# 8. PATIENT HISTORY
# ==========================================

@app.get("/api/history")
async def patient_history():

    connection = get_db()

    records = connection.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
    """).fetchall()

    connection.close()

    return {
        "records": [dict(row) for row in records]
    }


# ==========================================
# 9. VIEW ONE PATIENT RECORD
# ==========================================

@app.get("/api/history/{record_id}")
async def patient_record(record_id: int):

    connection = get_db()

    record = connection.execute("""
        SELECT *
        FROM predictions
        WHERE id = ?
    """, (record_id,)).fetchone()

    connection.close()

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Patient record not found."
        )

    return dict(record)


# ==========================================
# 10. ANALYTICS
# ==========================================

@app.get("/api/analytics")
async def analytics():

    connection = get_db()

    total = connection.execute(
        "SELECT COUNT(*) FROM predictions"
    ).fetchone()[0]

    positive = connection.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction = 1"
    ).fetchone()[0]

    negative = connection.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction = 0"
    ).fetchone()[0]

    connection.close()

    return {
        "total": total,
        "heart_disease": positive,
        "no_heart_disease": negative
    }


# ==========================================
# 11. ABOUT / MODEL INFORMATION
# ==========================================

@app.get("/api/about")
async def about():

    return {
        "system": "HeartCare",
        "purpose": "Heart Disease Prediction",
        "algorithm": "Logistic Regression",
        "dataset": "Hospital Synthetic Dataset",
        "target": "heart_disease",
        "model_loaded": model is not None
    }


# ==========================================
# 12. APPLICATION STATUS
# ==========================================

@app.get("/health")
async def health():

    return {
        "status": "HeartCare is running",
        "model_loaded": model is not None,
        "database": "Connected"
    }

