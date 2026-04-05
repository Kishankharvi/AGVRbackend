"""
FastAPI backend for AGVRSystem VR Hand Rehabilitation app.
Stores session data in SQLite, provides REST endpoints matching Unity APIManager.

Run: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import json
import os
import sqlite3
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    AnalysisResponse,
    ExerciseMetricModel,
    ForecastResponse,
    HealthResponse,
    PatientSessionsResponse,
    SessionDataModel,
    SessionResponse,
)
from analysis import analyse_patient
from ai_summary import generate_summary

DB_PATH = os.environ.get("AGVR_DB_PATH", "agvr_sessions.db")


def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            start_timestamp TEXT NOT NULL,
            end_timestamp TEXT NOT NULL,
            overall_accuracy REAL NOT NULL,
            average_grip_strength REAL NOT NULL,
            total_duration REAL NOT NULL,
            exercises_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_patient_id ON sessions(patient_id)
    """)
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="AGVRSystem Session API",
    description="REST API for VR Hand Rehabilitation session storage and AI analysis",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _fetch_patient_sessions_raw(patient_id: str) -> list[dict]:
    """Fetch raw session dicts from the database for a given patient."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM sessions WHERE patient_id = ? ORDER BY start_timestamp ASC",
        (patient_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    sessions = []
    for row in rows:
        exercises = json.loads(row["exercises_json"])
        sessions.append({
            "sessionId": row["session_id"],
            "patientId": row["patient_id"],
            "startTimestamp": row["start_timestamp"],
            "endTimestamp": row["end_timestamp"],
            "overallAccuracy": row["overall_accuracy"],
            "averageGripStrength": row["average_grip_strength"],
            "totalDuration": row["total_duration"],
            "exercises": exercises,
        })

    return sessions


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.get("/api/patients", response_model=List[str])
async def get_all_patients():
    """Retrieve all unique patient IDs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT patient_id FROM sessions ORDER BY patient_id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]


@app.post("/api/session", response_model=SessionResponse, status_code=201)
async def create_session(session: SessionDataModel):
    """
    Store a completed rehabilitation session.
    Called by Unity APIManager.PostSession().
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session.sessionId,))
        if cursor.fetchone():
            conn.close()
            raise HTTPException(
                status_code=409,
                detail=f"Session {session.sessionId} already exists"
            )

        exercises_json = json.dumps([e.model_dump() for e in session.exercises])

        cursor.execute(
            """
            INSERT INTO sessions
            (session_id, patient_id, start_timestamp, end_timestamp,
             overall_accuracy, average_grip_strength, total_duration, exercises_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.sessionId,
                session.patientId,
                session.startTimestamp,
                session.endTimestamp,
                session.overallAccuracy,
                session.averageGripStrength,
                session.totalDuration,
                exercises_json,
            ),
        )
        conn.commit()
    except HTTPException:
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

    return SessionResponse(
        message="Session stored successfully",
        sessionId=session.sessionId
    )


@app.get("/api/patient/{patient_id}", response_model=PatientSessionsResponse)
async def get_patient_sessions(patient_id: str):
    """Retrieve all sessions for a given patient (newest first)."""
    raw = _fetch_patient_sessions_raw(patient_id)

    sessions: List[SessionDataModel] = []
    for s in reversed(raw):
        exercises = [ExerciseMetricModel(**e) for e in s["exercises"]]
        sessions.append(
            SessionDataModel(
                sessionId=s["sessionId"],
                patientId=s["patientId"],
                startTimestamp=s["startTimestamp"],
                endTimestamp=s["endTimestamp"],
                overallAccuracy=s["overallAccuracy"],
                averageGripStrength=s["averageGripStrength"],
                totalDuration=s["totalDuration"],
                exercises=exercises,
            )
        )

    return PatientSessionsResponse(
        patientId=patient_id,
        sessionCount=len(sessions),
        sessions=sessions,
    )


@app.get("/api/patient/{patient_id}/analyse", response_model=AnalysisResponse)
async def analyse_patient_endpoint(patient_id: str):
    """Run ML analysis and Gemini AI summary for a patient."""
    raw = _fetch_patient_sessions_raw(patient_id)

    if not raw:
        raise HTTPException(status_code=404, detail=f"No sessions found for patient {patient_id}")

    analysis = analyse_patient(raw)
    ai_summary = await generate_summary(patient_id, analysis)

    return AnalysisResponse(
        patientId=patient_id,
        analysis=analysis,
        aiSummary=ai_summary,
    )


@app.get("/api/patient/{patient_id}/forecast", response_model=ForecastResponse)
async def forecast_patient_endpoint(patient_id: str):
    """Return accuracy forecast and trend for a patient (lightweight, no AI call)."""
    raw = _fetch_patient_sessions_raw(patient_id)

    if not raw:
        raise HTTPException(status_code=404, detail=f"No sessions found for patient {patient_id}")

    analysis = analyse_patient(raw)

    return ForecastResponse(
        patientId=patient_id,
        accuracy_trend=analysis["accuracy_trend"],
        forecast=analysis["forecast"],
    )
