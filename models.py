from pydantic import BaseModel, Field
from typing import Any, List


class ExerciseMetricModel(BaseModel):
    exerciseName: str
    accuracy: float = Field(ge=0, le=100)
    gripStrength: float = Field(ge=0, le=100)
    repsCompleted: int = Field(ge=0)
    targetReps: int = Field(ge=0)
    duration: float = Field(ge=0)
    startTimestamp: str
    endTimestamp: str


class SessionDataModel(BaseModel):
    sessionId: str
    patientId: str
    startTimestamp: str
    endTimestamp: str
    overallAccuracy: float = Field(ge=0, le=100)
    averageGripStrength: float = Field(ge=0, le=100)
    totalDuration: float = Field(ge=0)
    exercises: List[ExerciseMetricModel] = Field(default_factory=list)


class SessionResponse(BaseModel):
    message: str
    sessionId: str


class PatientSessionsResponse(BaseModel):
    patientId: str
    sessionCount: int
    sessions: List[SessionDataModel]


class HealthResponse(BaseModel):
    status: str


class AnalysisResponse(BaseModel):
    patientId: str
    analysis: dict[str, Any]
    aiSummary: str


class ForecastResponse(BaseModel):
    patientId: str
    accuracy_trend: str
    forecast: list[dict[str, Any]]
