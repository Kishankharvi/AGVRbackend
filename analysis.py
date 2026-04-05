"""
Pure-Python analysis module for user rehabilitation session data.
Computes per-exercise metrics, accuracy trends, and ML-based forecasting.
"""

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression

SLOPE_IMPROVING_THRESHOLD = 1.0
SLOPE_DECLINING_THRESHOLD = -1.0
FORECAST_HORIZON = 3
MIN_SESSIONS_FOR_FORECAST = 2
ACCURACY_FLOOR = 0.0
ACCURACY_CEILING = 100.0


def analyse_user(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analyse a list of session dicts (matching SessionDataModel schema) and return
    structured metrics including per-exercise breakdown, trend classification,
    and a linear-regression accuracy forecast.
    """
    session_count = len(sessions)

    if session_count == 0:
        return _empty_analysis()

    sorted_sessions = sorted(sessions, key=lambda s: s.get("startTimestamp", ""))

    accuracies = [s["overallAccuracy"] for s in sorted_sessions]
    grips = [s["averageGripStrength"] for s in sorted_sessions]

    avg_accuracy = float(np.mean(accuracies))
    avg_grip = float(np.mean(grips))
    best_accuracy = float(np.max(accuracies))
    worst_accuracy = float(np.min(accuracies))

    per_exercise = _compute_per_exercise(sorted_sessions)
    accuracy_trend, forecast = _compute_forecast(accuracies)

    return {
        "session_count": session_count,
        "avg_accuracy": round(avg_accuracy, 1),
        "avg_grip": round(avg_grip, 1),
        "best_accuracy": round(best_accuracy, 1),
        "worst_accuracy": round(worst_accuracy, 1),
        "accuracy_trend": accuracy_trend,
        "per_exercise": per_exercise,
        "forecast": forecast,
    }


def _empty_analysis() -> dict[str, Any]:
    return {
        "session_count": 0,
        "avg_accuracy": 0.0,
        "avg_grip": 0.0,
        "best_accuracy": 0.0,
        "worst_accuracy": 0.0,
        "accuracy_trend": "insufficient data",
        "per_exercise": {},
        "forecast": [],
    }


def _compute_per_exercise(sorted_sessions: list[dict]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict]] = defaultdict(list)

    for session in sorted_sessions:
        for exercise in session.get("exercises", []):
            name = exercise.get("exerciseName", "Unknown")
            groups[name].append(exercise)

    result = {}
    for name, entries in groups.items():
        acc_values = [e["accuracy"] for e in entries]
        grip_values = [e["gripStrength"] for e in entries]
        reps_pct_values = [
            (e["repsCompleted"] / e["targetReps"] * 100.0) if e.get("targetReps", 0) > 0 else 0.0
            for e in entries
        ]

        result[name] = {
            "avg_accuracy": round(float(np.mean(acc_values)), 1),
            "avg_grip": round(float(np.mean(grip_values)), 1),
            "avg_reps_pct": round(float(np.mean(reps_pct_values)), 1),
        }

    return result


def _compute_forecast(accuracies: list[float]) -> tuple[str, list[dict[str, Any]]]:
    n = len(accuracies)

    if n < MIN_SESSIONS_FOR_FORECAST:
        return "insufficient data", []

    X = np.arange(n).reshape(-1, 1)
    y = np.array(accuracies)

    model = LinearRegression()
    model.fit(X, y)

    slope = float(model.coef_[0])

    if slope > SLOPE_IMPROVING_THRESHOLD:
        trend = "improving"
    elif slope < SLOPE_DECLINING_THRESHOLD:
        trend = "declining"
    else:
        trend = "stable"

    future_indices = np.arange(n, n + FORECAST_HORIZON).reshape(-1, 1)
    predictions = model.predict(future_indices)

    forecast = []
    for i, pred in enumerate(predictions):
        clamped = float(np.clip(pred, ACCURACY_FLOOR, ACCURACY_CEILING))
        forecast.append({
            "session": f"S+{i + 1}",
            "predicted_accuracy": round(clamped, 1),
        })

    return trend, forecast
