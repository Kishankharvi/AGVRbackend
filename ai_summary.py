"""
Gemini AI integration for generating clinical user progress summaries.
"""

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"
FALLBACK_MESSAGE = "AI summary unavailable. Please check that GEMINI_API_KEY is set correctly."


async def generate_summary(user_id: str, analysis: dict[str, Any]) -> str:
    """
    Build a clinical prompt from analysis data and call Gemini to produce
    a 3-paragraph progress report. Returns fallback string on any failure.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        return FALLBACK_MESSAGE

    prompt = _build_prompt(user_id, analysis)

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text or FALLBACK_MESSAGE
    except Exception as e:
        return f"AI summary unavailable: {str(e)}"


def _build_prompt(user_id: str, analysis: dict[str, Any]) -> str:
    per_exercise = analysis.get("per_exercise", {})
    exercise_table = _format_exercise_table(per_exercise)

    forecast = analysis.get("forecast", [])
    forecast_text = ", ".join(
        f"{f['session']}: {f['predicted_accuracy']}%"
        for f in forecast
    ) if forecast else "Insufficient data for forecast"

    return f"""You are a physiotherapy AI assistant analysing VR hand rehabilitation data.

User ID: {user_id}
Sessions completed: {analysis.get('session_count', 0)}
Overall accuracy trend: {analysis.get('accuracy_trend', 'unknown')}
Average accuracy: {analysis.get('avg_accuracy', 0):.1f}%
Best accuracy: {analysis.get('best_accuracy', 0):.1f}%
Worst accuracy: {analysis.get('worst_accuracy', 0):.1f}%
Average grip strength: {analysis.get('avg_grip', 0):.1f}

Per-exercise breakdown:
{exercise_table}

Forecasted next 3 sessions: {forecast_text}

Based on this rehabilitation data, write a concise 3-paragraph clinical progress report:

Paragraph 1 - Progress Assessment: Summarise the user's overall rehabilitation progress,
highlighting accuracy trends and grip strength development.

Paragraph 2 - Exercise Analysis: Identify the weakest exercises and provide specific
physiotherapy suggestions for improvement. Reference the per-exercise metrics.

Paragraph 3 - Recovery Outlook: Based on the forecast data, provide a recovery outlook
and recommend adjustments to the rehabilitation programme intensity or focus areas.

Keep the tone professional but accessible. Use specific numbers from the data.
Do not use markdown formatting, bullet points, or headers - write flowing paragraphs only."""


def _format_exercise_table(per_exercise: dict[str, dict]) -> str:
    if not per_exercise:
        return "  No exercise data available."

    lines = []
    for name, metrics in per_exercise.items():
        lines.append(
            f"  - {name}: "
            f"Accuracy={metrics.get('avg_accuracy', 0):.1f}%, "
            f"Grip={metrics.get('avg_grip', 0):.1f}, "
            f"Reps Completion={metrics.get('avg_reps_pct', 0):.1f}%"
        )
    return "\n".join(lines)
