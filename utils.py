"""Utility helpers for the fitness activity recognition app."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fitness_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
HISTORY_PATH = BASE_DIR / "workout_history.csv"

FEATURE_FIELDS: List[Dict[str, Any]] = [
    {
        "name": "age",
        "label": "Age",
        "input": "number",
        "default": 30,
        "min_value": 10,
        "max_value": 100,
        "step": 1,
    },
    {
        "name": "gender",
        "label": "Gender",
        "input": "selectbox",
        "options": ["Male", "Female", "Other"],
        "default": "Male",
        "mapping": {"Male": 0, "Female": 1, "Other": 2},
    },
    {
        "name": "height_cm",
        "label": "Height (cm)",
        "input": "number",
        "default": 170,
        "min_value": 100,
        "max_value": 220,
        "step": 1,
    },
    {
        "name": "weight_kg",
        "label": "Weight (kg)",
        "input": "number",
        "default": 70,
        "min_value": 30,
        "max_value": 220,
        "step": 1,
    },
    {
        "name": "heart_rate",
        "label": "Heart Rate (bpm)",
        "input": "number",
        "default": 90,
        "min_value": 40,
        "max_value": 220,
        "step": 1,
    },
    {
        "name": "body_temp_c",
        "label": "Body Temperature (°C)",
        "input": "number",
        "default": 36.6,
        "min_value": 35.0,
        "max_value": 42.0,
        "step": 0.1,
    },
    {
        "name": "duration_min",
        "label": "Duration (minutes)",
        "input": "number",
        "default": 20,
        "min_value": 1,
        "max_value": 300,
        "step": 1,
    },
    {
        "name": "calories_burned",
        "label": "Calories Burned",
        "input": "number",
        "default": 250,
        "min_value": 0,
        "max_value": 5000,
        "step": 1,
    },
    {
        "name": "steps",
        "label": "Steps",
        "input": "number",
        "default": 8000,
        "min_value": 0,
        "max_value": 500000,
        "step": 100,
    },
    {
        "name": "distance_km",
        "label": "Distance (km)",
        "input": "number",
        "default": 5.0,
        "min_value": 0.0,
        "max_value": 100.0,
        "step": 0.1,
    },
]


def load_model() -> Dict[str, Any]:
    """Load the trained model, scaler, and feature names from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    feature_names = list(getattr(model, "feature_names_in_", [field["name"] for field in FEATURE_FIELDS]))
    return {"model": model, "scaler": scaler, "feature_names": feature_names}


def predict_activity(
    model: Any,
    scaler: Any,
    input_values: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
) -> Tuple[str, Optional[float]]:
    """Prepare input features, scale them, and predict an activity label."""
    if model is None or scaler is None:
        raise ValueError("Model and scaler must be loaded before prediction.")

    if feature_names is None:
        feature_names = [field["name"] for field in FEATURE_FIELDS]

    prepared_features: Dict[str, Any] = {}
    for field in FEATURE_FIELDS:
        name = field["name"]
        if name not in input_values:
            continue
        value = input_values[name]
        mapping = field.get("mapping")
        if mapping and isinstance(value, str):
            prepared_features[name] = mapping.get(value.title(), value)
        else:
            try:
                prepared_features[name] = float(value)
            except (TypeError, ValueError):
                prepared_features[name] = value

    missing = [name for name in feature_names if name not in prepared_features]
    if missing:
        raise ValueError(f"Missing feature values for: {', '.join(missing)}")

    feature_frame = pd.DataFrame([prepared_features], columns=feature_names)
    scaled_features = scaler.transform(feature_frame)
    prediction = model.predict(scaled_features)[0]

    confidence: Optional[float] = None
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba(scaled_features)[0]))

    return str(prediction), confidence


def save_workout(workout: Dict[str, Any]) -> Path:
    """Append a workout entry to the workout history CSV file."""
    history_path = HISTORY_PATH
    history_path.touch(exist_ok=True)

    new_entry = pd.DataFrame([workout])
    if history_path.stat().st_size == 0:
        new_entry.to_csv(history_path, index=False)
    else:
        existing = pd.read_csv(history_path)
        combined = pd.concat([existing, new_entry], ignore_index=True)
        combined.to_csv(history_path, index=False)

    return history_path


def load_history() -> pd.DataFrame:
    """Load workout history from the CSV file or return an empty dataframe."""
    if not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size == 0:
        return pd.DataFrame(columns=["exercise_name", "sets", "reps", "duration", "calories_burned", "date"])

    history = pd.read_csv(HISTORY_PATH)
    return history
