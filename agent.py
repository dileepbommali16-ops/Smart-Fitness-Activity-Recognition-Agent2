"""
agent.py
--------
Smart Fitness AI Agent.

This module turns the existing prediction model, recommendation engine,
and workout tracker (see utils.py / recommendation.py) into a single
conversational "agent" that can:

  1. Understand what the user wants (predict / log workout / get advice /
     see progress / just chat) from a free-text message.
  2. Ask follow-up questions to fill in any missing information
     (slot-filling), instead of forcing the user to use separate forms.
  3. Decide the next action on its own and call the right underlying
     function (predict_activity, save_workout, recommendation lookup,
     history summary).
  4. Reply in natural language with the result.

No external LLM / API key is required, so this keeps working on a free
Streamlit Cloud deployment exactly like the rest of the app. If you later
want to plug in a real LLM (OpenAI / Anthropic) for richer conversation,
see `generate_smalltalk_reply()` at the bottom -- that is the single
place you'd swap in an API call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from recommendation import (
    DEFAULT_RECOMMENDATION,
    EXERCISE_DESCRIPTIONS,
    EXERCISE_RECOMMENDATIONS,
)
from utils import FEATURE_FIELDS, load_history, load_model, predict_activity, save_workout

# --------------------------------------------------------------------------
# Slot definitions
# --------------------------------------------------------------------------

PREDICT_FIELD_NAMES: List[str] = [f["name"] for f in FEATURE_FIELDS]

# Keywords (and a fallback "number + unit" pattern) used to pull each slot
# value straight out of a free-text sentence, e.g. "I'm 25, weigh 70kg,
# heart rate 110, ran for 30 minutes and burned 300 calories".
FIELD_EXTRACTORS: Dict[str, Dict[str, Any]] = {
    "age": {"keywords": [r"age"], "unit": None},
    "gender": {"keywords": [], "unit": None},  # handled separately (text match)
    "height_cm": {"keywords": [r"height"], "unit": r"(\d+(?:\.\d+)?)\s*cm\b"},
    "weight_kg": {"keywords": [r"weight", r"weigh"], "unit": r"(\d+(?:\.\d+)?)\s*kg\b"},
    "heart_rate": {"keywords": [r"heart\s*rate", r"\bhr\b"], "unit": r"(\d+(?:\.\d+)?)\s*bpm\b"},
    "body_temp_c": {"keywords": [r"temp(?:erature)?"], "unit": r"(\d+(?:\.\d+)?)\s*(?:°c|c\b)"},
    "duration_min": {"keywords": [r"duration"], "unit": r"(\d+(?:\.\d+)?)\s*min"},
    "calories_burned": {"keywords": [r"calories", r"kcal"], "unit": r"(\d+(?:\.\d+)?)\s*kcal\b"},
    "steps": {"keywords": [r"steps"], "unit": r"(\d+(?:\.\d+)?)\s*steps\b"},
    "distance_km": {"keywords": [r"distance"], "unit": r"(\d+(?:\.\d+)?)\s*km\b"},
}

GENDER_WORDS = {"male": "Male", "female": "Female", "other": "Other"}

FIELD_LABELS: Dict[str, str] = {f["name"]: f["label"] for f in FEATURE_FIELDS}


def _extract_number(text: str, keywords: List[str], unit_pattern: Optional[str]) -> Optional[float]:
    for kw in keywords:
        match = re.search(rf"{kw}\D{{0,6}}?(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    if unit_pattern:
        match = re.search(unit_pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def extract_prediction_slots(text: str) -> Dict[str, Any]:
    """Pull as many prediction fields as possible out of a free-text message."""
    found: Dict[str, Any] = {}

    for word, canonical in GENDER_WORDS.items():
        if re.search(rf"\b{word}\b", text, flags=re.IGNORECASE):
            found["gender"] = canonical
            break

    for name, spec in FIELD_EXTRACTORS.items():
        if name == "gender":
            continue
        value = _extract_number(text, spec["keywords"], spec["unit"])
        if value is not None:
            found[name] = value

    return found


def extract_workout_slots(text: str) -> Dict[str, Any]:
    """Pull workout-log fields (exercise name, sets, reps, duration, calories, date)."""
    found: Dict[str, Any] = {}

    name_match = re.search(
        r"(?:did|logged?|log|record(?:ed)?|add)\s+([a-zA-Z\- ]+?)(?:\s+for|\s+\d|,|\.|$)",
        text,
        flags=re.IGNORECASE,
    )
    FILLER_WORDS = {"workout", "exercise", "session", "a workout", "my workout"}
    if name_match:
        candidate = name_match.group(1).strip().title()
        if candidate and candidate.lower() not in FILLER_WORDS:
            found["exercise_name"] = candidate

    patterns = {
        "sets": r"(\d+)\s*sets?\b",
        "reps": r"(\d+)\s*reps?\b",
        "duration": r"(\d+(?:\.\d+)?)\s*(?:min|minutes?)\b",
        "calories_burned": r"(\d+(?:\.\d+)?)\s*(?:kcal|cal(?:ories)?)\b",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            found[name] = float(match.group(1))

    found.setdefault("date", str(date.today()))
    return found


# --------------------------------------------------------------------------
# Intent detection
# --------------------------------------------------------------------------

INTENT_KEYWORDS = {
    "predict": ["predict", "which exercise", "what exercise", "recognize", "detect", "identify my"],
    "log_workout": ["log", "save workout", "record", "add workout", "i did", "i finished", "just did"],
    "recommend": ["recommend", "suggestion", "tips", "advice", "how should i train", "guide me"],
    "history": ["history", "progress", "stats", "summary", "how many workouts", "dashboard"],
    "greet": ["hi", "hello", "hey", "namaste", "hai", "good morning", "good evening"],
}


def detect_intent(message: str) -> str:
    text = message.lower().strip()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return intent
    return "smalltalk"


# --------------------------------------------------------------------------
# Agent state (slot-filling memory for a multi-turn conversation)
# --------------------------------------------------------------------------

@dataclass
class AgentState:
    pending_intent: Optional[str] = None
    slots: Dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.pending_intent = None
        self.slots = {}


# --------------------------------------------------------------------------
# The Agent
# --------------------------------------------------------------------------

class FitnessAgent:
    """Conversational wrapper around the model, tracker, and recommendations."""

    def __init__(self) -> None:
        self._model_bundle: Optional[Dict[str, Any]] = None
        self._model_error: Optional[str] = None

    # ---- model loading -----------------------------------------------
    def _ensure_model(self) -> bool:
        if self._model_bundle is not None:
            return True
        if self._model_error is not None:
            return False
        try:
            self._model_bundle = load_model()
            return True
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as a chat message
            self._model_error = str(exc)
            return False

    # ---- main entry point ----------------------------------------------
    def respond(self, message: str, state: AgentState) -> str:
        """Decide what to do with `message` given the current AgentState, act, and reply."""
        message = message.strip()
        if not message:
            return "Cheppandi, meeku em kavali? (predict / log workout / recommendation / progress)"

        # If we're in the middle of collecting slots for a pending action, keep filling them.
        if state.pending_intent == "predict":
            return self._continue_predict(message, state)
        if state.pending_intent == "log_workout":
            return self._continue_log_workout(message, state)

        intent = detect_intent(message)

        if intent == "greet":
            return (
                "Hi! Nenu nee Smart Fitness AI Agent 🤖. Nenu ee panulu cheyagalanu:\n"
                "- **Predict**: your body/movement metrics tho exercise em ani cheptanu\n"
                "- **Log workout**: 'I did squats, 3 sets 12 reps 20 min 200 calories' laga cheppandi\n"
                "- **Recommend**: 'give me tips for running' ani adagandi\n"
                "- **Progress**: 'show my stats' ani adagandi\n"
                "Em kavali cheppandi!"
            )

        if intent == "predict":
            state.pending_intent = "predict"
            state.slots = extract_prediction_slots(message)
            return self._continue_predict(message, state, already_parsed=True)

        if intent == "log_workout":
            state.pending_intent = "log_workout"
            state.slots = extract_workout_slots(message)
            return self._continue_log_workout(message, state, already_parsed=True)

        if intent == "recommend":
            return self._handle_recommend(message)

        if intent == "history":
            return self._handle_history()

        return self._smalltalk(message)

    # ---- PREDICT flow ----------------------------------------------------
    def _continue_predict(self, message: str, state: AgentState, already_parsed: bool = False) -> str:
        if not already_parsed:
            newly_found = extract_prediction_slots(message)
            state.slots.update(newly_found)

            # If nothing was matched by keyword/unit, treat a bare reply as the
            # answer to whichever field we just asked about (e.g. user just
            # types "170" after being asked for height, or "male" for gender).
            if not newly_found:
                missing_now = [name for name in PREDICT_FIELD_NAMES if name not in state.slots]
                if missing_now:
                    next_field = missing_now[0]
                    bare_number = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", message)
                    if next_field == "gender":
                        for word, canonical in GENDER_WORDS.items():
                            if word in message.lower():
                                state.slots["gender"] = canonical
                                break
                    elif bare_number:
                        state.slots[next_field] = float(bare_number.group(1))

        missing = [name for name in PREDICT_FIELD_NAMES if name not in state.slots]
        if missing:
            next_field = missing[0]
            label = FIELD_LABELS.get(next_field, next_field)
            if next_field == "gender":
                return f"{label} enti? (Male / Female / Other)"
            return f"{label} entha? (oka number cheppandi)"

        # All slots collected -> run the model.
        if not self._ensure_model():
            state.reset()
            return (
                f"Model files dorakaledu ({self._model_error}). "
                "fitness_model.pkl mariyu scaler.pkl repo root lo unnayo check cheyandi."
            )

        try:
            prediction, confidence = predict_activity(
                self._model_bundle["model"],
                self._model_bundle["scaler"],
                state.slots,
                feature_names=self._model_bundle["feature_names"],
            )
        except Exception as exc:  # noqa: BLE001
            state.reset()
            return f"Prediction fail ayyindi: {exc}"

        description = EXERCISE_DESCRIPTIONS.get(prediction, "")
        conf_text = f" (confidence {confidence * 100:.1f}%)" if confidence is not None else ""
        state.reset()
        reply = f"🎯 Predicted exercise: **{prediction}**{conf_text}."
        if description:
            reply += f"\n{description}"
        reply += "\n\n'recommend' ani cheppandi ee exercise ki tips kosam, leda 'log workout' ani cheppi save cheskovachu."
        return reply

    # ---- LOG WORKOUT flow -------------------------------------------------
    def _continue_log_workout(self, message: str, state: AgentState, already_parsed: bool = False) -> str:
        required = ["exercise_name", "sets", "reps", "duration", "calories_burned"]

        if not already_parsed:
            newly_found = extract_workout_slots(message)
            state.slots.update(newly_found)

            useful_fields = {k: v for k, v in newly_found.items() if k != "date"}
            if not useful_fields:
                missing_now = [name for name in required if name not in state.slots]
                if missing_now:
                    next_field = missing_now[0]
                    bare_number = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", message)
                    if next_field == "exercise_name" and not bare_number:
                        state.slots["exercise_name"] = message.strip().title()
                    elif bare_number and next_field != "exercise_name":
                        state.slots[next_field] = float(bare_number.group(1))

        state.slots.setdefault("date", str(date.today()))
        missing = [name for name in required if name not in state.slots]
        if missing:
            prompts = {
                "exercise_name": "Exercise peru enti?",
                "sets": "Ela sets chesaru?",
                "reps": "Prathi set ki ela reps?",
                "duration": "Enni minutes chesaru?",
                "calories_burned": "Enni calories burn ayyayi (approx ayina parledu)?",
            }
            return prompts[missing[0]]

        try:
            payload = {
                "exercise_name": str(state.slots["exercise_name"]).strip() or "Unknown",
                "sets": int(state.slots["sets"]),
                "reps": int(state.slots["reps"]),
                "duration": float(state.slots["duration"]),
                "calories_burned": float(state.slots["calories_burned"]),
                "date": state.slots.get("date", str(date.today())),
            }
            save_workout(payload)
        except Exception as exc:  # noqa: BLE001
            state.reset()
            return f"Workout save cheyadam fail ayyindi: {exc}"

        state.reset()
        return (
            f"✅ Saved! **{payload['exercise_name']}** — {payload['sets']} sets x {payload['reps']} reps, "
            f"{payload['duration']:.0f} min, {payload['calories_burned']:.0f} kcal ({payload['date']}).\n"
            "'show my stats' ani cheppi progress chudochu."
        )

    # ---- RECOMMEND ---------------------------------------------------------
    def _handle_recommend(self, message: str) -> str:
        text = message.lower()
        matched_exercise: Optional[str] = None
        for exercise_name in EXERCISE_RECOMMENDATIONS.keys():
            if exercise_name.lower() in text:
                matched_exercise = exercise_name
                break

        if matched_exercise is None:
            options = ", ".join(list(EXERCISE_RECOMMENDATIONS.keys())[:8])
            return (
                "Ee exercise ki tips kavalo cheppandi (e.g. 'tips for running'). "
                f"Available options: {options}"
            )

        rec = EXERCISE_RECOMMENDATIONS.get(matched_exercise, DEFAULT_RECOMMENDATION)
        description = EXERCISE_DESCRIPTIONS.get(matched_exercise, "")
        lines = [f"💡 **{matched_exercise} recommendations**"]
        if description:
            lines.append(description)
        warm_up = rec.get("warm_up", [])
        if warm_up:
            lines.append("Warm-up: " + "; ".join(warm_up))
        lines.append("Sets & reps: " + str(rec.get("sets_reps", DEFAULT_RECOMMENDATION["sets_reps"])))
        lines.append("Rest time: " + str(rec.get("rest_time", DEFAULT_RECOMMENDATION["rest_time"])))
        lines.append("Hydration: " + str(rec.get("hydration", DEFAULT_RECOMMENDATION["hydration"])))
        lines.append("Nutrition: " + str(rec.get("nutrition", DEFAULT_RECOMMENDATION["nutrition"])))
        lines.append("Recovery: " + str(rec.get("recovery", DEFAULT_RECOMMENDATION["recovery"])))
        return "\n".join(lines)

    # ---- HISTORY / PROGRESS -------------------------------------------------
    def _handle_history(self) -> str:
        history = load_history()
        if history.empty:
            return "Inka workout logs ledu. 'log workout' ani cheppi first entry add cheyandi."

        total_workouts = len(history)
        total_calories = float(history["calories_burned"].sum())
        total_duration = float(history["duration"].sum())
        top_exercise = (
            history["exercise_name"].value_counts().idxmax()
            if "exercise_name" in history.columns and not history.empty
            else "N/A"
        )
        return (
            f"📊 **Progress summary**\n"
            f"- Total workouts: {total_workouts}\n"
            f"- Total calories burned: {total_calories:.0f} kcal\n"
            f"- Total duration: {total_duration:.0f} min\n"
            f"- Most frequent exercise: {top_exercise}"
        )

    # ---- fallback smalltalk ---------------------------------------------
    def _smalltalk(self, message: str) -> str:
        return generate_smalltalk_reply(message)


def generate_smalltalk_reply(message: str) -> str:
    """
    Rule-based fallback for anything that isn't predict/log/recommend/history.

    To upgrade this into a full LLM-powered agent later, replace the body of
    this function with a call to your preferred chat model (OpenAI, Anthropic,
    etc.), passing `message` as the prompt. Keeping it isolated here means the
    rest of the agent (slot-filling, model calls, tracker) doesn't need to change.
    """
    tips = [
        "Prathi roju konchem aina move avvadam consistency ki key.",
        "Workout tarvata 20-30 nims lo protein tీసుకోవడం recovery ki manchidi.",
        "Nidra (7-8 hrs) kuda training antha important - recovery lo major role.",
        "Hydration mర్చిపోకండి - workout ki mundu, madhya, tarvata నీళ్ళు తాగండి.",
    ]
    return (
        "Idi naku ardham kaledu 🙂 Nenu ee panulu cheyagalanu: predict / log workout / "
        "recommend / progress. Try cheyandi, e.g. 'predict my exercise' leda 'show my stats'.\n\n"
        f"Quick tip: {tips[hash(message) % len(tips)]}"
    )
