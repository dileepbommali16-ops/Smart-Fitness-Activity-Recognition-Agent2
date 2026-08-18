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
    # Exact field currently being requested. This prevents the agent from
    # assigning a short answer to the wrong field.
    pending_field: Optional[str] = None
    # Number of times the user has given an unrecognized/unknown answer
    # for the current field.
    unknown_attempts: int = 0
    slots: Dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.pending_intent = None
        self.pending_field = None
        self.unknown_attempts = 0
        self.slots = {}


# --------------------------------------------------------------------------
# Validation / conversational helpers
# --------------------------------------------------------------------------

# Conservative ranges used to catch clearly invalid values. These do not
# replace the trained model; they only prevent obviously bad inputs.
FIELD_RANGES: Dict[str, tuple[float, float]] = {
    "age": (5, 100),
    "height_cm": (50, 250),
    "weight_kg": (20, 300),
    "heart_rate": (30, 250),
    "body_temp_c": (30, 45),
    "duration_min": (1, 600),
    "calories_burned": (1, 5000),
    "steps": (0, 100000),
    "distance_km": (0, 500),
}

UNKNOWN_WORDS = (
    "telidhu",
    "teliyadu",
    "teliyaledu",
    "don't know",
    "dont know",
    "i don't know",
    "i dont know",
    "unknown",
    "not sure",
    "no idea",
    "skip",
    "naaku telidhu",
    "naku telidhu",
)

def _is_unknown_answer(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return any(word in normalized for word in UNKNOWN_WORDS)

def _validate_prediction_value(field_name: str, value: Any) -> tuple[bool, str]:
    if field_name == "gender":
        if str(value).title() in {"Male", "Female", "Other"}:
            return True, ""
        return False, "Gender **Male / Female / Other** lo enter cheyyandi."

    if field_name not in FIELD_RANGES:
        return True, ""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False, f"{FIELD_LABELS.get(field_name, field_name)} ki valid number enter cheyyandi."

    low, high = FIELD_RANGES[field_name]
    if not (low <= number <= high):
        label = FIELD_LABELS.get(field_name, field_name)
        return False, (
            f"❌ **{label} = {number:g}** valid range lo ledu. "
            f"Please {low:g}–{high:g} madhya value enter cheyyandi."
        )
    return True, ""

def _parse_answer_for_field(message: str, field_name: str) -> Any:
    text = message.strip()

    if field_name == "gender":
        for word, canonical in GENDER_WORDS.items():
            if re.search(rf"\b{word}\b", text, flags=re.IGNORECASE):
                return canonical
        return None

    if field_name == "exercise_name":
        if text and not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
            return text.title()
        return None

    # Accept a bare number as well as "80 bpm", "70 kg", etc.
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None

def _prediction_question(field_name: str) -> str:
    label = FIELD_LABELS.get(field_name, field_name)
    if field_name == "gender":
        return f"{label} enti? (Male / Female / Other)"
    return f"{label} entha? (value cheppandi)"

def _unknown_response(field_name: str, state: AgentState) -> str:
    label = FIELD_LABELS.get(field_name, field_name)
    state.unknown_attempts += 1
    if state.unknown_attempts == 1:
        return (
            f"Okay 👍 **{label}** value meeku teliyakapothe parledhu.\n"
            f"Kaani ee prediction model ki {label} required.\n\n"
            f"Please actual value enter cheyyandi. Example: "
            f"`{_prediction_example(field_name)}`"
        )
    return (
        f"⚠️ **{label}** ni skip cheyyadam valla prediction run cheyyalenu.\n"
        f"Please {label} value enter cheyyandi, or measurement chesi try cheyyandi.\n\n"
        f"Example: `{_prediction_example(field_name)}`"
    )

def _prediction_example(field_name: str) -> str:
    examples = {
        "age": "21",
        "gender": "Male",
        "height_cm": "170",
        "weight_kg": "65",
        "heart_rate": "80",
        "body_temp_c": "36.8",
        "duration_min": "30",
        "calories_burned": "250",
        "steps": "5000",
        "distance_km": "3.5",
    }
    return examples.get(field_name, "100")

def get_workout_question(field_name: str) -> str:
    questions = {
        "exercise_name": "Exercise peru enti?",
        "sets": "Enni sets chesaru?",
        "reps": "Prathi set ki enni reps chesaru?",
        "duration": "Enni minutes chesaru?",
        "calories_burned": "Enni calories burn ayyayi? Approx value ayina parledu?",
    }
    return questions.get(field_name, f"{field_name} enti?")

def validate_workout_field(field_name: str, value: Any) -> tuple[bool, str]:
    if field_name == "exercise_name":
        if not str(value).strip():
            return False, "Exercise name empty ga undakudadhu."
        return True, ""

    ranges = {
        "sets": (1, 100),
        "reps": (1, 1000),
        "duration": (1, 600),
        "calories_burned": (1, 5000),
    }

    if field_name not in ranges:
        return True, ""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False, f"{field_name} ki valid number enter cheyyandi."

    low, high = ranges[field_name]
    if not (low <= number <= high):
        labels = {
            "sets": "Sets",
            "reps": "Reps",
            "duration": "Duration",
            "calories_burned": "Calories",
        }
        label = labels.get(field_name, field_name)
        return (
            False,
            f"❌ **{label} = {number:g}** valid range lo ledu. "
            f"Please {low:g}–{high:g} madhya value enter cheyyandi.",
        )
    return True, ""

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
        # First priority: if the agent asked a specific question, the next
        # answer belongs to that exact field.
        if not already_parsed and state.pending_field:
            field_name = state.pending_field

            if _is_unknown_answer(message):
                return _unknown_response(field_name, state)

            value = _parse_answer_for_field(message, field_name)

            if value is None:
                return (
                    f"❌ Naku `{message}` ni {FIELD_LABELS.get(field_name, field_name)} "
                    f"ga ardham kaaledu.\n\n"
                    f"{_prediction_question(field_name)}"
                )

            valid, error = _validate_prediction_value(field_name, value)
            if not valid:
                return f"{error}\n\n{_prediction_question(field_name)}"

            state.slots[field_name] = value
            state.pending_field = None
            state.unknown_attempts = 0

        # On the initial message, extract all values that were explicitly
        # mentioned. Invalid values are not silently accepted.
        if already_parsed:
            extracted = extract_prediction_slots(message)
            cleaned: Dict[str, Any] = {}
            for name, value in extracted.items():
                valid, _ = _validate_prediction_value(name, value)
                if valid:
                    cleaned[name] = value
            state.slots = cleaned

        missing = [
            name for name in PREDICT_FIELD_NAMES
            if name not in state.slots
        ]

        if missing:
            next_field = missing[0]
            state.pending_field = next_field
            return _prediction_question(next_field)

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
        except Exception as exc:
            state.reset()
            return f"Prediction fail ayyindi: {exc}"

        description = EXERCISE_DESCRIPTIONS.get(prediction, "")
        conf_text = (
            f" (confidence {confidence * 100:.1f}%)"
            if confidence is not None else ""
        )

        state.reset()

        reply = f"🎯 Predicted exercise: **{prediction}**{conf_text}."
        if description:
            reply += f"\n{description}"
        reply += (
            "\n\n'recommend' ani cheppandi ee exercise ki tips kosam, "
            "leda 'log workout' ani cheppi save cheskovachu."
        )
        return reply

    # ---- LOG WORKOUT flow -----------------------------------------------
    def _continue_log_workout(self, message: str, state: AgentState, already_parsed: bool = False) -> str:
        required = ["exercise_name", "sets", "reps", "duration", "calories_burned"]

        if not already_parsed and state.pending_field:
            field_name = state.pending_field

            if _is_unknown_answer(message):
                label = {
                    "exercise_name": "exercise name",
                    "sets": "sets",
                    "reps": "reps",
                    "duration": "duration",
                    "calories_burned": "calories",
                }.get(field_name, field_name)

                return (
                    f"Okay 👍 **{label}** value teliyakapothe save cheyyalenu.\n"
                    f"Please actual value enter cheyyandi.\n"
                    f"{get_workout_question(field_name)}"
                )

            value = _parse_answer_for_field(message, field_name)

            if value is None:
                return (
                    f"❌ Naku adi ardham kaaledu.\n"
                    f"{get_workout_question(field_name)}"
                )

            valid, error = validate_workout_field(field_name, value)
            if not valid:
                return f"{error}\n\n{get_workout_question(field_name)}"

            state.slots[field_name] = value
            state.pending_field = None
            state.unknown_attempts = 0

        if already_parsed:
            extracted = extract_workout_slots(message)
            for name, value in extracted.items():
                if name == "date":
                    state.slots[name] = value
                    continue
                valid, _ = validate_workout_field(name, value)
                if valid:
                    state.slots[name] = value

        state.slots.setdefault("date", str(date.today()))

        missing = [name for name in required if name not in state.slots]

        if missing:
            next_field = missing[0]
            state.pending_field = next_field
            return get_workout_question(next_field)

        try:
            payload = {
                "exercise_name": str(state.slots["exercise_name"]).strip() or "Unknown",
                "sets": int(float(state.slots["sets"])),
                "reps": int(float(state.slots["reps"])),
                "duration": float(state.slots["duration"]),
                "calories_burned": float(state.slots["calories_burned"]),
                "date": state.slots.get("date", str(date.today())),
            }
            save_workout(payload)
        except Exception as exc:
            state.reset()
            return f"Workout save cheyadam fail ayyindi: {exc}"

        state.reset()
        return (
            f"✅ Saved! **{payload['exercise_name']}** — "
            f"{payload['sets']} sets x {payload['reps']} reps, "
            f"{payload['duration']:.0f} min, "
            f"{payload['calories_burned']:.0f} kcal ({payload['date']}).\n"
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
