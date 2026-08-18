"""
Smart Fitness AI Agent.

This module turns the existing prediction model, recommendation engine,
and workout tracker into a conversational agent with:

1. Intent detection
2. Multi-turn conversation memory
3. Exact pending-field tracking
4. Input validation
5. ML activity prediction
6. Workout logging
7. Recommendations
8. Progress/history analysis

No external LLM/API is required.
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

from utils import (
    FEATURE_FIELDS,
    load_history,
    load_model,
    predict_activity,
    save_workout,
)


# ============================================================================
# FIELD DEFINITIONS
# ============================================================================

PREDICT_FIELD_NAMES: List[str] = [f["name"] for f in FEATURE_FIELDS]

FIELD_LABELS: Dict[str, str] = {
    f["name"]: f["label"] for f in FEATURE_FIELDS
}

GENDER_WORDS = {
    "male": "Male",
    "female": "Female",
    "other": "Other",
}


# ============================================================================
# VALIDATION RANGES
# ============================================================================

# These ranges are used only to catch clearly invalid input.
# They do not change the ML model itself.
FIELD_RANGES = {
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


# ============================================================================
# FIELD EXTRACTION
# ============================================================================

FIELD_EXTRACTORS: Dict[str, Dict[str, Any]] = {
    "age": {
        "keywords": [r"\bage\b"],
        "unit": None,
    },
    "gender": {
        "keywords": [],
        "unit": None,
    },
    "height_cm": {
        "keywords": [r"\bheight\b"],
        "unit": r"(\d+(?:\.\d+)?)\s*cm\b",
    },
    "weight_kg": {
        "keywords": [r"\bweight\b", r"\bweigh\b"],
        "unit": r"(\d+(?:\.\d+)?)\s*kg\b",
    },
    "heart_rate": {
        "keywords": [r"heart\s*rate", r"\bhr\b"],
        "unit": r"(\d+(?:\.\d+)?)\s*bpm\b",
    },
    "body_temp_c": {
        "keywords": [r"temp(?:erature)?"],
        "unit": r"(\d+(?:\.\d+)?)\s*(?:°c|c\b)",
    },
    "duration_min": {
        "keywords": [r"\bduration\b"],
        "unit": r"(\d+(?:\.\d+)?)\s*(?:min|minutes?)\b",
    },
    "calories_burned": {
        "keywords": [r"\bcalories\b", r"\bkcal\b"],
        "unit": r"(\d+(?:\.\d+)?)\s*(?:kcal|cal(?:ories)?)\b",
    },
    "steps": {
        "keywords": [r"\bsteps\b"],
        "unit": r"(\d+(?:\.\d+)?)\s*steps?\b",
    },
    "distance_km": {
        "keywords": [r"\bdistance\b"],
        "unit": r"(\d+(?:\.\d+)?)\s*km\b",
    },
}


def _extract_number(
    text: str,
    keywords: List[str],
    unit_pattern: Optional[str],
) -> Optional[float]:

    for keyword in keywords:
        match = re.search(
            rf"{keyword}\D{{0,8}}?(\d+(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return float(match.group(1))

    if unit_pattern:
        match = re.search(
            unit_pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return float(match.group(1))

    return None


def extract_prediction_slots(text: str) -> Dict[str, Any]:
    """
    Extract as many prediction fields as possible from free text.
    """

    found: Dict[str, Any] = {}

    # Gender
    for word, canonical in GENDER_WORDS.items():
        if re.search(rf"\b{word}\b", text, flags=re.IGNORECASE):
            found["gender"] = canonical
            break

    # Numeric fields
    for name, spec in FIELD_EXTRACTORS.items():

        if name == "gender":
            continue

        value = _extract_number(
            text,
            spec["keywords"],
            spec["unit"],
        )

        if value is not None:
            found[name] = value

    return found


# ============================================================================
# WORKOUT EXTRACTION
# ============================================================================

def extract_workout_slots(text: str) -> Dict[str, Any]:
    """
    Extract workout logging fields from free text.
    """

    found: Dict[str, Any] = {}

    # Exercise name
    name_match = re.search(
        r"(?:did|logged?|log|record(?:ed)?|add)\s+"
        r"([a-zA-Z\- ]+?)"
        r"(?:\s+for|\s+\d|,|\.|$)",
        text,
        flags=re.IGNORECASE,
    )

    filler_words = {
        "workout",
        "exercise",
        "session",
        "a workout",
        "my workout",
    }

    if name_match:
        candidate = name_match.group(1).strip().title()

        if candidate and candidate.lower() not in filler_words:
            found["exercise_name"] = candidate

    patterns = {
        "sets": r"(\d+)\s*sets?\b",
        "reps": r"(\d+)\s*reps?\b",
        "duration": r"(\d+(?:\.\d+)?)\s*(?:min|minutes?)\b",
        "calories_burned": r"(\d+(?:\.\d+)?)\s*(?:kcal|cal(?:ories)?)\b",
    }

    for name, pattern in patterns.items():

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            found[name] = float(match.group(1))

    found.setdefault("date", str(date.today()))

    return found


# ============================================================================
# INTENT DETECTION
# ============================================================================

INTENT_KEYWORDS = {
    "predict": [
        "predict",
        "which exercise",
        "what exercise",
        "recognize",
        "detect",
        "identify my",
    ],

    "log_workout": [
        "log",
        "save workout",
        "record",
        "add workout",
        "i did",
        "i finished",
        "just did",
    ],

    "recommend": [
        "recommend",
        "suggestion",
        "tips",
        "advice",
        "how should i train",
        "guide me",
    ],

    "history": [
        "history",
        "progress",
        "stats",
        "summary",
        "how many workouts",
        "dashboard",
    ],

    "greet": [
        "hi",
        "hello",
        "hey",
        "namaste",
        "hai",
        "good morning",
        "good evening",
    ],
}


def detect_intent(message: str) -> str:

    text = message.lower().strip()

    for intent, keywords in INTENT_KEYWORDS.items():

        for keyword in keywords:

            if keyword in text:
                return intent

    return "smalltalk"


# ============================================================================
# AGENT STATE
# ============================================================================

@dataclass
class AgentState:

    pending_intent: Optional[str] = None

    # NEW:
    # Exact field for which the agent is currently waiting for an answer.
    pending_field: Optional[str] = None

    slots: Dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:

        self.pending_intent = None
        self.pending_field = None
        self.slots = {}


# ============================================================================
# VALIDATION
# ============================================================================

def validate_field(
    field_name: str,
    value: Any,
) -> tuple[bool, str]:

    if field_name == "gender":

        if str(value).title() not in {
            "Male",
            "Female",
            "Other",
        }:
            return (
                False,
                "అయ్యో, అది నాకు అర్థం కాలేదు 🙂 Gender కోసం "
                "**Male**, **Female**, లేదా **Other** అని matrame రాయండి.",
            )

        return True, ""

    if field_name not in FIELD_RANGES:
        return True, ""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        label = FIELD_LABELS.get(field_name, field_name)
        return (
            False,
            f"అది నాకు ఒక సంఖ్యలా అనిపించలేదు 🙂 {label} కోసం దయచేసి "
            f"కేవలం ఒక number మాత్రమే enter చేయండి (ఉదా: 25).",
        )

    minimum, maximum = FIELD_RANGES[field_name]

    if numeric_value < minimum or numeric_value > maximum:

        label = FIELD_LABELS.get(
            field_name,
            field_name,
        )

        return (
            False,
            f"హ్మ్, `{numeric_value:g}` అనేది {label} కి కొంచెం సరైన విలువ "
            f"కాదు అనిపిస్తోంది — ఇది సాధారణంగా **{minimum} నుండి {maximum}** మధ్య "
            f"ఉంటుంది. దయచేసి మళ్ళీ సరైన విలువ చెప్పండి.",
        )

    return True, ""


def parse_answer_for_field(
    message: str,
    field_name: str,
) -> Any:

    text = message.strip()

    # ------------------------------------------------------------
    # Gender
    # ------------------------------------------------------------

    if field_name == "gender":

        for word, canonical in GENDER_WORDS.items():

            if re.search(
                rf"\b{word}\b",
                text,
                flags=re.IGNORECASE,
            ):
                return canonical

        return None

    # ------------------------------------------------------------
    # Exercise name
    # ------------------------------------------------------------

    if field_name == "exercise_name":

        if text and not re.fullmatch(
            r"\d+(?:\.\d+)?",
            text,
        ):
            return text.title()

        return None

    # ------------------------------------------------------------
    # Number
    # ------------------------------------------------------------

    number_match = re.search(
        r"[-+]?\d+(?:\.\d+)?",
        text,
    )

    if number_match:

        try:
            return float(number_match.group(0))
        except ValueError:
            return None

    return None


# ============================================================================
# FIELD QUESTIONS
# ============================================================================

def get_prediction_question(field_name: str) -> str:

    label = FIELD_LABELS.get(
        field_name,
        field_name,
    )

    friendly_questions = {
        "gender": "మీ **Gender** ఏంటో చెప్పండి — Male, Female, లేదా Other అని రాయండి 🙂",
        "age": "మీ **Age** ఎంత? (ఉదా: 25)",
        "height_cm": "మీ **Height** ఎంత, cm లో చెప్పండి (ఉదా: 170)",
        "weight_kg": "మీ **Weight** ఎంత, kg లో చెప్పండి (ఉదా: 65)",
        "heart_rate": "Workout చేసేటప్పుడు మీ **Heart Rate** ఎంత ఉంటుంది, bpm లో? (ఉదా: 110)",
        "body_temp_c": "మీ **Body Temperature** ఎంత, °C లో? (ఉదా: 37)",
        "duration_min": "ఎంత సేపు workout చేసారు, minutes లో చెప్పండి (ఉదా: 30)",
        "calories_burned": "సుమారుగా ఎన్ని **Calories** burn అయ్యాయో చెప్పండి (ఉదా: 250)",
        "steps": "ఎన్ని **Steps** వేసారు? (ఉదా: 5000)",
        "distance_km": "ఎంత **Distance** కవర్ చేసారు, km లో? (ఉదా: 3.5)",
    }

    return friendly_questions.get(
        field_name,
        f"{label} entha? దయచేసి ఒక విలువ చెప్పండి.",
    )


def get_workout_question(field_name: str) -> str:

    questions = {

        "exercise_name":
            "బాగుంది! ఏ **exercise** చేసారో పేరు చెప్పండి (ఉదా: Running, Squats, Cycling) 🏋️",

        "sets":
            "సూపర్! ఎన్ని **sets** చేసారు? (ఉదా: 3)",

        "reps":
            "ప్రతి set కి ఎన్ని **reps** చేసారు? (ఉదా: 12)",

        "duration":
            "మొత్తం ఎన్ని **minutes** workout చేసారు? (ఉదా: 20)",

        "calories_burned":
            "సుమారుగా ఎన్ని **calories** burn అయ్యాయో చెప్పండి — exact గా తెలియకపోతే అంచనా చెప్పినా చాలు (ఉదా: 180)",
    }

    return questions.get(
        field_name,
        f"{field_name} గురించి చెప్పండి?",
    )


# ============================================================================
# FITNESS AGENT
# ============================================================================

class FitnessAgent:
    """
    Conversational wrapper around the model,
    workout tracker and recommendation system.
    """

    def __init__(self) -> None:

        self._model_bundle: Optional[Dict[str, Any]] = None

        self._model_error: Optional[str] = None

    # ------------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------------

    def _ensure_model(self) -> bool:

        if self._model_bundle is not None:
            return True

        if self._model_error is not None:
            return False

        try:

            self._model_bundle = load_model()

            return True

        except Exception as exc:

            self._model_error = str(exc)

            return False

    # ------------------------------------------------------------------------
    # MAIN RESPOND
    # ------------------------------------------------------------------------

    def respond(
        self,
        message: str,
        state: AgentState,
    ) -> str:

        message = message.strip()

        if not message:

            return (
                "Cheppandi, meeku em kavali? "
                "(predict / log workout / recommendation / progress)"
            )

        # ====================================================================
        # IMPORTANT:
        # If agent is waiting for a specific field, process the answer
        # against THAT exact field.
        # ====================================================================

        if state.pending_intent == "predict":

            return self._continue_predict(
                message,
                state,
            )

        if state.pending_intent == "log_workout":

            return self._continue_log_workout(
                message,
                state,
            )

        # ====================================================================
        # New intent
        # ====================================================================

        intent = detect_intent(message)

        # --------------------------------------------------------------------
        # Greeting
        # --------------------------------------------------------------------

        if intent == "greet":

            return (
                "Hi! Nenu nee Smart Fitness AI Agent 🤖.\n\n"
                "Nenu ee panulu cheyagalanu:\n"
                "- **Predict**: nee body/movement metrics tho exercise predict chestanu\n"
                "- **Log workout**: workout details save chestanu\n"
                "- **Recommend**: exercise ki fitness tips istanu\n"
                "- **Progress**: workout history and stats chupistanu\n\n"
                "Em kavali cheppandi!"
            )

        # --------------------------------------------------------------------
        # Predict
        # --------------------------------------------------------------------

        if intent == "predict":

            state.pending_intent = "predict"

            state.slots = extract_prediction_slots(
                message
            )

            return self._continue_predict(
                message,
                state,
                already_parsed=True,
            )

        # --------------------------------------------------------------------
        # Workout logging
        # --------------------------------------------------------------------

        if intent == "log_workout":

            state.pending_intent = "log_workout"

            state.slots = extract_workout_slots(
                message
            )

            return self._continue_log_workout(
                message,
                state,
                already_parsed=True,
            )

        # --------------------------------------------------------------------
        # Recommendation
        # --------------------------------------------------------------------

        if intent == "recommend":

            return self._handle_recommend(
                message
            )

        # --------------------------------------------------------------------
        # History
        # --------------------------------------------------------------------

        if intent == "history":

            return self._handle_history()

        # --------------------------------------------------------------------
        # Smalltalk
        # --------------------------------------------------------------------

        return self._smalltalk(message)

    # =========================================================================
    # PREDICT FLOW
    # =========================================================================

    def _continue_predict(
        self,
        message: str,
        state: AgentState,
        already_parsed: bool = False,
    ) -> str:

        # =====================================================================
        # STEP 1:
        # If we explicitly asked for a field, answer MUST be processed
        # against that field first.
        # =====================================================================

        if not already_parsed and state.pending_field:

            field_name = state.pending_field

            value = parse_answer_for_field(
                message,
                field_name,
            )

            # ---------------------------------------------------------------
            # Couldn't understand
            # ---------------------------------------------------------------

            if value is None:

                return (
                    f"అయ్యో, నాకు అది సరిగ్గా అర్థం కాలేదు 🙂 మళ్ళీ ఒకసారి ప్రయత్నించండి:\n\n"
                    f"{get_prediction_question(field_name)}"
                )

            # ---------------------------------------------------------------
            # Validate
            # ---------------------------------------------------------------

            valid, error_message = validate_field(
                field_name,
                value,
            )

            if not valid:

                return (
                    f"{error_message}\n\n"
                    f"{get_prediction_question(field_name)}"
                )

            # ---------------------------------------------------------------
            # Save answer
            # ---------------------------------------------------------------

            state.slots[field_name] = value

            # IMPORTANT:
            # Clear old pending field so it won't repeat.
            state.pending_field = None

        # =====================================================================
        # STEP 2:
        # If this is the first prediction message, extract all fields.
        # =====================================================================

        if already_parsed:

            # Validate extracted fields before storing them.

            extracted = dict(state.slots)

            validated_slots: Dict[str, Any] = {}

            for field_name, value in extracted.items():

                valid, _ = validate_field(
                    field_name,
                    value,
                )

                if valid:
                    validated_slots[field_name] = value

            state.slots = validated_slots

        # =====================================================================
        # STEP 3:
        # Find the next missing field.
        # =====================================================================

        missing = [
            name
            for name in PREDICT_FIELD_NAMES
            if name not in state.slots
        ]

        if missing:

            next_field = missing[0]

            # IMPORTANT:
            # Remember EXACTLY what we are asking.
            state.pending_field = next_field

            return get_prediction_question(
                next_field
            )

        # =====================================================================
        # STEP 4:
        # All fields collected -> prediction
        # =====================================================================

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

            return (
                f"Prediction fail ayyindi: {exc}"
            )

        description = EXERCISE_DESCRIPTIONS.get(
            prediction,
            "",
        )

        conf_text = (
            f" (confidence {confidence * 100:.1f}%)"
            if confidence is not None
            else ""
        )

        state.reset()

        reply = (
            f"🎯 Predicted exercise: **{prediction}**"
            f"{conf_text}."
        )

        if description:
            reply += f"\n{description}"

        reply += (
            "\n\n"
            "'recommend' ani cheppandi ee exercise ki tips kosam, "
            "leda 'log workout' ani cheppi save cheskovachu."
        )

        return reply

    # =========================================================================
    # LOG WORKOUT FLOW
    # =========================================================================

    def _continue_log_workout(
        self,
        message: str,
        state: AgentState,
        already_parsed: bool = False,
    ) -> str:

        required = [
            "exercise_name",
            "sets",
            "reps",
            "duration",
            "calories_burned",
        ]

        # =====================================================================
        # First message
        # =====================================================================

        if already_parsed:

            extracted = extract_workout_slots(
                message
            )

            validated_slots: Dict[str, Any] = {}

            for field_name, value in extracted.items():

                if field_name == "date":

                    validated_slots[field_name] = value
                    continue

                valid, _ = validate_workout_field(
                    field_name,
                    value,
                )

                if valid:
                    validated_slots[field_name] = value

            state.slots.update(
                validated_slots
            )

        # =====================================================================
        # Follow-up answer
        # =====================================================================

        elif state.pending_field:

            field_name = state.pending_field

            value = parse_answer_for_field(
                message,
                field_name,
            )

            if value is None:

                return (
                    f"అయ్యో, నాకు అది సరిగ్గా అర్థం కాలేదు 🙂 మళ్ళీ ఒకసారి ప్రయత్నించండి:\n\n"
                    f"{get_workout_question(field_name)}"
                )

            valid, error_message = validate_workout_field(
                field_name,
                value,
            )

            if not valid:

                return (
                    f"{error_message}\n\n"
                    f"{get_workout_question(field_name)}"
                )

            state.slots[field_name] = value

            state.pending_field = None

        # =====================================================================
        # Date
        # =====================================================================

        state.slots.setdefault(
            "date",
            str(date.today()),
        )

        # =====================================================================
        # Find next missing field
        # =====================================================================

        missing = [
            name
            for name in required
            if name not in state.slots
        ]

        if missing:

            next_field = missing[0]

            # IMPORTANT:
            # Remember exact question.
            state.pending_field = next_field

            return get_workout_question(
                next_field
            )

        # =====================================================================
        # Save workout
        # =====================================================================

        try:

            payload = {
                "exercise_name":
                    str(
                        state.slots["exercise_name"]
                    ).strip(),

                "sets":
                    int(
                        float(
                            state.slots["sets"]
                        )
                    ),

                "reps":
                    int(
                        float(
                            state.slots["reps"]
                        )
                    ),

                "duration":
                    float(
                        state.slots["duration"]
                    ),

                "calories_burned":
                    float(
                        state.slots["calories_burned"]
                    ),

                "date":
                    state.slots.get(
                        "date",
                        str(date.today()),
                    ),
            }

            save_workout(
                payload
            )

        except Exception as exc:

            state.reset()

            return (
                f"Workout save cheyadam fail ayyindi: {exc}"
            )

        state.reset()

        return (
            f"✅ Saved! **{payload['exercise_name']}** — "
            f"{payload['sets']} sets x "
            f"{payload['reps']} reps, "
            f"{payload['duration']:.0f} min, "
            f"{payload['calories_burned']:.0f} kcal "
            f"({payload['date']}).\n\n"
            "'show my stats' ani cheppi progress chudochu."
        )

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================

    def _handle_recommend(
        self,
        message: str,
    ) -> str:

        text = message.lower()

        matched_exercise: Optional[str] = None

        for exercise_name in EXERCISE_RECOMMENDATIONS.keys():

            if exercise_name.lower() in text:

                matched_exercise = exercise_name

                break

        if matched_exercise is None:

            options = ", ".join(
                list(
                    EXERCISE_RECOMMENDATIONS.keys()
                )[:8]
            )

            return (
                "Ee exercise ki tips kavalo cheppandi "
                "(e.g. 'tips for running').\n"
                f"Available options: {options}"
            )

        rec = EXERCISE_RECOMMENDATIONS.get(
            matched_exercise,
            DEFAULT_RECOMMENDATION,
        )

        description = EXERCISE_DESCRIPTIONS.get(
            matched_exercise,
            "",
        )

        lines = [
            f"💡 **{matched_exercise} recommendations**"
        ]

        if description:
            lines.append(
                description
            )

        warm_up = rec.get(
            "warm_up",
            [],
        )

        if warm_up:

            lines.append(
                "Warm-up: "
                + "; ".join(warm_up)
            )

        lines.append(
            "Sets & reps: "
            + str(
                rec.get(
                    "sets_reps",
                    DEFAULT_RECOMMENDATION[
                        "sets_reps"
                    ],
                )
            )
        )

        lines.append(
            "Rest time: "
            + str(
                rec.get(
                    "rest_time",
                    DEFAULT_RECOMMENDATION[
                        "rest_time"
                    ],
                )
            )
        )

        lines.append(
            "Hydration: "
            + str(
                rec.get(
                    "hydration",
                    DEFAULT_RECOMMENDATION[
                        "hydration"
                    ],
                )
            )
        )

        lines.append(
            "Nutrition: "
            + str(
                rec.get(
                    "nutrition",
                    DEFAULT_RECOMMENDATION[
                        "nutrition"
                    ],
                )
            )
        )

        lines.append(
            "Recovery: "
            + str(
                rec.get(
                    "recovery",
                    DEFAULT_RECOMMENDATION[
                        "recovery"
                    ],
                )
            )
        )

        return "\n".join(lines)

    # =========================================================================
    # HISTORY
    # =========================================================================

    def _handle_history(self) -> str:

        try:

            history = load_history()

        except Exception as exc:

            return (
                f"History load cheyyadam fail ayyindi: {exc}"
            )

        if history.empty:

            return (
                "Inka workout logs ledu. "
                "'log workout' ani cheppi first entry add cheyandi."
            )

        total_workouts = len(
            history
        )

        total_calories = float(
            history["calories_burned"].sum()
        )

        total_duration = float(
            history["duration"].sum()
        )

        top_exercise = (
            history["exercise_name"]
            .value_counts()
            .idxmax()
            if (
                "exercise_name" in history.columns
                and not history.empty
            )
            else "N/A"
        )

        return (
            "📊 **Progress summary**\n"
            f"- Total workouts: {total_workouts}\n"
            f"- Total calories burned: {total_calories:.0f} kcal\n"
            f"- Total duration: {total_duration:.0f} min\n"
            f"- Most frequent exercise: {top_exercise}"
        )

    # =========================================================================
    # SMALL TALK
    # =========================================================================

    def _smalltalk(
        self,
        message: str,
    ) -> str:

        return generate_smalltalk_reply(
            message
        )


# ============================================================================
# WORKOUT VALIDATION
# ============================================================================

def validate_workout_field(
    field_name: str,
    value: Any,
) -> tuple[bool, str]:

    ranges = {
        "sets": (1, 100),
        "reps": (1, 1000),
        "duration": (1, 600),
        "calories_burned": (1, 5000),
    }

    if field_name == "exercise_name":

        if not str(value).strip():

            return (
                False,
                "అయ్యో, exercise పేరు ఖాళీగా ఉండకూడదు 🙂 దయచేసి ఏదైనా "
                "exercise పేరు చెప్పండి (ఉదా: Running, Push-ups, Cycling).",
            )

        return True, ""

    if field_name not in ranges:

        return True, ""

    try:

        numeric_value = float(value)

    except (TypeError, ValueError):

        labels = {
            "sets": "Sets",
            "reps": "Reps",
            "duration": "Duration",
            "calories_burned": "Calories",
        }
        label = labels.get(field_name, field_name)

        return (
            False,
            f"అది నాకు సంఖ్యలా అనిపించలేదు 🙂 {label} కోసం దయచేసి "
            f"కేవలం ఒక number మాత్రమే చెప్పండి.",
        )

    minimum, maximum = ranges[
        field_name
    ]

    if numeric_value < minimum or numeric_value > maximum:

        labels = {
            "sets": "Sets",
            "reps": "Reps",
            "duration": "Duration",
            "calories_burned": "Calories",
        }

        label = labels.get(
            field_name,
            field_name,
        )

        return (
            False,
            f"హ్మ్, `{numeric_value:g}` {label} కి కొంచెం సరైనది "
            f"కాదు అనిపిస్తోంది — ఇది సాధారణంగా **{minimum} నుండి {maximum}** "
            f"మధ్య ఉంటుంది. దయచేసి మళ్ళీ సరైన విలువ చెప్పండి.",
        )

    return True, ""


# ============================================================================
# SMALL TALK
# ============================================================================

def generate_smalltalk_reply(
    message: str,
) -> str:

    tips = [
        "Prathi roju konchem aina move avvadam consistency ki key.",
        "Workout tarvata recovery ki proper rest important.",
        "Nidra training antha important - recovery lo major role.",
        "Hydration marchipokandi - workout mundu, madhya, tarvata neellu tagandi.",
    ]

    return (
        "Idi naku ardham kaledu 🙂\n\n"
        "Nenu ee panulu cheyagalanu:\n"
        "• Predict exercise\n"
        "• Log workout\n"
        "• Give recommendations\n"
        "• Show progress\n\n"
        "Example:\n"
        "`predict my exercise`\n"
        "`I did running for 30 minutes`\n"
        "`give me tips for running`\n"
        "`show my stats`\n\n"
        f"💡 Quick tip: "
        f"{tips[hash(message) % len(tips)]}"
    )
