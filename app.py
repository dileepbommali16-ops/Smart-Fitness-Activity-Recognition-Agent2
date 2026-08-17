"""Smart Fitness Activity Recognition Agent built with Streamlit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from agent import AgentState, FitnessAgent
from recommendation import EXERCISE_DESCRIPTIONS, EXERCISE_RECOMMENDATIONS, DEFAULT_RECOMMENDATION
from utils import FEATURE_FIELDS, load_history, load_model, predict_activity, save_workout

st.set_page_config(page_title="Smart Fitness Agent", page_icon="🏋️", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%); color: white; }
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stSidebar"] { background: rgba(15, 23, 42, 0.95); }
    .stMetric { background: rgba(255,255,255,0.12); padding: 1rem; border-radius: 16px; }
    .card { background: rgba(255,255,255,0.14); padding: 1rem 1.2rem; border-radius: 16px; backdrop-filter: blur(8px); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model_bundle() -> Dict[str, Any]:
    """Load the model bundle once and reuse it across the app."""
    return load_model()


@st.cache_resource
def get_agent() -> FitnessAgent:
    """Create the AI Agent once and reuse it across reruns."""
    return FitnessAgent()


def render_sidebar() -> str:
    """Render the sidebar navigation and return the selected page."""
    st.sidebar.title("🏋️ Fitness Agent")
    st.sidebar.caption("AI-powered activity recognition and fitness coaching")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Home",
            "AI Agent",
            "Predict Exercise",
            "Workout Tracker",
            "Fitness Recommendations",
            "Workout History",
            "About",
        ],
        index=0,
    )
    return page


def render_feature_form() -> Dict[str, Any]:
    """Render input controls for the prediction form."""
    values: Dict[str, Any] = {}
    for field in FEATURE_FIELDS:
        name = field["name"]
        label = field["label"]
        input_type = field.get("input", "number")
        if input_type == "selectbox":
            values[name] = st.selectbox(label, options=field.get("options", []), index=0)
        else:
            values[name] = st.number_input(
                label,
                min_value=field.get("min_value"),
                max_value=field.get("max_value"),
                value=field.get("default", 0),
                step=field.get("step", 1),
            )
    return values


def render_home() -> None:
    """Render the home dashboard with workout insights."""
    st.title("🏠 Smart Fitness Activity Recognition")
    st.write("Track workouts, recognize activities, and receive personalized fitness guidance in one place.")

    history = load_history()
    if history.empty:
        st.info("No workouts logged yet. Start by adding your first workout and the dashboard will populate automatically.")
        return

    total_workouts = len(history)
    total_calories = round(float(history["calories_burned"].sum()), 2)
    total_duration = round(float(history["duration"].sum()), 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Workouts", total_workouts)
    col2.metric("Calories Burned", f"{total_calories:.0f} kcal")
    col3.metric("Total Duration", f"{total_duration:.0f} min")

    st.markdown("### 📊 Dashboard Insights")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        calories_chart = px.bar(
            history,
            x="date",
            y="calories_burned",
            color="exercise_name",
            title="Calories Burned by Date",
            labels={"date": "Date", "calories_burned": "Calories"},
        )
        st.plotly_chart(calories_chart, use_container_width=True)
    with chart_col2:
        freq_chart = history.groupby("date").size().reset_index(name="workouts")
        freq_chart = px.line(freq_chart, x="date", y="workouts", title="Workout Frequency", markers=True)
        st.plotly_chart(freq_chart, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        dist_chart = px.pie(
            history,
            names="exercise_name",
            values="calories_burned",
            title="Exercise Distribution",
        )
        st.plotly_chart(dist_chart, use_container_width=True)
    with chart_col4:
        duration_chart = px.line(
            history,
            x="date",
            y="duration",
            color="exercise_name",
            title="Duration Trend",
            markers=True,
        )
        st.plotly_chart(duration_chart, use_container_width=True)


def render_agent_page() -> None:
    """Render the conversational AI Agent chat interface."""
    st.title("🤖 AI Fitness Agent")
    st.write(
        "Matladandi natural language lo — predict cheyamani cheppandi, "
        "workout log cheyamani cheppandi, tips adagandi, leda progress "
        "adagandi. Agent decide chesukuni correct action tీసుకుంటుంది."
    )

    agent = get_agent()

    if "agent_chat_history" not in st.session_state:
        st.session_state.agent_chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Hi! Nenu nee AI Fitness Agent. Try: "
                    "'predict my exercise', 'I did running 30 min 300 calories', "
                    "'give me tips for cycling', or 'show my stats'."
                ),
            }
        ]
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = AgentState()

    for turn in st.session_state.agent_chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    user_message = st.chat_input("Type your message...")
    if user_message:
        st.session_state.agent_chat_history.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        reply = agent.respond(user_message, st.session_state.agent_state)

        st.session_state.agent_chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    if st.button("Reset conversation"):
        st.session_state.agent_chat_history = []
        st.session_state.agent_state = AgentState()
        st.rerun()


def render_prediction_page() -> None:
    """Render the prediction page and show the recognized activity."""
    st.title("🔮 Predict Exercise")
    st.write("Enter movement and body metrics to estimate the exercise being performed.")

    try:
        model_bundle = get_model_bundle()
    except FileNotFoundError as exc:
        st.error(f"Model files are missing: {exc}")
        st.info("Place your fitness_model.pkl and scaler.pkl files in the project root before running predictions.")
        return

    with st.form("prediction_form"):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        feature_inputs = render_feature_form()
        st.markdown("</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Exercise", use_container_width=True)

    if submitted:
        try:
            prediction, confidence = predict_activity(
                model_bundle["model"],
                model_bundle["scaler"],
                feature_inputs,
                feature_names=model_bundle["feature_names"],
            )
            st.session_state["last_prediction"] = prediction
            st.session_state["last_confidence"] = confidence
            st.success("Exercise prediction completed successfully.")

            st.markdown("### 🎯 Prediction Result")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Exercise", prediction)
            with col2:
                if confidence is not None:
                    st.metric("Confidence Score", f"{confidence * 100:.2f}%")
                else:
                    st.metric("Confidence Score", "Not available")

            description = EXERCISE_DESCRIPTIONS.get(prediction, "This activity is a great addition to your fitness routine.")
            st.markdown(f"**Exercise Description:** {description}")
        except Exception as exc:  # pragma: no cover - defensive UI handling
            st.error(f"Prediction failed: {exc}")


def render_tracker_page() -> None:
    """Render a workout tracker form and save entries to CSV."""
    st.title("📝 Workout Tracker")
    st.write("Log each workout session to build a history and monitor your progress.")

    with st.form("tracker_form"):
        exercise_name = st.text_input("Exercise Name", placeholder="e.g. Running")
        sets = st.number_input("Sets", min_value=1, max_value=20, value=3)
        reps = st.number_input("Reps", min_value=1, max_value=200, value=10)
        duration = st.number_input("Duration (minutes)", min_value=1, max_value=480, value=20)
        calories = st.number_input("Calories Burned", min_value=0, max_value=5000, value=250)
        date = st.date_input("Date")
        submitted = st.form_submit_button("Save Workout", use_container_width=True)

    if submitted:
        try:
            payload = {
                "exercise_name": exercise_name.strip() or "Unknown",
                "sets": int(sets),
                "reps": int(reps),
                "duration": float(duration),
                "calories_burned": float(calories),
                "date": str(date),
            }
            save_workout(payload)
            st.success("Workout saved successfully.")
        except Exception as exc:  # pragma: no cover - defensive UI handling
            st.error(f"Failed to save workout: {exc}")


def render_history_page() -> None:
    """Render workout history, summaries, and filtering controls."""
    st.title("📚 Workout History")

    history = load_history()
    if history.empty:
        st.info("Workout history is empty. Log a workout to see it here.")
        return

    selected_date = st.date_input("Filter by Date")
    if selected_date:
        history = history[history["date"] == str(selected_date)]

    st.markdown("### 📈 Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Workouts", len(history))
    col2.metric("Total Calories", f"{float(history['calories_burned'].sum()):.0f} kcal")
    col3.metric("Total Duration", f"{float(history['duration'].sum()):.0f} min")

    st.dataframe(history, use_container_width=True)


def render_recommendations_page() -> None:
    """Render personalized fitness recommendations based on the detected exercise."""
    st.title("💡 Fitness Recommendations")
    st.write("Receive tailored workout guidance and recovery advice based on the predicted activity.")

    selected_exercise = st.selectbox(
        "Exercise",
        options=list(EXERCISE_RECOMMENDATIONS.keys()),
        index=0,
    )

    if "last_prediction" in st.session_state:
        selected_exercise = st.session_state["last_prediction"]

    recommendation = EXERCISE_RECOMMENDATIONS.get(selected_exercise, DEFAULT_RECOMMENDATION)

    st.markdown(f"### 🌟 Recommendations for {selected_exercise}")
    st.markdown(f"**Description:** {EXERCISE_DESCRIPTIONS.get(selected_exercise, 'Stay active and keep progressing.')}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Warm-up")
        for item in recommendation.get("warm_up", []):
            st.write(f"• {item}")
        st.subheader("Recommended Sets & Reps")
        st.write(recommendation.get("sets_reps", DEFAULT_RECOMMENDATION["sets_reps"]))
        st.subheader("Rest Time")
        st.write(recommendation.get("rest_time", DEFAULT_RECOMMENDATION["rest_time"]))
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Hydration Advice")
        st.write(recommendation.get("hydration", DEFAULT_RECOMMENDATION["hydration"]))
        st.subheader("Nutrition Tips")
        st.write(recommendation.get("nutrition", DEFAULT_RECOMMENDATION["nutrition"]))
        st.subheader("Protein Recommendation")
        st.write(recommendation.get("protein", DEFAULT_RECOMMENDATION["protein"]))
        st.subheader("Recovery Advice")
        st.write(recommendation.get("recovery", DEFAULT_RECOMMENDATION["recovery"]))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🧭 Skill Levels")
    levels = recommendation.get("levels", DEFAULT_RECOMMENDATION["levels"])
    for level, advice in levels.items():
        st.write(f"**{level}:** {advice}")


def render_about_page() -> None:
    """Render the about page."""
    st.title("ℹ️ About")
    st.write("This app combines machine learning with fitness coaching to make exercise recognition and progress tracking more intuitive.")
    st.markdown(
        """
        - Load a trained classifier and scaler from pickle files.
        - Predict the most likely activity from user-entered health and movement metrics.
        - Chat with an AI Agent that decides on its own whether to predict, log, recommend, or report progress.
        - Save workouts locally in CSV format.
        - Receive suggestions for warm-up, nutrition, protein, and recovery.
        """
    )


def main() -> None:
    """Main entry point for the Streamlit app."""
    page = render_sidebar()

    if page == "Home":
        render_home()
    elif page == "AI Agent":
        render_agent_page()
    elif page == "Predict Exercise":
        render_prediction_page()
    elif page == "Workout Tracker":
        render_tracker_page()
    elif page == "Fitness Recommendations":
        render_recommendations_page()
    elif page == "Workout History":
        render_history_page()
    else:
        render_about_page()


if __name__ == "__main__":
    main()
