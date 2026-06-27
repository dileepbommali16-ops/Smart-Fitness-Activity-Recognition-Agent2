# Smart Fitness Activity Recognition Agent

A modern, production-ready Streamlit application for fitness activity recognition, workout tracking, and personalized recommendations.

## Features

- Home dashboard with workout insights and visual charts
- Activity prediction using a trained machine learning model and scaler
- Workout tracker that saves sessions to CSV
- Workout history viewer with filtering and summaries
- Personalized recommendations for warm-up, nutrition, recovery, and progression
- Responsive, polished UI with a professional theme

## Project Structure

```text
FitnessAgent/
├── app.py
├── fitness_model.pkl
├── scaler.pkl
├── workout_history.csv
├── recommendation.py
├── utils.py
├── requirements.txt
├── README.md
└── assets/
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```

## Notes

- Ensure that the trained pickle files named `fitness_model.pkl` and `scaler.pkl` are present in the project root.
- If the model files are missing, the app will display a helpful error message and continue gracefully.
- You can customize the input fields and recommendation logic in [utils.py](utils.py) and [recommendation.py](recommendation.py).
