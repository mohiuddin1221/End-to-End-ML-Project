import joblib
from fastapi import APIRouter, FastAPI

import pandas as pd


from .schemas import UserInputRequest


app = FastAPI()


model = joblib.load("./models/best_model.pkl")
scaler = joblib.load("./models/scaler.pkl")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/predict")
async def predict_score(user_input: UserInputRequest):

    inputs = {
        "study_hours_per_day": user_input.study,
        "social_media_hours": user_input.social,
        "netflix_hours": user_input.netflix,
        "attendance_percentage": user_input.attendance,
        "sleep_hours": user_input.sleep,
        "exercise_frequency": user_input.exercise,
        "mental_health_rating": user_input.mental,
    }

    input_data = pd.DataFrame([inputs])
    expected_columns = [
        "study_hours_per_day",
        "social_media_hours",
        "netflix_hours",
        "attendance_percentage",
        "sleep_hours",
        "exercise_frequency",
        "mental_health_rating",
    ]

    input_data = input_data[expected_columns]

    Scaler_Data = scaler.transform(input_data)
    prediction = model.predict(Scaler_Data)[0]
    final_score = max(0, min(100, prediction))

    return {"predicted_score": final_score, "success": True}


# Run the API server
# uvicorn src.web.api:app --reload