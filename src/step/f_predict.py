import joblib
import pandas as pd

from error_logs import configure_logger

logger = configure_logger()


def model_predict():
    try:
        model = joblib.load("./models/best_model.pkl")
        scaler = joblib.load("./models/scaler.pkl")

        print("\n" + "=" * 30)
        print("STUDENT SCORE PREDICTOR")
        print("=" * 30)

        study = float(input("Study Hours/Day (0-16): "))
        social = float(input("Social Media Hours (0-12): "))
        netflix = float(input("Netflix Hours (0-12): "))
        attendance = float(input("Attendance % (1-100): "))
        sleep = float(input("Sleep Hours (1-12): "))
        exercise = float(input("Exercise (1-5): "))
        mental = float(input("Mental Health (1-10): "))

        inputs = {
            "study_hours_per_day": study,
            "social_media_hours": social,
            "netflix_hours": netflix,
            "attendance_percentage": attendance,
            "sleep_hours": sleep,
            "exercise_frequency": exercise,
            "mental_health_rating": mental,
        }

        input_data = pd.DataFrame([inputs])

        # column order
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

        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]

        final_score = max(0, min(100, prediction))

        print("\n" + "-" * 30)
        print(f"RESULT: Estimated Exam Score: {final_score:.2f}%")
        if prediction >= 50:
            print("Status: PASS! Congratulations on your predicted success!")
        else:
            print(
                "your selected values indicate you might need to work harder to pass. Keep pushing!"
            )
        print("-" * 30)

    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        return


if __name__ == "__main__":
    model_predict()

# Run: python -m src.step.f_predict
