import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from error_logs import configure_logger

logger = configure_logger()


def model_training(data: pd.DataFrame):
    try:
        logger.info("Starting model training process.")
        feature = data.drop(columns=["exam_score"])
        target = data["exam_score"]

        X_train, X_test, y_train, y_test = train_test_split(
            feature, target, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        models = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, max_depth=5, random_state=42
            ),
            "xgiboost": XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ),
        }
        results = {}

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            accuracy = model.score(X_test_scaled, y_test)

            results[name] = {
                "model": model,
                "MAE": mae,
                "MSE": mse,
                "R2": r2,
                "Accuracy": accuracy,
            }

        best_model_name = max(results, key=lambda x: results[x]["R2"])
        best_model_metrics = results[best_model_name]

        summary = (
            f"\n{'='*30}\n"
            f"MODEL TRAINING REPORT\n"
            f"{'='*30}\n"
            f"Best Model:       {best_model_name}\n"
            f"R² Score:         {best_model_metrics['R2']:.4f}\n"
            f"Mean Absolute Error: {best_model_metrics['MAE']:.4f}\n"
            f"Root Mean Squared Error: {best_model_metrics['MSE']:.4f}\n"
            f"{'='*30}"
        )
        logger.info(summary)
        logger.info("==> Model Training completed successfully.\n\n")

        model_path = os.path.join("models", "best_model.pkl")
        scaler_path = os.path.join("models", "scaler.pkl")

        joblib.dump(best_model_metrics["model"], model_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Best model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")

        return best_model_metrics["model"], scaler, best_model_metrics
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return None, None, None


# python -m src.step.e_model_training
