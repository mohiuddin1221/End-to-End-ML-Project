import os
import pandas as pd

from error_logs import configure_logger

logger = configure_logger()


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Starting feature engineering process.")
        data_feature = data.copy()

        # Ordinal Encoding (Mapped by Rank/Logic)
        ordinal_mappings = {
            "parental_education_level": {
                "Unknown": 0,
                "High School": 1,
                "Bachelor": 2,
                "Master": 3,
            },
            "internet_quality": {"Poor": 1, "Average": 2, "Good": 3},
            "diet_quality": {"Poor": 1, "Fair": 2, "Good": 3},
        }

        # Apply the mapping to each column
        for col, mapping in ordinal_mappings.items():
            data_feature[col] = data_feature[col].map(mapping)
        logger.info("==> Ordinal Encoding complete for Education, Internet, and Diet.")

        # Binary Encoding (Yes/No to 1/0)
        binary_encoding_config = {
            "part_time_job": {"No": 0, "Yes": 1},
            "extracurricular_participation": {"No": 0, "Yes": 1},
        }
        for col, mapping in binary_encoding_config.items():
            data_feature[col] = data_feature[col].map(mapping)
        logger.info(
            "==> Binary Encoding complete for Part-Time Job and Extracurricular Participation."
        )

        # Apply One-Hot Encoding
        data_feature = pd.get_dummies(
            data_feature, columns=["gender"], drop_first=True, dtype=int
        )

        # Drop Unnecessary Columns
        data_feature.drop(columns=["student_id"], inplace=True)
        data_feature.head()
        logger.info("==> One-Hot Encoding complete for Gender.")

        summary = (
            f"\n{'='*30}\n"
            f"FEATURE ENGINEERING REPORT\n"
            f"{'='*30}\n"
            f"Rows: {data_feature.shape[0]} | Columns: {data_feature.shape[1]}\n"
            f"Dtypes: {data_feature.dtypes.value_counts().to_dict()}\n"
            f"Columns: {data_feature.columns.tolist()}\n"
            f"{'='*30}"
        )
        logger.info(summary)
        logger.info("==> Feature engineering process completed successfully.\n\n")
        return data_feature
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        return None


# Run: python -m src.step.c_feature_engineering
