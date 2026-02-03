import pandas as pd

from error_logs import configure_logger

logger = configure_logger()


def feature_selection(data: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Starting feature selection process.")
        
        target = "exam_score"

        # Get absolute correlation values for the target
        correlations = data.corr()[target].abs()
        selected_features = correlations[correlations > 0.05].index.tolist()

        df_final = data[selected_features]

        summary = (
            f"\n{'='*30}\n"
            f"FEATURE SELECTION REPORT\n"
            f"{'='*30}\n"
            f"Original Features: {data.shape[1]}\n"
            f"Selected Features: {df_final.shape[1]}\n"
            f"Target Variable:   {target}\n"
            f"Final Columns:     {df_final.columns.tolist()}\n"
            f"{'='*30}"
        )
        logger.info(summary)
        logger.info("==> Feature Selection completed successfully.\n\n")
        return df_final
    except Exception as e:
        logger.error(f"Error during feature selection: {e}")
        return None
