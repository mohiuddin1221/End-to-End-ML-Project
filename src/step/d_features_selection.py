import pandas as pd

from error_logs import configure_logger

logger = configure_logger()

def feature_selection(data: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Starting feature selection process.")
        target = 'exam_score'

        # Correlation Data
        corr_matrix = data.corr()
        target_corr = corr_matrix['exam_score'].sort_values(ascending=False)
        selected_features = target_corr[target_corr > 0.05].index.tolist()
        # Create the final dataframe with only selected columns
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



    