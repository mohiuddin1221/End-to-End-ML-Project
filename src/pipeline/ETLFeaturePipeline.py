from config import DATA_SOURCE

from src.step.a_ingest import ingest_data
from src.step.b_cleaning import clean_data
from src.step.c_feature_engineering import feature_engineering
from src.step.d_features_selection import feature_selection
from src.step.e_model_training import model_training

from error_logs import configure_logger
logger = configure_logger()


def run_pipeline():
    """
    Run the end-to-end ETL and feature pipeline for the ML project.
    """
    try:
        logger.info("Starting ETL and Feature Pipeline.")

        # Data Ingestion    
        data = ingest_data(data_source = DATA_SOURCE)
        if data is None or data.empty:
            logger.error("Ingested data is empty. Exiting pipeline.")
            return
        
        # Data Cleaning
        cleaned_data = clean_data(data)

        # Feature Engineering
        featured_data = feature_engineering(cleaned_data)

        # Feature Selection
        selected_data = feature_selection(featured_data)

        # Model Training
        model, scaler, metrics = model_training(selected_data)
        logger.info("ETL and Feature Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Error in ETL and Feature Pipeline: {e}") 
        return e
    

if __name__ == "__main__":
    run_pipeline()


# Run: python -m src.pipeline.ETLFeaturePipeline