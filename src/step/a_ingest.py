import os
import pandas as pd
from pathlib import Path


from config import DATA_SOURCE
from error_logs import configure_logger

logger = configure_logger()


def ingest_data(data_source: str = DATA_SOURCE) -> pd.DataFrame:

    try:
        logger.info("Starting data ingestion process.")

        # Convert string path to Path object for robust checking
        data_path = Path(DATA_SOURCE)

        if not data_path.exists():
            logger.error(f"Data source not found at {data_source}")
            raise FileNotFoundError(f"Data source not found at {data_source}")

        data = pd.read_csv(data_source)
        summary = (
            f"\n{'='*30}\n"
            f"DATA INGESTION REPORT\n"
            f"{'='*30}\n"
            f"Source: {data_path.name}\n"
            f"Rows: {data.shape[0]} | Columns: {data.shape[1]}\n"
            f"File Integrity: {'PASS' if not data.empty else 'FAIL'}\n"
            f"Columns Found: {data.columns.tolist()[:5]}... (Total {len(data.columns)})\n"
            f"{'='*30}"
        )
        logger.info(summary)
        logger.info("==> Data ingestion process completed successfully.\n\n")
        return data

    except Exception as e:
        logger.exception("Data ingestion failed due to an error.")
        raise e


# if __name__ == "__main__":
#     data = ingest_data(data_source = DATA_SOURCE)
#     print(data.head())
#     print(data.isnull().sum().value_counts())
#     print(data.duplicated().sum())
#     print(f"Data shape: {data.shape}")


#python -m src.step.a_ingest