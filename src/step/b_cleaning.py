import os
import sys

import pandas as pd


from error_logs import configure_logger
logger = configure_logger()


def clean_data (data: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("starting data cleaning process.")
        data_clean = data.copy()


        #Handling Duplicate Rows
        duplicate_count = data_clean.duplicated().sum()
        if duplicate_count > 0:
            data_clean = data_clean.drop_duplicates()
            logger.info(f"Removed {duplicate_count} duplicate rows.")
        else:
            logger.info("No duplicate rows found.")
        
        #Handling Missing Values

        if data_clean['parental_education_level'].isnull().any():
            data_clean['parental_education_level'] = data_clean['parental_education_level'].fillna('Unknown')
            logger.info("Filled missing values in 'parental_education_level' with 'Unknown'.")
        else:
            logger.info("No missing values in 'parental_education_level'.")

        # Final Integrity Check
        remaining_nulls = data_clean.isnull().sum().sum()

                # Structured Summary
        summary = (
            f"\n{'='*30}\n"
            f"DATA CLEANING REPORT\n"
            f"{'='*30}\n"
            f"Rows: {data_clean.shape[0]} | Columns: {data_clean.shape[1]}\n"
            f"Duplicates Removed: {duplicate_count}\n"
            f"Remaining Nulls: {remaining_nulls}\n"
            f"Data Cleanliness: {'PASS' if remaining_nulls == 0 else 'FAIL'}\n"
            f"{'='*30}"
        )
        logger.info(summary)
        logger.info("==> Data cleaning process completed successfully.\n\n")
        return data_clean
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        return None

#python -m src.step.b_cleaning

