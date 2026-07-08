"""
=========================================================
File        : utils.py
Project     : Fake News Detection

Description:
------------
This module contains reusable utility functions used
throughout the project.

Functions:
    - create_directory()
    - get_logger()
    - load_csv()
    - save_csv()
    - dataset_info()
    - save_model()
    - load_model()
=========================================================
"""

import logging
from pathlib import Path
import joblib
import pandas as pd


# ==========================================================
# Logger
# ==========================================================

def get_logger(name: str = __name__) -> logging.Logger:
    """
    Create and return a logger.
    """

    logger = logging.getLogger(name)

    if not logger.handlers:

        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        console_handler = logging.StreamHandler()

        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger


logger = get_logger()


# ==========================================================
# Directory
# ==========================================================

def create_directory(path):
    """
    Create directory if it does not exist.
    """

    Path(path).mkdir(parents=True, exist_ok=True)


# ==========================================================
# CSV Functions
# ==========================================================

def load_csv(file_path):
    """
    Load CSV file.
    """

    try:

        df = pd.read_csv(file_path)

        logger.info(f"Loaded CSV : {file_path}")

        return df

    except Exception as e:

        logger.error(e)

        raise


def save_csv(df, file_path):
    """
    Save DataFrame as CSV.
    """

    try:

        create_directory(Path(file_path).parent)

        df.to_csv(file_path, index=False)

        logger.info(f"Saved CSV : {file_path}")

    except Exception as e:

        logger.error(e)

        raise


# ==========================================================
# Dataset Information
# ==========================================================

def dataset_info(df):
    """
    Display dataset information.
    """

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(f"Rows              : {df.shape[0]}")
    print(f"Columns           : {df.shape[1]}")

    print("\nColumn Names")
    print("-" * 60)
    print(df.columns.tolist())

    print("\nMissing Values")
    print("-" * 60)
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print("-" * 60)
    print(df.duplicated().sum())

    print("\nData Types")
    print("-" * 60)
    print(df.dtypes)

    print("\nFirst Five Rows")
    print("-" * 60)
    print(df.head())

    print("=" * 60)


# ==========================================================
# Model Functions
# ==========================================================

def save_model(model, file_path):
    """
    Save ML model.
    """

    try:

        create_directory(Path(file_path).parent)

        joblib.dump(model, file_path)

        logger.info(f"Model Saved : {file_path}")

    except Exception as e:

        logger.error(e)

        raise


def load_model(file_path):
    """
    Load ML model.
    """

    try:

        model = joblib.load(file_path)

        logger.info(f"Model Loaded : {file_path}")

        return model

    except Exception as e:

        logger.error(e)

        raise

print("Complete Run ")