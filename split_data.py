"""
=========================================================
File        : split_data.py
Project     : Fake News Detection

Description:
------------
Split the processed dataset into training and testing sets.

Input:
    data/processed_news.csv

Output:
    data/train.csv
    data/test.csv
=========================================================
"""

from sklearn.model_selection import train_test_split

from utils import (
    load_csv,
    save_csv,
    get_logger
)

logger = get_logger(__name__)

# ==========================================================
# Configuration
# ==========================================================

INPUT_FILE = "data/processed_news.csv"

TRAIN_FILE = "data/train.csv"

TEST_FILE = "data/test.csv"

TEXT_COLUMN = "content"

TARGET_COLUMN = "label"

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ==========================================================
# Split Dataset
# ==========================================================

def split_dataset():

    logger.info("Loading processed dataset...")

    df = load_csv(INPUT_FILE)

    print("\n" + "=" * 60)
    print("ORIGINAL DATASET")
    print("=" * 60)

    print(f"Rows : {len(df)}")

    # ------------------------------------------------------
    # Remove missing values
    # ------------------------------------------------------

    logger.info("Removing missing values...")

    df = df.dropna(
        subset=[TEXT_COLUMN, TARGET_COLUMN]
    )

    # ------------------------------------------------------
    # Remove duplicate news articles
    # ------------------------------------------------------

    logger.info("Removing duplicate news...")

    df = df.drop_duplicates(
        subset=[TEXT_COLUMN]
    )

    df = df.reset_index(drop=True)

    print(f"Rows After Cleaning : {len(df)}")

    # ------------------------------------------------------
    # Train-Test Split
    # ------------------------------------------------------

    logger.info("Performing stratified train-test split...")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df[TARGET_COLUMN]
    )

    # ------------------------------------------------------
    # Save datasets
    # ------------------------------------------------------

    save_csv(train_df, TRAIN_FILE)

    save_csv(test_df, TEST_FILE)

    logger.info("Train and Test datasets saved successfully.")

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)

    print(f"Total Samples : {len(df)}")
    print(f"Train Samples : {len(train_df)}")
    print(f"Test Samples  : {len(test_df)}")

    print("\nTrain Shape :", train_df.shape)

    print("Test Shape  :", test_df.shape)

    # ------------------------------------------------------
    # Class Distribution
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAIN CLASS DISTRIBUTION")
    print("=" * 60)

    train_distribution = (
        train_df[TARGET_COLUMN]
        .value_counts()
        .rename(index={0: "Fake", 1: "Real"})
    )

    print(train_distribution)

    print("\nPercentage")

    print(
        (
            train_df[TARGET_COLUMN]
            .value_counts(normalize=True)
            .rename(index={0: "Fake", 1: "Real"})
            * 100
        ).round(2)
    )

    print("\n" + "=" * 60)
    print("TEST CLASS DISTRIBUTION")
    print("=" * 60)

    test_distribution = (
        test_df[TARGET_COLUMN]
        .value_counts()
        .rename(index={0: "Fake", 1: "Real"})
    )

    print(test_distribution)

    print("\nPercentage")

    print(
        (
            test_df[TARGET_COLUMN]
            .value_counts(normalize=True)
            .rename(index={0: "Fake", 1: "Real"})
            * 100
        ).round(2)
    )

    print("\n" + "=" * 60)
    print("Dataset Split Completed Successfully")
    print("=" * 60)


# ==========================================================
# Main
# ==========================================================

def main():

    split_dataset()


if __name__ == "__main__":

    main()