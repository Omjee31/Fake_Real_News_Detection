"""
=========================================================
File        : preprocess.py
Project     : Fake News Detection

Description:
------------
Preprocess Fake and Real news datasets.

Input:
    data/Fake.csv
    data/True.csv

Output:
    data/processed_news.csv
=========================================================
"""

import re
import string
import pandas as pd

from utils import (
    load_csv,
    save_csv,
    dataset_info,
    get_logger
)

logger = get_logger(__name__)


# ==========================================================
# Configuration
# ==========================================================

FAKE_FILE = "data/Fake.csv"

TRUE_FILE = "data/True.csv"

OUTPUT_FILE = "data/processed_news.csv"


# ==========================================================
# Text Cleaner
# ==========================================================

class TextCleaner:

    def remove_html(self, text):

        return re.sub(r"<.*?>", " ", text)

    def remove_urls(self, text):

        return re.sub(r"http\S+|www\S+", " ", text)

    def remove_email(self, text):

        return re.sub(r"\S+@\S+", " ", text)

    def remove_numbers(self, text):

        return re.sub(r"\d+", " ", text)

    def remove_punctuation(self, text):

        return text.translate(
            str.maketrans("", "", string.punctuation)
        )

    def clean_text(self, text):

        if pd.isna(text):
            return ""

        text = str(text).lower()

        text = self.remove_html(text)

        text = self.remove_urls(text)

        text = self.remove_email(text)

        text = self.remove_numbers(text)

        text = self.remove_punctuation(text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()


# ==========================================================
# Main Processing
# ==========================================================

def preprocess():

    logger.info("Loading datasets...")

    fake_df = load_csv(FAKE_FILE)

    true_df = load_csv(TRUE_FILE)

    logger.info("Assigning labels...")

    fake_df["label"] = 0

    true_df["label"] = 1

    logger.info("Combining datasets...")

    df = pd.concat(
        [fake_df, true_df],
        ignore_index=True
    )

    logger.info("Removing duplicate news...")

    df.drop_duplicates(
        subset=["title", "text"],
        inplace=True
    )

    logger.info("Creating content column...")

    df["content"] = (
        df["title"].fillna("").astype(str)
        + " "
        + df["text"].fillna("").astype(str)
    )

    cleaner = TextCleaner()

    logger.info("Cleaning news text...")

    df["content"] = df["content"].apply(
        cleaner.clean_text
    )

    logger.info("Removing empty rows...")

    df = df[df["content"].str.strip() != ""]

    logger.info("Shuffling dataset...")

    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    logger.info("Saving processed dataset...")

    save_csv(df, OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)

    dataset_info(df)

    print("\nClass Distribution")
    print("-" * 60)
    print(df["label"].value_counts())

    print("\nSample News")
    print("-" * 60)
    print(df[["label", "content"]].head())


# ==========================================================
# Main
# ==========================================================

def main():

    preprocess()


if __name__ == "__main__":

    main()
    