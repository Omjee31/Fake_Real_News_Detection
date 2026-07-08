"""
=========================================================
File        : eda.py
Project     : Fake News Detection

Description:
------------
Perform Exploratory Data Analysis (EDA) on the
processed news dataset and save plots.

Input:
    data/processed_news.csv

Output:
    plots/
=========================================================
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from utils import (
    load_csv,
    create_directory,
    dataset_info,
    get_logger
)

logger = get_logger(__name__)

# ==========================================================
# Configuration
# ==========================================================

INPUT_FILE = "data/processed_news.csv"

PLOT_FOLDER = "plots"

TEXT_COLUMN = "content"

LABEL_COLUMN = "label"

SUBJECT_COLUMN = "subject"

# ==========================================================
# Create Plot Directory
# ==========================================================

create_directory(PLOT_FOLDER)

sns.set_style("whitegrid")

# ==========================================================
# Helper Function
# ==========================================================


def save_plot(filename):

    plt.tight_layout()

    plt.savefig(
        os.path.join(PLOT_FOLDER, filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# EDA
# ==========================================================

def perform_eda():

    logger.info("Loading processed dataset...")

    df = load_csv(INPUT_FILE)

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    dataset_info(df)

    # ------------------------------------------------------
    # Missing Values
    # ------------------------------------------------------

    print("\nMissing Values\n")

    print(df.isnull().sum())

    # ------------------------------------------------------
    # Duplicate Rows
    # ------------------------------------------------------

    print("\nDuplicate Rows")

    print(df.duplicated().sum())

    # ------------------------------------------------------
    # Class Distribution
    # ------------------------------------------------------

    plt.figure(figsize=(6,5))

    sns.countplot(
        data=df,
        x=LABEL_COLUMN
    )

    plt.xticks(
        [0,1],
        ["Fake","Real"]
    )

    plt.title("Class Distribution")

    save_plot("class_distribution.png")

    # ------------------------------------------------------
    # Subject Distribution
    # ------------------------------------------------------

    plt.figure(figsize=(10,6))

    sns.countplot(
        data=df,
        y=SUBJECT_COLUMN,
        order=df[SUBJECT_COLUMN].value_counts().index
    )

    plt.title("Subject Distribution")

    save_plot("subject_distribution.png")

    # ------------------------------------------------------
    # News Length
    # ------------------------------------------------------

    df["news_length"] = df[TEXT_COLUMN].apply(len)

    plt.figure(figsize=(8,5))

    sns.histplot(
        df["news_length"],
        bins=40,
        kde=True
    )

    plt.title("News Length Distribution")

    save_plot("news_length_distribution.png")

    # ------------------------------------------------------
    # Box Plot
    # ------------------------------------------------------

    plt.figure(figsize=(6,5))

    sns.boxplot(
        y=df["news_length"]
    )

    plt.title("News Length Boxplot")

    save_plot("news_length_boxplot.png")

    # ------------------------------------------------------
    # Top Words
    # ------------------------------------------------------

    words = " ".join(df[TEXT_COLUMN]).split()

    common_words = Counter(words).most_common(20)

    common_df = pd.DataFrame(
        common_words,
        columns=["Word","Frequency"]
    )

    plt.figure(figsize=(12,6))

    sns.barplot(
        data=common_df,
        x="Frequency",
        y="Word"
    )

    plt.title("Top 20 Frequent Words")

    save_plot("top_words.png")

    # ------------------------------------------------------
    # Fake WordCloud
    # ------------------------------------------------------

    fake_text = " ".join(
        df[df[LABEL_COLUMN]==0][TEXT_COLUMN]
    )

    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    ).generate(fake_text)

    plt.figure(figsize=(12,6))

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.title("Fake News WordCloud")

    save_plot("fake_wordcloud.png")

    # ------------------------------------------------------
    # Real WordCloud
    # ------------------------------------------------------

    real_text = " ".join(
        df[df[LABEL_COLUMN]==1][TEXT_COLUMN]
    )

    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    ).generate(real_text)

    plt.figure(figsize=(12,6))

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.title("Real News WordCloud")

    save_plot("real_wordcloud.png")

    # ------------------------------------------------------
    # Correlation
    # ------------------------------------------------------

    corr = df[["label","news_length"]].corr()

    plt.figure(figsize=(5,4))

    sns.heatmap(
        corr,
        annot=True,
        cmap="Blues"
    )

    plt.title("Correlation Heatmap")

    save_plot("correlation_heatmap.png")

    logger.info("EDA Completed Successfully.")

    print("\n" + "=" * 60)
    print("EDA COMPLETED")
    print("=" * 60)

    print(f"\nPlots saved inside '{PLOT_FOLDER}/' folder.")


# ==========================================================
# Main
# ==========================================================

def main():

    perform_eda()


if __name__ == "__main__":

    main()