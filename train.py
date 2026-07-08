"""
=========================================================
File        : train.py
Project     : Fake News Detection

Description:
------------
Train a Linear SVM model using TF-IDF features.

Input:
    data/train.csv

Output:
    models/tfidf_vectorizer.pkl
    models/fake_news_model.pkl
=========================================================
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from utils import (
    load_csv,
    save_model,
    create_directory,
    get_logger
)

logger = get_logger(__name__)

# ==========================================================
# Configuration
# ==========================================================

TRAIN_FILE = "data/train.csv"

MODEL_FOLDER = "models"

VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"

MODEL_FILE = "models/fake_news_model.pkl"

TEXT_COLUMN = "content"

TARGET_COLUMN = "label"


# ==========================================================
# Train Model
# ==========================================================

def train_model():

    logger.info("Loading training dataset...")

    train_df = load_csv(TRAIN_FILE)

    # ------------------------------------------------------
    # Remove missing values
    # ------------------------------------------------------

    train_df = train_df.dropna(
        subset=[TEXT_COLUMN, TARGET_COLUMN]
    )

    X_train = train_df[TEXT_COLUMN].astype(str)

    y_train = train_df[TARGET_COLUMN]

    logger.info("Creating TF-IDF Vectorizer...")

    vectorizer = TfidfVectorizer(
        lowercase=False,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=20000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    logger.info("Training Linear SVM model...")

    model = LinearSVC(
        C=2.0,
        class_weight="balanced",
        random_state=42,
        max_iter=5000
    )

    model.fit(X_train_tfidf, y_train)

    # ------------------------------------------------------
    # Save Model
    # ------------------------------------------------------

    create_directory(MODEL_FOLDER)

    save_model(vectorizer, VECTORIZER_FILE)

    save_model(model, MODEL_FILE)

    logger.info("Training completed successfully.")

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)

    print(f"Training Samples   : {len(train_df)}")
    print(f"Vocabulary Size    : {len(vectorizer.vocabulary_)}")
    print(f"Number of Classes  : {len(model.classes_)}")

    print("\nClasses")

    class_names = {
        0: "Fake News",
        1: "Real News"
    }

    for cls in model.classes_:
        print(f"{cls} -> {class_names.get(cls, cls)}")

    print("\nModel Saved To")
    print(f"Vectorizer : {VECTORIZER_FILE}")
    print(f"Model      : {MODEL_FILE}")


# ==========================================================
# Main
# ==========================================================

def main():

    train_model()


if __name__ == "__main__":

    main()