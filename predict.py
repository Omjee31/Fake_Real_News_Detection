"""
=========================================================
File        : predict.py
Project     : Fake News Detection

Description:
------------
Predict whether a news article is Fake or Real
using the trained TF-IDF + Linear SVM model.

Input:
    News Title
    News Content

Output:
    Fake News / Real News
=========================================================
"""

import re
import string

from utils import (
    load_model,
    get_logger
)

logger = get_logger(__name__)

# ==========================================================
# Configuration
# ==========================================================

MODEL_FILE = "models/fake_news_model.pkl"

VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"

# ==========================================================
# Load Saved Model
# ==========================================================

logger.info("Loading trained model...")

vectorizer = load_model(VECTORIZER_FILE)

model = load_model(MODEL_FILE)

logger.info("Model loaded successfully.")

# ==========================================================
# Text Cleaning
# ==========================================================

def clean_text(text):
    """
    Apply the same preprocessing used during training.
    """

    if text is None:
        return ""

    text = str(text).lower()

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove Email
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove Numbers
    text = re.sub(r"\d+", " ", text)

    # Remove Punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove Extra Spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# Prediction Function
# ==========================================================

def predict_news(title, article):
    """
    Predict Fake or Real News.

    Parameters
    ----------
    title : str
        News headline

    article : str
        News body

    Returns
    -------
    dict
        prediction
        label
        decision_score
    """

    content = f"{title} {article}"

    content = clean_text(content)

    vector = vectorizer.transform([content])

    prediction = model.predict(vector)[0]

    score = float(model.decision_function(vector)[0])

    if prediction == 1:
        label = "Real News"
    else:
        label = "Fake News"

    return {
        "prediction": int(prediction),
        "label": label,
        "decision_score": score
    }


# ==========================================================
# CLI
# ==========================================================

def main():

    print("\n" + "=" * 60)
    print("FAKE NEWS DETECTOR")
    print("=" * 60)

    title = input("\nEnter News Title:\n")

    article = input("\nEnter News Content:\n")

    result = predict_news(title, article)

    print("\n" + "=" * 60)
    print("Prediction Result")
    print("=" * 60)

    print(f"Prediction     : {result['label']}")
    print(f"Class          : {result['prediction']}")
    print(f"Decision Score : {result['decision_score']:.4f}")

    if result["prediction"] == 1:
        print("\nThis article is predicted as REAL NEWS.")
    else:
        print("\nThis article is predicted as FAKE NEWS.")

    print("=" * 60)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    main()