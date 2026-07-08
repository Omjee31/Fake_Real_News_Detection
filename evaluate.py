"""
=========================================================
File        : evaluate.py
Project     : Fake News Detection

Description:
------------
Evaluate the trained Linear SVM model.

Input:
    data/test.csv
    models/tfidf_vectorizer.pkl
    models/fake_news_model.pkl

Output:
    plots/confusion_matrix.png
    plots/roc_curve.png
    plots/precision_recall_curve.png
=========================================================
"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from utils import (
    load_csv,
    load_model,
    create_directory,
    get_logger
)

logger = get_logger(__name__)

# ==========================================================
# Configuration
# ==========================================================

TEST_FILE = "data/test.csv"

MODEL_FILE = "models/fake_news_model.pkl"

VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"

PLOT_FOLDER = "plots"

TEXT_COLUMN = "content"

TARGET_COLUMN = "label"

# ==========================================================
# Evaluate
# ==========================================================

def evaluate():

    logger.info("Loading test dataset...")

    test_df = load_csv(TEST_FILE)

    logger.info("Loading vectorizer...")

    vectorizer = load_model(VECTORIZER_FILE)

    logger.info("Loading trained model...")

    model = load_model(MODEL_FILE)

    X_test = test_df[TEXT_COLUMN].astype(str)

    y_test = test_df[TARGET_COLUMN]

    X_test_tfidf = vectorizer.transform(X_test)

    logger.info("Predicting labels...")

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report\n")

    print(classification_report(
        y_test,
        y_pred,
        target_names=[
            "Fake News",
            "Real News"
        ]
    ))

    # ======================================================
    # Confusion Matrix
    # ======================================================

    create_directory(PLOT_FOLDER)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Fake",
            "Real"
        ]
    )

    plt.figure(figsize=(6,6))

    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix")

    plt.savefig(
        f"{PLOT_FOLDER}/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ======================================================
    # ROC Curve
    # ======================================================

    scores = model.decision_function(X_test_tfidf)

    fpr, tpr, _ = roc_curve(
        y_test,
        scores
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(7,6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.3f}"
    )

    plt.plot(
        [0,1],
        [0,1],
        "--"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.savefig(
        f"{PLOT_FOLDER}/roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ======================================================
    # Precision Recall Curve
    # ======================================================

    precision_curve, recall_curve, _ = precision_recall_curve(
        y_test,
        scores
    )

    ap = average_precision_score(
        y_test,
        scores
    )

    plt.figure(figsize=(7,6))

    plt.plot(
        recall_curve,
        precision_curve,
        label=f"AP = {ap:.3f}"
    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")

    plt.legend()

    plt.savefig(
        f"{PLOT_FOLDER}/precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nPlots Saved")

    print("------------------------------")

    print("confusion_matrix.png")

    print("roc_curve.png")

    print("precision_recall_curve.png")

    logger.info("Evaluation completed successfully.")


# ==========================================================
# Main
# ==========================================================

def main():

    evaluate()


if __name__ == "__main__":

    main()