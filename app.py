"""
=========================================================
File        : app.py
Project     : Fake News Detection

Description:
------------
Streamlit Web Application for Fake News Detection.
=========================================================
"""

import os

import pandas as pd
import streamlit as st

from predict import predict_news

# ==========================================================
# Configuration
# ==========================================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("📰 Fake News Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset",
        "Prediction",
        "EDA",
        "Model Performance",
        "About"
    ]
)

# ==========================================================
# HOME
# ==========================================================

if page == "Home":

    st.title("📰 Fake News Detection using Machine Learning")

    st.markdown("---")

    st.write("""
This application detects whether a news article is **Fake** or **Real**
using **TF-IDF** and **Linear SVM**.

### Features

- News Classification
- Exploratory Data Analysis
- Model Performance
- Interactive Prediction

### Model

- TF-IDF Vectorizer
- Linear Support Vector Machine
- Accuracy ≈ 99%
""")

# ==========================================================
# DATASET
# ==========================================================

elif page == "Dataset":

    st.title("📊 Dataset Overview")

    df = pd.read_csv("data/processed_news.csv")

    st.write("### Dataset Shape")

    st.write(df.shape)

    st.write("### First Five Rows")

    st.dataframe(df.head())

    st.write("### Class Distribution")

    st.bar_chart(df["label"].value_counts())

# ==========================================================
# PREDICTION
# ==========================================================

elif page == "Prediction":

    st.title("🤖 Fake News Prediction")

    title = st.text_input(
        "News Title"
    )

    article = st.text_area(
        "News Content",
        height=250
    )

    if st.button("Predict"):

        if title.strip() == "" and article.strip() == "":

            st.warning("Please enter a news article.")

        else:

            result = predict_news(title, article)

            if result["prediction"] == 1:

                st.success("✅ Real News")

            else:

                st.error("❌ Fake News")

            st.write(
                f"Decision Score : **{result['decision_score']:.4f}**"
            )

# ==========================================================
# EDA
# ==========================================================

elif page == "EDA":

    st.title("📈 Exploratory Data Analysis")

    plot_folder = "plots"

    plots = [

        "class_distribution.png",

        "subject_distribution.png",

        "news_length_distribution.png",

        "news_length_boxplot.png",

        "top_words.png",

        "fake_wordcloud.png",

        "real_wordcloud.png",

        "correlation_heatmap.png",

        "confusion_matrix.png",

        "roc_curve.png",

        "precision_recall_curve.png"

    ]

    for plot in plots:

        path = os.path.join(
            plot_folder,
            plot
        )

        if os.path.exists(path):

            st.image(
                path,
                caption=plot.replace("_", " ").replace(".png", "")
            )

# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

elif page == "Model Performance":

    st.title("📉 Model Performance")

    st.metric(
        "Accuracy",
        "99%"
    )

    st.metric(
        "Precision",
        "99%"
    )

    st.metric(
        "Recall",
        "99%"
    )

    st.metric(
        "F1 Score",
        "99%"
    )

    st.success(
        "The model performs exceptionally well on the test dataset."
    )

# ==========================================================
# ABOUT
# ==========================================================

elif page == "About":

    st.title("ℹ️ About")

    st.write("""
### Fake News Detection

This project was developed using

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Linear SVM
- Streamlit

### Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis
4. Train/Test Split
5. Model Training
6. Model Evaluation
7. News Prediction
8. Streamlit Deployment

Created as an end-to-end Machine Learning project.
""")