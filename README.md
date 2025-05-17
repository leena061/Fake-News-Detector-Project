# Fake News Detection Web App

This project implements a Machine Learning model to classify news articles as either **Fake** or **True** based on their textual content.

You can interact with the deployed web application here:

**[https://fake-news-detector-project-6i9cdgtgavkqmteg9frjhq.streamlit.app/]**

## Project Overview
*   Objective:To build a reliable model capable of distinguishing between fake and true news articles.
*   Methodology:
    *   Data loading and cleaning of news articles.
    *   Text preprocessing (lowercasing, punctuation removal, stopword removal).
    *   Feature extraction using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
    *   Training a **Logistic Regression** classification model.
    *   Deployment as an interactive web application using **Streamlit**.
## Repository Contents
*   `app.py`: The Streamlit application script that runs the web app.
*   `requirements.txt`: Lists all the Python libraries required to run the app.
*   `trained_model/`: Directory containing the saved machine learning model (`logistic_regression_model.joblib`) and the fitted TF-IDF vectorizer (`tfidf_vectorizer.joblib`).
*   `README.md`: This file.

## How to Run Locally 
1.  Clone this repository
2.  Ensure you have Python installed.
3.  Install the required libraries
4.  Download NLTK stopwords.
5.  Make sure the `trained_model` directory containing the `.joblib` files is in the same directory as `app.py`.
6.  Run the Streamlit application.
