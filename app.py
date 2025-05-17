# %%capture
# Install Streamlit and other necessary libraries if not already installed
# You might have these installed from your notebook, but include them here
# for a self-contained app environment.
# !pip install streamlit joblib scikit-learn nltk pandas

import streamlit as st
import joblib
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)


# Define the path to your saved model and vectorizer
# Make sure these paths are correct relative to where your app.py file is
MODEL_PATH = 'trained_model/logistic_regression_model.joblib'
VECTORIZER_PATH = 'trained_model/tfidf_vectorizer.joblib'

# Load the model and vectorizer
# Use st.cache_resource to load once and cache the results

@st.cache_resource
def load_model(model_path):
    """Loads the trained model."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_vectorizer(vectorizer_path):
    """Loads the fitted TF-IDF vectorizer."""
    try:
        vectorizer = joblib.load(vectorizer_path)
        return vectorizer
    except FileNotFoundError:
        st.error(f"Vectorizer file not found at: {vectorizer_path}")
        return None
    except Exception as e:
        st.error(f"Error loading vectorizer: {e}")
        return None

# Load the model and vectorizer when the app starts
model = load_model(MODEL_PATH)
vectorizer = load_vectorizer(VECTORIZER_PATH)

# Define the preprocessing function (same as in your notebook)
# It's important to use the EXACT same preprocessing steps
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    stop_words_set = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words_set])

    return text

# --- Streamlit App UI ---
st.title("Fake News Detection App")
st.write("Enter the text of a news article below to determine if it's likely fake or true.")

# Text input area
article_text = st.text_area("Enter news article text here:", height=300)

# Prediction button
if st.button("Predict"):
    if article_text:
        # Perform preprocessing and prediction
        cleaned_text = preprocess_text(article_text)

       
        if vectorizer is not None and model is not None:
  # Transform the cleaned text using the loaded vectorizer
            text_vector = vectorizer.transform([cleaned_text]) # Vectorizer expects an iterable

    # Make prediction
    prediction = model.predict(text_vector)

    # --- Add this line ---
    st.write("Raw prediction output:")
    st.write(prediction) # This will show something like [0] or [1]
    # --- End Addition ---

    # Display result
    st.subheader("Prediction:")
    if prediction[0] == 0: # Check if this condition is always met
        st.error("This article is predicted as: **FAKE NEWS**")
    else: # Check if this condition is never met
        st.success("This article is predicted as: **TRUE NEWS**")
            

# Optional: Add some info about the model
st.markdown("---")
st.write("Model used: Logistic Regression")
st.write("Feature Extraction: TF-IDF")