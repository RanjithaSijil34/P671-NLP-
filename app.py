import streamlit as st
import pickle
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR, "best_model.pkl"), "rb"))
tfidf = pickle.load(open(os.path.join(BASE_DIR, "tfidf.pkl"), "rb"))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# =========================
# SIDEBAR NAVIGATION
# =========================
page = st.sidebar.selectbox("Navigation", 
                           ["Home", "Sentiment Analysis", "Model Performance", "About"])

# =========================
# HOME PAGE
# =========================
if page == "Home":
    st.title("🛍️ Product Review Sentiment Analysis")
    st.write("Welcome to the NLP project!")
    st.write("This app analyzes customer reviews and predicts sentiment.")

# =========================
# SENTIMENT PAGE
# =========================
elif page == "Sentiment Analysis":
    st.title("🔍 Analyze Sentiment")

    user_input = st.text_area("Enter Review")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a review")
        else:
            cleaned = clean_text(user_input)
            vector = tfidf.transform([cleaned]).toarray()
            prediction = model.predict(vector)

            st.success(f"Sentiment: {prediction[0]}")

# =========================
# MODEL PERFORMANCE PAGE
# =========================
elif page == "Model Performance":
    st.title("📊 Model Comparison")

    st.write("""
    Models Used:
    - Logistic Regression
    - Naive Bayes
    - SVM ✅ (Best)
    - Random Forest
    - KNN
    """)

    st.write("SVM performed best due to high-dimensional TF-IDF features.")

# =========================
# ABOUT PAGE
# =========================
elif page == "About":
    st.title("ℹ️ About Project")

    st.write("""
    **Project:** Sentiment Analysis on Customer Reviews  
    **Tech Stack:** Python, NLP, TF-IDF, Machine Learning  
    **Deployment:** Streamlit  

    This project analyzes customer reviews and classifies them into:
    - Positive
    - Negative
    - Neutral
    """)
