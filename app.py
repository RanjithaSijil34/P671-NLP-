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
                           ["Home", "Sentiment Analysis", "Model Performance"])

# =========================
# HOME PAGE
# =========================
if page == "Home":
    st.title("🛍️ Product Review Sentiment Analysis")
    st.write("This application predicts sentiment from customer reviews.")
    st.write("Built using NLP and Machine Learning.")

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
    - Logistic Regression ✅ (Best Model)
    - Naive Bayes
    - SVM
    - Random Forest
    - KNN
    """)

    st.write("""
    Logistic Regression performed best on this dataset due to:
    - Efficient handling of TF-IDF features  
    - Good generalization performance  
    - Simplicity and interpretability  
    """)
