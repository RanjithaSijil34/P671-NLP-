import streamlit as st
import pickle
import re
import nltk
import os
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# =========================
# LOAD FILES
# =========================
BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR, "best_model.pkl"), "rb"))
tfidf = pickle.load(open(os.path.join(BASE_DIR, "tfidf.pkl"), "rb"))

# Load dataset for EDA
df = pd.read_excel(os.path.join(BASE_DIR, "dataset.xlsx"))

# normalize column names
df.columns = df.columns.str.lower()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# =========================
# CLEAN FUNCTION
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# =========================
# SIDEBAR NAVIGATION
# =========================
page = st.sidebar.selectbox(
    "Navigation",
    ["Home", "EDA", "Sentiment Analysis", "Model Performance"]
)

# =========================
# HOME
# =========================
if page == "Home":
    st.title("🛍️ Product Review Sentiment Analysis")
    st.write("Analyze customer reviews using NLP and Machine Learning.")

# =========================
# EDA PAGE
# =========================
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # -------------------------
    # Missing values
    # -------------------------
    st.write("### Missing Values")
    st.write(df.isnull().sum())

    # -------------------------
    # Review Length
    # -------------------------
    if 'review' in df.columns:
        df['review_length'] = df['review'].apply(lambda x: len(str(x)))

        st.write("### Review Length Distribution")
        fig1, ax1 = plt.subplots()
        ax1.hist(df['review_length'], bins=30)
        ax1.set_title("Review Length")
        st.pyplot(fig1)

    # -------------------------
    # Word Count
    # -------------------------
    if 'review' in df.columns:
        df['word_count'] = df['review'].apply(lambda x: len(str(x).split()))

        st.write("### Word Count Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(df['word_count'], bins=30)
        ax2.set_title("Word Count")
        st.pyplot(fig2)

    # -------------------------
    # Rating Distribution
    # -------------------------
    if 'rating' in df.columns:
        st.write("### Rating Distribution")
        fig3, ax3 = plt.subplots()
        df['rating'].value_counts().sort_index().plot(kind='bar', ax=ax3)
        ax3.set_title("Ratings")
        st.pyplot(fig3)

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
# MODEL PERFORMANCE
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
    Logistic Regression performed best due to:
    - Efficient handling of TF-IDF features  
    - Good generalization  
    - Simplicity and interpretability  
    """)
