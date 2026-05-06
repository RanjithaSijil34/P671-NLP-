import streamlit as st
import pickle
import re
import nltk
import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# DOWNLOAD NLTK
# =========================
nltk.download('stopwords')
nltk.download('wordnet')

# =========================
# LOAD FILES
# =========================
BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR, "best_model.pkl"), "rb"))
tfidf = pickle.load(open(os.path.join(BASE_DIR, "tfidf.pkl"), "rb"))

# Load dataset (for EDA only)
df = pd.read_excel(os.path.join(BASE_DIR, "dataset.xlsx"))

# Normalize column names
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
# HOME PAGE
# =========================
if page == "Home":
    st.title("🛍️ Product Review Sentiment Analysis")
    st.write("This app predicts sentiment from customer reviews using NLP and Machine Learning.")

# =========================
# EDA PAGE (Graphs + WordCloud)
# =========================
elif page == "EDA":
    st.title("📊 Data Visualization")

    if 'review' not in df.columns:
        st.error("❌ 'review' column not found in dataset")
    else:
        # -------------------------
        # Word Count
        # -------------------------
        df['word_count'] = df['review'].apply(lambda x: len(str(x).split()))

        st.subheader("Word Count Distribution")
        fig1, ax1 = plt.subplots()
        ax1.hist(df['word_count'], bins=30)
        ax1.set_title("Word Count")
        st.pyplot(fig1)

        # -------------------------
        # Review Length
        # -------------------------
        df['review_length'] = df['review'].apply(lambda x: len(str(x)))

        st.subheader("Review Length Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(df['review_length'], bins=30)
        ax2.set_title("Review Length")
        st.pyplot(fig2)

        # -------------------------
        # WordCloud
        # -------------------------
        st.subheader("WordCloud")

        text = " ".join(df['review'].dropna().astype(str))

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white'
        ).generate(text)

        fig3, ax3 = plt.subplots()
        ax3.imshow(wordcloud, interpolation='bilinear')
        ax3.axis("off")

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
    Logistic Regression performed best because:
    - Works efficiently with TF-IDF features  
    - Good generalization  
    - Simple and interpretable  
    """)
