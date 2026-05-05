import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

model = pickle.load(open("best_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

st.title("🛍️ Product Review Sentiment Analysis")

user_input = st.text_area("Enter your review:")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vector = tfidf.transform([cleaned]).toarray()
    prediction = model.predict(vector)

    st.success(f"Sentiment: {prediction[0]}")
