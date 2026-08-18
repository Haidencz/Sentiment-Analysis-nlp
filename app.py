import streamlit as st
import joblib

model = joblib.load("sentiment_model.pk1")
vectorizer = joblib.load("tfidf_vectorizer.pk1")
st.title("Movie Review sentiment Analyser")
review = st.text_area("enter a movie review")
if st.button("Analyse"):
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)
    if prediction == 1:
        st.success("positive review")
    else:
        st.error("negative review")