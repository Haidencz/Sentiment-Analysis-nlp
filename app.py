import streamlit as st
import joblib

model = joblib.load("sentiment_model.pk1")
vectorizer = joblib.load("tfidf_vectorizer.pk1")
st.title("Movie Review sentiment Analyser")
review = st.text_area("enter a movie review")
if st.button("Analyse"):
    review_vector = vectorizer.transform([review])
    probabilities = model.predict_proba(review_vector)[0]
    prediction = model.predict(review_vector)
    confidence = max(probabilities) * 100
    if prediction == 1:
        st.success(f"positive review {confidence: .1f}% confidence")
    else:
        st.error(f"negative review {confidence: .1f}% confidence")