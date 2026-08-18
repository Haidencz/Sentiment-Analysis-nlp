import streamlit as st
import joblib
from PIL import Image
positive_image = Image.open("assets/positive1.png")
negative_image = Image.open("assets/negative.png")

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
st.title("Movie Review sentiment Analyser")
review = st.text_area("enter a movie review and the model will try to predict whether the sentiment is positive or negative")
#st.write("Enter a movie review and the model will try to predict whether the sentiment is positive or negative")
if st.button("Analyse"):
    if not review.strip():
        st.warning("please enter a review first.")
    else:
        review_vector = vectorizer.transform([review])
        probabilities = model.predict_proba(review_vector)[0]
        prediction = model.predict(review_vector)
        confidence = max(probabilities) * 100
        negative_prob = probabilities[0] * 100
        positive_prob = probabilities[1] * 100
        if prediction == 1:
            st.success(f"positive review {confidence: .1f}% confidence")
            st.image(positive_image, caption="positive!!!", width=200)
        else:
            st.error(f"negative review {confidence: .1f}% confidence")
            st.image(negative_image, caption="negative!!!", width=200)
        st.write(f"positive probability: {positive_prob:.1f}%")
        st.write(f"negative probability: {negative_prob:.1f}%")