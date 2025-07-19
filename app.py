import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Prediction function
def predict_news(text):
    vect = vectorizer.transform([text])
    prediction = model.predict(vect)[0]
    confidence = max(model.predict_proba(vect)[0]) * 100
    return prediction, confidence

# Streamlit app setup
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detector")
st.write("Paste any news content or headline below, and the app will check whether it is **Real or Fake** using your Machine Learning model.")

# Text input
news_input = st.text_area("Paste or Type News Text Here 👇", height=200)

if st.button("Check This News"):
    if news_input.strip() != "":
        with st.spinner("Analyzing..."):
            pred, conf = predict_news(news_input)
            st.write(f"**Confidence Score:** {conf:.2f}%")
            if pred == 0:
                st.success("✅ This news appears to be **Real**.")
            else:
                st.error("❌ This news appears to be **Fake**.")
    else:
        st.warning("⚠️ Please enter some text before checking.")
