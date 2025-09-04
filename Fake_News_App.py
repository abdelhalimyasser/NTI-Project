import streamlit as st
import joblib

# Load vectorizer and models
vectorizer = joblib.load("vectorizer.pkl")
models = {
    "Logistic Regression": joblib.load("logistic_model.pkl"),
    "Decision Tree": joblib.load("decision_tree_model.pkl"),
    "Random Forest": joblib.load("random_forest_model.pkl"),
    "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
}

# Streamlit page configuration
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.write("Enter a news article and compare predictions from multiple models.")

# User input
user_input = st.text_area("✍️ Enter news text:")

if st.button("🔍 Predict"):
    if user_input.strip() != "":
        # Transform input text into vector
        X = vectorizer.transform([user_input])

        st.subheader("🔎 Model Results:")
        best_model = None
        best_confidence = 0
        best_prediction = None

        # Loop through each model and show prediction
        for name, model in models.items():
            prediction = model.predict(X)[0]
            
            # Some models support probability (predict_proba)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidence = max(proba)
            else:
                confidence = 1.0  # fallback if not supported

            result = "✅ Real" if prediction == 0 else "❌ Fake"
            st.write(f"**{name}** → {result} (Confidence: {confidence*100:.2f}%)")

            # Track the best model based on confidence
            if confidence > best_confidence:
                best_confidence = confidence
                best_model = name
                best_prediction = result

        # Display the best model
        st.subheader("🏆 Best Model")
        st.success(f"{best_model} → {best_prediction} (Confidence: {best_confidence*100:.2f}%)")
    else:
        st.warning("⚠️ Please enter some text to test.")

