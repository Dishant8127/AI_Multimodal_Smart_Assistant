import streamlit as st
from src.image_module.cnn_predict import predict_image
from src.text_module.transformer_predict import predict_sentiment, summarize_text
from src.ml_module.regression import predict_regression_sample

st.title("AI-Powered Multimodal Smart Assistant")
st.sidebar.title("Modules")
module = st.sidebar.selectbox("Choose module", ["Image AI", "Text AI", "ML Predictions"])

if module == "Image AI":
    st.header("Image Classification (CNN)")
    uploaded = st.file_uploader("Upload an image", type=['png','jpg','jpeg'])
    if uploaded:
        label, score = predict_image(uploaded)
        st.write(f"Prediction: **{label}** (confidence {score:.3f})")
        st.image(uploaded)

if module == "Text AI":
    st.header("Text: Sentiment & Summarization")
    input_text = st.text_area("Enter text for analysis")
    if st.button("Analyze Text"):
        if input_text.strip()=="":
            st.warning("Please enter text")
        else:
            sent = predict_sentiment(input_text)
            summary = summarize_text(input_text)
            st.write("**Sentiment:**", sent)
            st.write("**Summary:**", summary)

if module == "ML Predictions":
    st.header("Simple Regression Demo (Sample)")
    st.write("This module demonstrates loading a scikit-learn model and predicting on sample input.")
    age = st.number_input("Feature: age", min_value=0, max_value=120, value=30)
    salary = st.number_input("Feature: salary", min_value=0, value=50000)
    if st.button("Predict"):
        pred = predict_regression_sample([[age, salary]])
        st.write("Prediction:", pred)
