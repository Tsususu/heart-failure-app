import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load trained model
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Heart Failure Prediction", layout="centered")
st.title("💓 Heart Failure Prediction")
st.markdown("Enter patient data to calculate heart failure probability.")

# Top features from your model
age = st.number_input("Age", min_value=24, max_value=77, value=45)
chol = st.number_input("Cholesterol", min_value=131, max_value=410, value=200)
trestbps = st.number_input("Resting Blood Pressure", min_value=100, max_value=200, value=120)
wbc = st.number_input("White Blood Cell Count", min_value=5000.0, max_value=19590.0, value=7000.0, step=10.0)
restecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
hemoglobin = st.number_input("Hemoglobin", min_value=9.0, max_value=18.0, value=13.5, step=0.1)
bgr = st.number_input("Blood Glucose (BGR)", min_value=60.0, max_value=565.0, value=100.0, step=1.0)
diabetes = st.selectbox("Diabetes", ["Yes", "No"])
bp = st.number_input("Blood Pressure (General BP)", min_value=80, max_value=191, value=120)
thalach = st.number_input("Thalach (maximum heart rate achieved)", min_value=5.0, max_value=198.0, value=5.0, step=1.0)

# Map text inputs to numbers
restecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
diabetes_map = {"No": 0, "Yes": 1}

# Prepare input DataFrame
input_data = {
    'chol': [chol],
    'age': [age],
    'trestbps': [trestbps],
    'wbc': [wbc],
    'restecg': [restecg_map[restecg]],
    'hemoglobin': [hemoglobin],
    'bgr': [bgr],
    'diabetes': [diabetes_map[diabetes]],
    'bp': [bp],
    'thalach': [thalach]
}

input_df = pd.DataFrame(input_data)

# Predict
if st.button("🔍 Predict"):
    prob = model.predict_proba(input_df)[0][1]
    st.success(f"Predicted heart failure probability: **{prob:.3f}**")

