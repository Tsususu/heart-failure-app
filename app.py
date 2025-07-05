import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Heart Failure Prediction", layout="centered")
st.title("💓 Heart Failure Prediction")
st.markdown("Enter patient data to calculate heart failure probability.")

# Top features from your importance chart
age = st.number_input("Age", 1, 120, 45)
chol = st.number_input("Cholesterol", 100, 400, 200)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
wbc = st.number_input("White Blood Cell Count", 2.0, 20.0, 6.0)
restecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
hemoglobin = st.number_input("Hemoglobin", 5.0, 20.0, 13.5)
bgr = st.number_input("Blood Glucose (BGR)", 50.0, 300.0, 100.0)
diabetes = st.selectbox("Diabetes", ["Yes", "No"])
bp = st.number_input("Blood Pressure (General BP)", 80, 200, 120)
co = st.number_input("Cardiac Output", 2.0, 10.0, 5.0)

# Map text inputs to numbers
restecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
diabetes_map = {"No": 0, "Yes": 1}

# Arrange input for prediction
input_data = np.array([[
    chol, age, trestbps, wbc,
    restecg_map[restecg], hemoglobin, bgr,
    diabetes_map[diabetes], bp, co
]])

if st.button("🔍 Predict"):
    prob = model.predict_proba(input_data)[0][1]
    st.success(f"Predicted heart failure probability: **{prob:.3f}**")
