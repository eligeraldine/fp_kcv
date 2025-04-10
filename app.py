import streamlit as st
import numpy as np
import joblib

# Konfigurasi halaman
st.set_page_config(page_title="Student Stress Predictor", layout="centered")

# ================================
# 1. LOAD MODEL & SCALER
# ================================
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("best_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_model_and_scaler()

# ================================
# 2. STREAMLIT UI
# ================================
st.title("🎓 Student Stress Prediction App")
st.markdown("Tell us about your daily habits and we'll predict your stress level!")

# Input dari user
sleep_quality = st.slider("Sleep Quality (1-10)", 1, 5, 1)
headaches = st.slider("Headaches per Week", 1, 5, 1)
academic_perf = st.slider("Academic Performance (1-10)", 1, 5, 1)
study_load = st.slider("Study Load (hours/day)", 1, 5, 1)
extracurricular = st.slider("Extracurricular Activities per Week", 1, 5, 1)

# Feature Engineering
load_activity_ratio = study_load / (extracurricular + 1)

# Predict
if st.button("🧠 Predict Stress Level"):
    user_input = np.array([
        sleep_quality, headaches, academic_perf,
        study_load, extracurricular, load_activity_ratio
    ]).reshape(1, -1)

    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]

    label = "Stress" if prediction == 1 else "Not Stress"
    st.success(f"✨ You are predicted to be: **{label}**")
