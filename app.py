import streamlit as st

# ⛳️ HARUS PALING ATAS SETELAH import streamlit
st.set_page_config(page_title="Student Stress Predictor", layout="centered")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ================================
# 1. LOAD & PREPROCESS DATA
# ================================
@st.cache_data
def load_and_train_model():
    df = pd.read_csv("Student Stress Factors.csv")
    df.drop('Timestamp', axis=1, inplace=True)
    df.columns = [
        'Sleep_Quality', 'Headaches_per_week', 'Academic_Performance',
        'Study_Load', 'Extracurricular_per_week', 'Stress_Level'
    ]

    df['Stress_Label'] = df['Stress_Level'].apply(lambda x: 0 if x <= 2 else 1)
    df.drop('Stress_Level', axis=1, inplace=True)

    # Feature Engineering
    df['Load_Activity_Ratio'] = df['Study_Load'] / (df['Extracurricular_per_week'] + 1)

    X = df.drop('Stress_Label', axis=1)
    y = df['Stress_Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler

model, scaler = load_and_train_model()

# ================================
# 2. STREAMLIT UI
# ================================
st.title("🎓 Student Stress Prediction App")
st.markdown("Isi faktor kehidupanmu sebagai mahasiswa, lalu lihat apakah kamu **berisiko stress** atau tidak.")

# Input
sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 6)
headaches = st.slider("Headaches per Week", 0, 7, 2)
academic_perf = st.slider("Academic Performance (1-10)", 1, 10, 7)
study_load = st.slider("Study Load (hours/day)", 0, 15, 5)
extracurricular = st.slider("Extracurricular Activities per Week", 0, 10, 2)

# Feature Engineering
load_activity_ratio = study_load / (extracurricular + 1)

if st.button("PREDICT"):
    user_input = np.array([
        sleep_quality, headaches, academic_perf,
        study_load, extracurricular, load_activity_ratio
    ]).reshape(1, -1)

    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]
    label = "Stress" if prediction == 1 else "Not Stress"

    st.success(f"🧠 You are predicted to be: **{label}**")
