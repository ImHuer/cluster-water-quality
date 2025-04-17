import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# === Load Model and Pipeline ===
pipeline = joblib.load("pipeline_inference.pkl")
model = joblib.load("trained_model.pkl")

st.title("🌊 Water Quality Cluster Predictor")

st.markdown("Enter values below to predict the cluster group for water quality conditions.")

# === User Input Form ===
with st.form("input_form"):
    timestamp = st.text_input("Timestamp (YYYY-MM-DD HH:MM:SS)", value=str(datetime.now()))
    chlorophyll = st.number_input("Chlorophyll", min_value=0.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
    dissolved_oxygen = st.number_input("Dissolved Oxygen", min_value=0.0, step=0.1)
    saturation = st.number_input("DO (% Saturation)", min_value=0.0, step=0.1)
    pH = st.number_input("pH", min_value=0.0, step=0.1)
    salinity = st.number_input("Salinity (ppt)", min_value=0.0, step=0.1)
    conductance = st.number_input("Specific Conductance", min_value=0.0, step=1.0)
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, step=0.1)

    submitted = st.form_submit_button("Predict Cluster")

if submitted:
    user_input = pd.DataFrame([{
        'Record number': 0,  # dummy value
        'Timestamp': timestamp,
        'Chlorophyll': chlorophyll,
        'Temperature': temperature,
        'Dissolved Oxygen': dissolved_oxygen,
        'Dissolved Oxygen (%Saturation)': saturation,
        'pH': pH,
        'Salinity': salinity,
        'Specific Conductance': conductance,
        'Turbidity': turbidity,
        'Chlorophyll [quality]': 'good',
        'Temperature [quality]': 'good',
        'Dissolved Oxygen [quality]': 'good',
        'Dissolved Oxygen (%Saturation) [quality]': 'good',
        'pH [quality]': 'good',
        'Salinity [quality]': 'good',
        'Specific Conductance [quality]': 'good',
        'Turbidity [quality]': 'good'
    }])

    # Preprocess and predict
    X_transformed = pipeline.transform(user_input)
    cluster = model.predict(X_transformed)[0]

    st.success(f"🔍 Predicted Cluster: **Cluster {cluster}**")

    # (Optional) Add cluster interpretation
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_transformed)[0]
        st.subheader("📊 Cluster Probabilities")
        for i, p in enumerate(probs):
            st.write(f"Cluster {i}: {p:.2%}")
