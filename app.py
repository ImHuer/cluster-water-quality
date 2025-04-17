import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer


# === Custom Transformer to Add Time Features ===
class TimeFeaturesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, time_column='Timestamp'):
        self.time_column = time_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        df['Hour'] = df[self.time_column].dt.hour
        df['DayOfYear'] = df[self.time_column].dt.dayofyear
        df['Month'] = df[self.time_column].dt.month
        return df.drop(columns=[self.time_column])

# === Custom Transformer to Drop Quality & Non-Feature Columns ===
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')

# === Custom Transformer to Remove Outliers ===
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check if input is a pandas DataFrame or a NumPy array
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            z = np.abs((df - df.mean()) / df.std())
            filtered_df = df[(z < self.threshold).all(axis=1)]
            return filtered_df.reset_index(drop=True)
        else:
            # Handle NumPy array
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            z_scores = np.abs((X - X_mean) / (X_std + 1e-10))  # Add small epsilon to avoid division by zero
            mask = (z_scores < self.threshold).all(axis=1)
            return X[mask]

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
