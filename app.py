import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objs as go
from sklearn.base import BaseEstimator, TransformerMixin
from fpdf import FPDF
import base64

# === Page Config ===
st.set_page_config(page_title="Water Quality Cluster Predictor", page_icon="🌊", layout="wide")

# === Custom Styling ===
st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .stNumberInput > div > input {
        height: 45px;
        font-size: 18px;
    }
    .stSelectbox > div > div {
        height: 45px;
        font-size: 18px;
    }
    .stForm button {
        font-size: 18px !important;
        height: 50px !important;
        background-color: #1f77b4;
        color: white;
    }
    .centered-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# === PDF Generator ===
def generate_pdf(user_input, cluster_label, interpretation):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Water Quality Cluster Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(100, 10, "User Input Parameters:", ln=True)
    pdf.set_font("Arial", size=11)

    for key, value in user_input.items():
        pdf.cell(200, 8, txt=f"{key}: {value}", ln=True)

    pdf.ln(8)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(100, 10, f"Predicted Cluster: Cluster {cluster_label}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    for line in interpretation:
        pdf.multi_cell(0, 8, line)

    return pdf.output(dest='S').encode('latin-1')

# === Load Model and Pipeline ===
pipeline = joblib.load("pipeline_inference.pkl")
model = joblib.load("trained_model.pkl")

# === App Header ===
st.markdown("<div class='centered-title'>🌊 Water Quality Cluster Predictor</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Use the form below to input water condition parameters and get your predicted cluster with visual insights.</p>", unsafe_allow_html=True)

# === Form Input Section ===
st.markdown("### 📥 <span style='color:#1f77b4'>Input Water Quality Parameters</span>", unsafe_allow_html=True)

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        avg_water_speed = st.number_input("Average Water Speed (m/s)", min_value=0.0, step=0.001, format="%.3f")
        avg_water_direction = st.number_input("Average Water Direction (°)", min_value=0.0, max_value=360.0, step=0.1, format="%.3f")
        chlorophyll = st.number_input("Chlorophyll", min_value=0.0, step=0.1, format="%.3f")
        temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1, format="%.3f")
        dissolved_oxygen = st.number_input("Dissolved Oxygen", min_value=0.0, step=0.1, format="%.3f")

    with col2:
        saturation = st.number_input("DO (% Saturation)", min_value=0.0, step=0.1, format="%.3f")
        pH = st.number_input("pH", min_value=0.0, step=0.1, format="%.3f")
        salinity = st.number_input("Salinity (ppt)", min_value=0.0, step=0.1, format="%.3f")
        conductance = st.number_input("Specific Conductance", min_value=0.0, step=1.0, format="%.3f")
        turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, step=0.1, format="%.3f")

    st.markdown("### 🕒 <span style='color:#1f77b4'>Input Time Information</span>", unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    with col3:
        month = st.selectbox("Month", list(range(1, 13)))
    with col4:
        day_of_year = st.selectbox("Day of Year", list(range(1, 367)))
    with col5:
        hour = st.selectbox("Hour", list(range(0, 24)))

    submitted = st.form_submit_button("🚀 Predict Cluster")

    if submitted:
        try:
            timestamp = datetime(2024, 1, 1) + pd.to_timedelta(day_of_year - 1, unit='d')
            timestamp = timestamp.replace(month=month, hour=hour)
            timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            st.error(f"Error creating timestamp: {e}")
            st.stop()
            
        st.session_state['form_submitted'] = True
        st.session_state['user_input'] = {
            'Record number': 0,
            'Timestamp': timestamp,
            'Average Water Speed': avg_water_speed,
            'Average Water Direction': avg_water_direction,
            'Chlorophyll': chlorophyll,
            'Temperature': temperature,
            'Dissolved Oxygen': dissolved_oxygen,
            'Dissolved Oxygen (%Saturation)': saturation,
            'pH': pH,
            'Salinity': salinity,
            'Specific Conductance': conductance,
            'Turbidity': turbidity
        }

# === After Submission ===
if st.session_state.get('form_submitted', False):
    user_input = pd.DataFrame([st.session_state['user_input']])
    X_transformed = pipeline.transform(user_input)
    cluster = model.predict(X_transformed)[0]

    st.markdown("### 🎯 <span style='color:#2ca02c'>Prediction Result</span>", unsafe_allow_html=True)
    st.success(f"✅ Predicted Cluster: **Cluster {cluster}**")

    if cluster == 0:
        interpretation = [
            "Cluster 0: Disturbed Water Conditions",
            "- High turbidity and lower salinity",
            "- Often caused by storm runoff or estuarine disturbance",
            "- Typically observed during wet-season periods"
        ]
    elif cluster == 1:
        interpretation = [
            "Cluster 1: Stable Marine-like Conditions",
            "- Clearer water and higher salinity",
            "- Suggests a dry-season or stable flow period",
            "- Indicates less anthropogenic disturbance"
        ]
    else:
        interpretation = ["Cluster info unavailable."]

    st.markdown("### 📄 Downloadable PDF Report")
    pdf_bytes = generate_pdf(st.session_state['user_input'], cluster, interpretation)
    b64 = base64.b64encode(pdf_bytes).decode()

    st.download_button(
        label="📥 Download Cluster Report (PDF)",
        data=pdf_bytes,
        file_name="cluster_prediction_report.pdf",
        mime="application/pdf"
    )

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_transformed)[0]
        st.markdown("### 📊 Cluster Probabilities")
        for i, p in enumerate(probs):
            st.write(f"Cluster {i}: {p:.2%}")

    try:
        pca_df = pd.read_csv('pca_with_clusters.csv')
    except Exception as e:
        st.error(f"❌ Could not load PCA data: {e}")
        st.stop()

    cluster_colors = ['#66c2a5', '#fc8d62', '#8da0cb']
    st.markdown("### 🖼 PCA Visualizations")
    fig_2d = go.Figure()
    for i in range(3):
        cluster_data = pca_df[pca_df['cluster'] == i]
        fig_2d.add_trace(go.Scatter(
            x=cluster_data['PC1_3d'],
            y=cluster_data['PC2_3d'],
            mode='markers',
            name=f'Cluster {i}',
            marker=dict(color=cluster_colors[i], size=5)
        ))

    fig_2d.add_trace(go.Scatter(
        x=[X_transformed[0, 0]],
        y=[X_transformed[0, 1]],
        mode='markers+text',
        name='Your Input',
        marker=dict(size=12, color="red", symbol="diamond"),
        text=["Your Input"]
    ))
    fig_2d.update_layout(title="PC1 vs PC2 Scatter Plot", xaxis_title="PC1", yaxis_title="PC2")
    st.plotly_chart(fig_2d, use_container_width=True)
