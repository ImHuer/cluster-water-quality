import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objs as go
from sklearn.base import BaseEstimator, TransformerMixin

from fpdf import FPDF #pip install fpdf
import base64
import plotly.io as pio

image_paths = []

if vis_option == "1D (PC1 Distribution)":
    pio.write_image(fig_1d, "chart_1d.png", format='png', width=800, height=500)
    image_paths.append("chart_1d.png")

elif vis_option == "2D (PC1 vs PC2)":
    pio.write_image(fig_2d, "chart_2d.png", format='png', width=800, height=500)
    image_paths.append("chart_2d.png")

elif vis_option == "3D (PC1 vs PC2 vs PC3)":
    pio.write_image(fig_3d, "chart_3d.png", format='png', width=800, height=500)
    image_paths.append("chart_3d.png")

elif vis_option == "Show All Visualizations":
    pio.write_image(fig_1d, "chart_1d.png", format='png', width=800, height=500)
    pio.write_image(fig_2d, "chart_2d.png", format='png', width=800, height=500)
    pio.write_image(fig_3d, "chart_3d.png", format='png', width=800, height=500)
    image_paths.extend(["chart_1d.png", "chart_2d.png", "chart_3d.png"])

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

# === Page Config ===
st.set_page_config(
    page_title="Water Quality Cluster Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === Animated Welcome Title and Subtitle ===
st.markdown("""
    <style>
    .title-container {
        text-align: center;
        padding-top: 30px;
        padding-bottom: 20px;
        animation: fadein 1.5s;
    }
    @keyframes fadein {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .title-container h1 {
        font-size: 60px;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

if 'form_submitted' not in st.session_state or not st.session_state['form_submitted']:
    st.markdown("""
        <div class="title-container">
            <h1>🌊 Water Quality Cluster Predictor</h1>
            <div class="subtitle">Please enter water parameters below to get started.</div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("# 🌊 Water Quality Cluster Predictor")

# === Custom Transformers ===
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

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            z = np.abs((df - df.mean()) / df.std())
            filtered_df = df[(z < self.threshold).all(axis=1)]
            return filtered_df.reset_index(drop=True)
        else:
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            z_scores = np.abs((X - X_mean) / (X_std + 1e-10))
            mask = (z_scores < self.threshold).all(axis=1)
            return X[mask]

# === Load Model and Pipeline ===
pipeline = joblib.load("pipeline_inference.pkl")
model = joblib.load("trained_model.pkl")

# === Form Input ===
with st.form("input_form"):
    col1, col2 = st.columns([2, 2])  # Two equal columns

    with col1:
        avg_water_speed = st.number_input("🌊 Average Water Speed (m/s)", min_value=0.0, format="%.3f")
        avg_water_direction = st.number_input("🧭 Average Water Direction (°)", min_value=0.0, max_value=360.0, format="%.3f")
        chlorophyll = st.number_input("🟢 Chlorophyll (µg/L)", min_value=0.0, format="%.3f")
        temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, format="%.3f")
        dissolved_oxygen = st.number_input("💨 Dissolved Oxygen (mg/L)", min_value=0.0, format="%.3f")

    with col2:
        saturation = st.number_input("💧 DO (% Saturation)", min_value=0.0, format="%.3f")
        pH = st.number_input("⚗️ pH Level", min_value=0.0, format="%.3f")
        salinity = st.number_input("🌊 Salinity (ppt)", min_value=0.0, format="%.3f")
        conductance = st.number_input("⚡ Specific Conductance (µS/cm)", min_value=0.0, format="%.3f")
        turbidity = st.number_input("🌫️ Turbidity (NTU)", min_value=0.0, format="%.3f")

    col3, col4, col5 = st.columns([1, 1, 1])
    with col3:
        month = st.selectbox("📅 Month (1-12)", list(range(1, 13)))
    with col4:
        day_of_year = st.selectbox("📆 Day of Year (1-366)", list(range(1, 367)))
    with col5:
        hour = st.selectbox("⏰ Hour (0-23)", list(range(0, 24)))

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

# === After Form Submitted ===
if st.session_state.get('form_submitted', False):
    user_input = pd.DataFrame([st.session_state['user_input']])
    X_transformed = pipeline.transform(user_input)
    cluster = model.predict(X_transformed)[0]

    st.success(f"🔍 Predicted Cluster: Cluster {cluster}")

    # Interpret cluster meaning
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

    # Show on screen
    st.markdown("### 🧪 Cluster Characteristics")
    for line in interpretation:
        st.markdown(f" {line}")

    pdf_bytes = generate_pdf(st.session_state['user_input'], cluster, interpretation, image_paths=image_paths)
    b64 = base64.b64encode(pdf_bytes).decode()

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name="cluster_prediction_report.pdf",
        mime="application/pdf"
    )

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_transformed)[0]
        st.subheader("📊 Cluster Probabilities")
        for i, p in enumerate(probs):
            st.write(f"Cluster {i}: {p:.2%}")

    # === Load PCA cluster data ===
    try:
        pca_df = pd.read_csv('pca_with_clusters.csv')
        if pca_df.empty:
            st.error("⚠ Loaded PCA dataset is empty. Cannot continue.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error loading PCA data: {e}")
        st.stop()

    cluster0 = pca_df[pca_df['cluster'] == 0]
    cluster1 = pca_df[pca_df['cluster'] == 1]
    cluster2 = pca_df[pca_df['cluster'] == 2]

    st.subheader("🖼 Choose Visualization Type")
    vis_option = st.radio(
        "Select a visualization:",
        ("1D (PC1 Distribution)", "2D (PC1 vs PC2)", "3D (PC1 vs PC2 vs PC3)", "Show All Visualizations")
    )

    # === Visualization ===

    # --- 1D Visualization ---
    if vis_option == "1D (PC1 Distribution)" or vis_option == "Show All Visualizations":
        st.subheader("📈 1D Visualization: PC1 Distribution")
        fig_1d = go.Figure()

        fig_1d.add_trace(go.Scatter(
            x=cluster0.index,
            y=cluster0['PC1_3d'],
            mode='markers',
            name='Cluster 0',
            marker=dict(color='#66c2a5', size=5),
            hovertemplate="Index: %{x}<br>PC1: %{y:.3f}",
            hoverlabel=dict(bgcolor='#66c2a5', font=dict(color='white'))
        ))
        fig_1d.add_trace(go.Scatter(
            x=cluster1.index,
            y=cluster1['PC1_3d'],
            mode='markers',
            name='Cluster 1',
            marker=dict(color='#fc8d62', size=5),
            hovertemplate="Index: %{x}<br>PC1: %{y:.3f}",
            hoverlabel=dict(bgcolor='#fc8d62', font=dict(color='white'))
        ))
        fig_1d.add_trace(go.Scatter(
            x=cluster2.index,
            y=cluster2['PC1_3d'],
            mode='markers',
            name='Cluster 2',
            marker=dict(color='#8da0cb', size=5),
            hovertemplate="Index: %{x}<br>PC1: %{y:.3f}",
            hoverlabel=dict(bgcolor='#8da0cb', font=dict(color='white'))
        ))
        fig_1d.add_trace(go.Scatter(
            x=[-1],
            y=[X_transformed[0, 0]],
            mode='markers+text',
            name='Your Input',
            marker=dict(size=12, color="red", symbol="diamond"),
            text=["Your Input"],
            hovertemplate="Index: %{x}<br>PC1: %{y:.3f}",
            hoverlabel=dict(bgcolor='red', font=dict(color='white'))
        ))

        fig_1d.update_layout(
            title="PC1 Distribution Across Samples",
            xaxis_title="Sample Index",
            yaxis_title="PC1 Value",
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig_1d, use_container_width=True)

    # --- 2D Visualization ---
    if vis_option == "2D (PC1 vs PC2)" or vis_option == "Show All Visualizations":
        st.subheader("📈 2D Visualization: PC1 vs PC2")
        fig_2d = go.Figure()

        fig_2d.add_trace(go.Scatter(
            x=cluster0['PC1_3d'],
            y=cluster0['PC2_3d'],
            mode='markers',
            name='Cluster 0',
            marker=dict(color='#66c2a5', size=5),
            hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}",
            hoverlabel=dict(bgcolor='#66c2a5', font=dict(color='white'))
        ))
        fig_2d.add_trace(go.Scatter(
            x=cluster1['PC1_3d'],
            y=cluster1['PC2_3d'],
            mode='markers',
            name='Cluster 1',
            marker=dict(color='#fc8d62', size=5),
            hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}",
            hoverlabel=dict(bgcolor='#fc8d62', font=dict(color='white'))
        ))
        fig_2d.add_trace(go.Scatter(
            x=cluster2['PC1_3d'],
            y=cluster2['PC2_3d'],
            mode='markers',
            name='Cluster 2',
            marker=dict(color='#8da0cb', size=5),
            hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}",
            hoverlabel=dict(bgcolor='#8da0cb', font=dict(color='white'))
        ))
        fig_2d.add_trace(go.Scatter(
            x=[X_transformed[0, 0]],
            y=[X_transformed[0, 1]],
            mode='markers+text',
            name='Your Input',
            marker=dict(size=12, color="red", symbol="diamond"),
            text=["Your Input"],
            hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}",
            hoverlabel=dict(bgcolor='red', font=dict(color='white'))
        ))

        fig_2d.update_layout(
            title="PC1 vs PC2 Scatter Plot",
            xaxis_title="PC1",
            yaxis_title="PC2",
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig_2d, use_container_width=True)

    # --- 3D Visualization ---
    if vis_option == "3D (PC1 vs PC2 vs PC3)" or vis_option == "Show All Visualizations":
        st.subheader("📈 3D Visualization: PC1 vs PC2 vs PC3")
        fig_3d = go.Figure()

        fig_3d.add_trace(go.Scatter3d(
            x=cluster0['PC1_3d'],
            y=cluster0['PC2_3d'],
            z=cluster0['PC3_3d'],
            mode='markers',
            name='Cluster 0',
            marker=dict(color='#66c2a5', size=5),
            hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}",
            hoverlabel=dict(bgcolor='#66c2a5', font=dict(color='white'))
        ))
        fig_3d.add_trace(go.Scatter3d(
            x=cluster1['PC1_3d'],
            y=cluster1['PC2_3d'],
            z=cluster1['PC3_3d'],
            mode='markers',
            name='Cluster 1',
            marker=dict(color='#fc8d62', size=5),
            hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}",
            hoverlabel=dict(bgcolor='#fc8d62', font=dict(color='white'))
        ))
        fig_3d.add_trace(go.Scatter3d(
            x=cluster2['PC1_3d'],
            y=cluster2['PC2_3d'],
            z=cluster2['PC3_3d'],
            mode='markers',
            name='Cluster 2',
            marker=dict(color='#8da0cb', size=5),
            hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}",
            hoverlabel=dict(bgcolor='#8da0cb', font=dict(color='white'))
        ))
        fig_3d.add_trace(go.Scatter3d(
            x=[X_transformed[0, 0]],
            y=[X_transformed[0, 1]],
            z=[X_transformed[0, 2]],
            mode='markers+text',
            name='Your Input',
            marker=dict(size=10, color='red', symbol='diamond'),
            text=["Your Input"],
            hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}",
            hoverlabel=dict(bgcolor='red', font=dict(color='white'))
        ))

        fig_3d.update_layout(
            title="PC1 vs PC2 vs PC3 Scatter Plot",
            scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
            margin=dict(l=0, r=0, b=0, t=40),
            scene_camera_eye=dict(x=1.2, y=1.2, z=1.2)
        )
        st.plotly_chart(fig_3d, use_container_width=True)
