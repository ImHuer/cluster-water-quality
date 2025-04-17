import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objs as go
from sklearn.base import BaseEstimator, TransformerMixin

# === Page Config ===
st.set_page_config(
    page_title="Water Quality Cluster Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

# === Streamlit App ===
st.title("🌊 Water Quality Cluster Predictor")
st.markdown("Enter values below to predict the cluster group for water quality conditions.")

# === Form Input ===
with st.form("input_form"):
    
    avg_water_speed = st.number_input("Average Water Speed (m/s)", min_value=0.0, step=0.001, format="%.3f")
    avg_water_direction = st.number_input("Average Water Direction (degrees)", min_value=0.0, max_value=360.0, step=0.1, format="%.3f")
    chlorophyll = st.number_input("Chlorophyll", min_value=0.0, step=0.1, format="%.3f")
    temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1, format="%.3f")
    dissolved_oxygen = st.number_input("Dissolved Oxygen", min_value=0.0, step=0.1, format="%.3f")
    saturation = st.number_input("DO (% Saturation)", min_value=0.0, step=0.1, format="%.3f")
    pH = st.number_input("pH", min_value=0.0, step=0.1, format="%.3f")
    salinity = st.number_input("Salinity (ppt)", min_value=0.0, step=0.1, format="%.3f")
    conductance = st.number_input("Specific Conductance", min_value=0.0, step=1.0, format="%.3f")
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, step=0.1, format="%.3f")
    month = st.selectbox("Month", list(range(1, 13)))
    day_of_year = st.selectbox("Day of Year", list(range(1, 367)))
    hour = st.selectbox("Hour", list(range(0, 24)))

    submitted = st.form_submit_button("Predict Cluster")

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

    st.success(f"🔍 Predicted Cluster: **Cluster {cluster}**")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_transformed)[0]
        st.subheader("📊 Cluster Probabilities")
        for i, p in enumerate(probs):
            st.write(f"Cluster {i}: {p:.2%}")

    # === Load PCA cluster data ===
    try:
        pca_df = pd.read_csv('pca_with_clusters.csv')
        if pca_df.empty:
            st.error("⚠️ Loaded PCA dataset is empty. Cannot continue.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error loading PCA data: {e}")
        st.stop()

    cluster0 = pca_df[pca_df['cluster'] == 0]
    cluster1 = pca_df[pca_df['cluster'] == 1]
    cluster2 = pca_df[pca_df['cluster'] == 2]

    st.subheader("🖼️ Choose Visualization Type")
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
