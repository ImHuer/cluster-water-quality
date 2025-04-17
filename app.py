import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objs as go
from sklearn.base import BaseEstimator, TransformerMixin

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

# === Load model and pipeline ===
pipeline = joblib.load("pipeline_inference.pkl")
model = joblib.load("trained_model.pkl")

# === Streamlit App ===
st.title("🌊 Water Quality Cluster Predictor")

st.markdown("Enter values below to predict the cluster group for water quality conditions.")

with st.form("input_form"):
    timestamp = st.text_input("Timestamp (YYYY-MM-DD HH:MM:SS)", value=str(datetime.now()))
    avg_water_speed = st.number_input("Average Water Speed (m/s)", min_value=0.0, step=0.01)
    avg_water_direction = st.number_input("Average Water Direction (degrees)", min_value=0.0, max_value=360.0, step=0.1)
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
    }])

    X_transformed = pipeline.transform(user_input)
    cluster = model.predict(X_transformed)[0]

    st.success(f"🔍 Predicted Cluster: **Cluster {cluster}**")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_transformed)[0]
        st.subheader("📊 Cluster Probabilities")
        for i, p in enumerate(probs):
            st.write(f"Cluster {i}: {p:.2%}")

    # === Visualization selection (only appear after submit) ===
    st.subheader("🖼️ Choose Visualization Type")

    vis_option = st.radio(
        "Select a visualization:",
        ("1D (PC1 Distribution)", "2D (PC1 vs PC2)", "3D (PC1 vs PC2 vs PC3)", "Show All Visualizations")
    )

    # === Load PCA cluster data
    try:
        pca_df = pd.read_csv('pca_with_clusters.csv')
    except Exception as e:
        st.error(f"❌ Error loading PCA data: {e}")
        st.stop()

    if not pca_df.empty:
        cluster0 = pca_df[pca_df['cluster'] == 0]
        cluster1 = pca_df[pca_df['cluster'] == 1]
        cluster2 = pca_df[pca_df['cluster'] == 2]

    if vis_option == "1D (PC1 Distribution)" or vis_option == "Show All Visualizations":
        st.subheader("📈 1D Visualization: PC1 Distribution")
    
        fig_1d = px.scatter(
            pca_df,
            x=pca_df.index,
            y="PC1_3d",
            color=pca_df['cluster'].astype(str),
            labels={"color": "Cluster"},
            title="PC1 Distribution Across Samples",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
    
        fig_1d.add_scatter(
            x=[-1],
            y=[X_transformed[0, 0]],
            mode="markers+text",
            marker=dict(size=10, color="red", symbol="diamond"),
            name="Your Input",
            text=["Your Input"],
            hoverinfo="text"
        )
    
        st.plotly_chart(fig_1d, use_container_width=True)
    
    if vis_option == "2D (PC1 vs PC2)" or vis_option == "Show All Visualizations":
        st.subheader("📈 2D Visualization: PC1 vs PC2")
    
        fig_2d = px.scatter(
            pca_df,
            x="PC1_3d",
            y="PC2_3d",
            color=pca_df['cluster'].astype(str),
            labels={"color": "Cluster"},
            title="PC1 vs PC2 Scatter Plot",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
    
        fig_2d.add_scatter(
            x=[X_transformed[0, 0]],
            y=[X_transformed[0, 1]],
            mode="markers+text",
            marker=dict(size=12, color="red", symbol="diamond"),
            name="Your Input",
            text=["Your Input"],
            hoverinfo="text"
        )
    
        st.plotly_chart(fig_2d, use_container_width=True)
    
    if vis_option == "3D (PC1 vs PC2 vs PC3)" or vis_option == "Show All Visualizations":
        st.subheader("📈 3D Visualization: PC1 vs PC2 vs PC3")
    
        trace0 = go.Scatter3d(
            x=cluster0['PC1_3d'], y=cluster0['PC2_3d'], z=cluster0['PC3_3d'],
            mode='markers', name='Cluster 0',
            marker=dict(size=5, color='rgba(255, 128, 255, 0.8)'),
            text=cluster0['hover_info'],
            hoverinfo='text'
        )
    
        trace1 = go.Scatter3d(
            x=cluster1['PC1_3d'], y=cluster1['PC2_3d'], z=cluster1['PC3_3d'],
            mode='markers', name='Cluster 1',
            marker=dict(size=5, color='rgba(255, 128, 2, 0.8)'),
            text=cluster1['hover_info'],
            hoverinfo='text'
        )
    
        trace2 = go.Scatter3d(
            x=cluster2['PC1_3d'], y=cluster2['PC2_3d'], z=cluster2['PC3_3d'],
            mode='markers', name='Cluster 2',
            marker=dict(size=5, color='rgba(0, 255, 200, 0.8)'),
            text=cluster2['hover_info'],
            hoverinfo='text'
        )
    
        user_point = go.Scatter3d(
            x=[X_transformed[0, 0]],
            y=[X_transformed[0, 1]],
            z=[X_transformed[0, 2]],
            mode='markers+text',
            name='Your Input',
            marker=dict(size=10, color='red', symbol='diamond'),
            text=["Your Input"],
            hoverinfo='text'
        )
    
        data = [trace0, trace1, trace2, user_point]
    
        layout = go.Layout(
            title="Visualizing Clusters in 3D (Including Your Input!)",
            scene=dict(
                xaxis=dict(title='PC1'),
                yaxis=dict(title='PC2'),
                zaxis=dict(title='PC3')
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
    
        fig_3d = go.Figure(data=data, layout=layout)
        fig_3d.update_layout(scene_camera_eye=dict(x=1.2, y=1.2, z=1.2))
    
        st.plotly_chart(fig_3d, use_container_width=True)
