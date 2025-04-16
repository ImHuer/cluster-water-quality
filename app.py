import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go

# Load pipeline
pipeline = joblib.load('pipeline_inference.pkl')
df_clusters = joblib.load('gmm_df.pkl')  # for 3D PCA visualization

st.title("Water Quality GMM Clustering")

# User Input
st.subheader("Enter Water Quality Parameters: ")

water_speed = st.number_input("Average Water Speed")
water_dir = st.number_input("Average Water Direction")
chlorophyll = st.number_input("Chlorophyll")
temp = st.number_input("Temperature")
do = st.number_input("Dissolved Oxygen")
do_sat = st.number_input("Dissolved Oxygen (%Saturation)")
pH = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
salinity = st.number_input("Salinity")
conductance = st.number_input("Specific Conductance")
turbidity = st.number_input("Turbidity")
hour = st.number_input("Hour", min_value=0, max_value=23)
doy = st.number_input("Day of Year", min_value=1, max_value=366)
month = st.number_input("Month", min_value=1, max_value=12)

input_data = np.array([[
    water_speed, water_dir, chlorophyll, temp, do,
    do_sat, pH, salinity, conductance, turbidity,
    hour, doy, month
]])


if st.button("Predict and Visualize"):
    # pipeline already includes scaler + PCA + GMM
    cluster = pipeline.predict(input_data)[0]
    input_pca = pipeline.transform(input_data)  # for 3D plot

    st.success(f"Predicted Cluster: {cluster}")

    # Prepare traces per cluster
    traces = []
    colors = ['rgba(255, 128, 255, 0.8)', 'rgba(255, 128, 2, 0.8)', 'rgba(0, 255, 200, 0.8)',
              'rgba(100, 100, 255, 0.8)', 'rgba(200, 255, 100, 0.8)']  # extend if more clusters

    for i in sorted(df_clusters['Cluster'].unique()):
        cluster_data = df_clusters[df_clusters['Cluster'] == i]
        trace = go.Scatter3d(
            x=cluster_data["PC1_3d"],
            y=cluster_data["PC2_3d"],
            z=cluster_data["PC3_3d"],
            mode="markers",
            name=f"Cluster {i}",
            marker=dict(color=colors[i], size=4, opacity=0.7),
            text=None
        )
        traces.append(trace)

    # Add user input point
    user_trace = go.Scatter3d(
        x=[input_pca[0][0]],
        y=[input_pca[0][1]],
        z=[input_pca[0][2]],
        mode="markers+text",
        name="Your Input",
        marker=dict(color='black', size=8, symbol='x'),
        text=[f"Input (Cluster {cluster})"],
        textposition="top center"
    )
    traces.append(user_trace)

    # Layout
    layout = dict(
        title="Visualizing Clusters in Three Dimensions Using PCA",
        scene=dict(
            xaxis=dict(title='PC1'),
            yaxis=dict(title='PC2'),
            zaxis=dict(title='PC3')
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig, use_container_width=True)
