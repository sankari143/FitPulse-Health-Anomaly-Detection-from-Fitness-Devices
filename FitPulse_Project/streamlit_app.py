import streamlit as st
import pandas as pd
from pipeline import run_pipeline

st.title("🏃 FitPulse Health Anomaly Detection")

uploaded_file = st.file_uploader("Upload your fitness data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.to_csv("uploaded.csv", index=False)
    st.write("### Raw Data", df.head())
    st.write("Running pipeline...")
    result = run_pipeline("uploaded.csv", "output.csv")
    st.write("### Processed Data", result.head())
    st.download_button("Download Results (CSV)", open("output.csv","rb"), "output.csv")
    st.line_chart(result['heart_rate'])
    st.write("Red points = detected anomalies")
