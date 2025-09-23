import streamlit as st
import plotly.express as px
from fitpulse_utils import (
    load_csv, load_json, preprocess,
    run_prophet, cluster_behaviour, detect_anomalies,
    plot_time_series, plot_prophet_forecast, plot_clusters,
    export_results_to_csv
)
from generate_sample_data import generate_sample_df

# -------------------
# 🎨 Streamlit Theme
# -------------------
st.set_page_config(
    page_title="FitPulse — Health Anomaly Detection",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #F0F8FF;
    }
    h1 {
        color: #D6336C;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("❤️ FitPulse — Health Anomaly Detection Dashboard")

# Sidebar
st.sidebar.header("📂 Data Input")
file = st.sidebar.file_uploader("Upload CSV/JSON (timestamp, heart_rate, steps)", type=['csv','json'])
if st.sidebar.button("Use Sample Data"):
    df_raw = generate_sample_df()
else:
    df_raw = None
    if file is not None:
        if file.name.endswith(".json"):
            df_raw = load_json(file)
        else:
            df_raw = load_csv(file)

if df_raw is None:
    st.info("Upload a file or use sample data to continue.")
    st.stop()

st.subheader("📊 Raw Data Preview")
st.dataframe(df_raw.head())

# Preprocess
df = preprocess(df_raw)
st.subheader("⚙️ Preprocessed Data (1-min Resample)")
st.dataframe(df.head())

# Prophet
model, forecast, merged = run_prophet(df)
st.subheader("📈 Prophet Forecast")
st.plotly_chart(plot_prophet_forecast(forecast, merged), use_container_width=True)

# Clustering
df = cluster_behaviour(df)
st.subheader("🔮 Behaviour Clustering")
st.plotly_chart(plot_clusters(df), use_container_width=True)

# Sidebar Controls
st.sidebar.header("⚠️ Anomaly Detection Settings")
hr_upper = st.sidebar.slider("Heart Rate Upper Limit", 100, 200, 180)
hr_lower = st.sidebar.slider("Heart Rate Lower Limit", 20, 80, 35)
residual_threshold = st.sidebar.slider("Residual Threshold", 5, 50, 25)

# Detect anomalies
df = detect_anomalies(df, merged,
                      hr_upper=int(hr_upper),
                      hr_lower=int(hr_lower),
                      residual_threshold=float(residual_threshold))

# Results
st.subheader("🚨 Anomaly Detection Results")
st.write(f"**Total anomalies detected:** {df['is_anomaly'].sum()}")
st.plotly_chart(plot_time_series(df), use_container_width=True)

# 🔥 Extra Visualization: Bar chart of anomalies by hour
df['hour'] = df['timestamp'].dt.hour
anom_counts = df[df['is_anomaly']].groupby('hour').size().reset_index(name='count')
if not anom_counts.empty:
    st.subheader("📊 Anomalies by Hour of Day")
    st.plotly_chart(px.bar(anom_counts, x='hour', y='count', color='count', title="Anomalies by Hour"), use_container_width=True)

# Export
st.sidebar.header("📤 Export Results")
if st.sidebar.button("Download Results"):
    buf = export_results_to_csv(df)
    st.download_button("Download CSV", buf.getvalue(), "results.csv", "text/csv")

st.success("✅ Analysis Complete — adjust sliders for better results!")
