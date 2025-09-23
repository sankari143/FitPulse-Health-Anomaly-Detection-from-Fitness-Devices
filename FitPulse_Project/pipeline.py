import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
import io

# ---------------------
# Load Data
# ---------------------
def load_csv(filelike):
    df = pd.read_csv(filelike)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def load_json(filelike):
    df = pd.read_json(filelike)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# ---------------------
# Preprocess
# ---------------------
def preprocess(df, freq='1T'):
    df = df.set_index('timestamp').resample(freq).mean().interpolate()
    df = df.reset_index()
    df['hr_roll_mean_5'] = df['heart_rate'].rolling(5, min_periods=1).mean()
    df['hr_roll_std_5'] = df['heart_rate'].rolling(5, min_periods=1).std().fillna(0)
    df['steps_roll_sum_60'] = df['steps'].rolling(60, min_periods=1).sum().fillna(0)
    return df

# ---------------------
# Features (TSFresh)
# ---------------------
def tsfresh_features(df):
    df_tf = df[['timestamp', 'heart_rate']].copy()
    df_tf['id'] = 0
    df_tf = df_tf.rename(columns={'timestamp': 'time', 'heart_rate': 'value'})
    extracted = extract_features(df_tf, column_id='id', column_sort='time', disable_progressbar=True)
    return extracted

# ---------------------
# Prophet Forecasting
# ---------------------
def run_prophet(df, column='heart_rate', periods=60):
    model_df = df[['timestamp', column]].rename(columns={'timestamp': 'ds', column: 'y'})
    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    m.fit(model_df)
    future = m.make_future_dataframe(periods=periods, freq='min')
    forecast = m.predict(future)
    merged = pd.merge(model_df, forecast[['ds','yhat']], on='ds', how='left')
    merged['residual'] = merged['y'] - merged['yhat']
    return m, forecast, merged

# ---------------------
# Clustering
# ---------------------
def cluster_behaviour(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['heart_rate','hr_roll_mean_5']])
    db = DBSCAN(eps=0.7, min_samples=5).fit(X)
    km = KMeans(n_clusters=3, random_state=42).fit(X)
    df['dbscan_label'] = db.labels_
    df['kmeans_label'] = km.labels_
    return df

# ---------------------
# Anomaly Detection
# ---------------------
def detect_anomalies(df, forecast_residuals=None, hr_upper=180, hr_lower=35, residual_threshold=30):
    df['rule_hr_anomaly'] = ((df['heart_rate'] > hr_upper) | (df['heart_rate'] < hr_lower))
    df['cluster_anomaly'] = df['dbscan_label'] == -1
    if forecast_residuals is not None:
        res = forecast_residuals[['ds','residual']].rename(columns={'ds':'timestamp'})
        res['timestamp'] = pd.to_datetime(res['timestamp'])
        df = pd.merge_asof(df.sort_values('timestamp'), res.sort_values('timestamp'),
                           on='timestamp', direction='nearest', tolerance=pd.Timedelta('1min'))
        df['residual_anomaly'] = df['residual'].abs() > residual_threshold
    else:
        df['residual_anomaly'] = False
    df['is_anomaly'] = df[['rule_hr_anomaly','cluster_anomaly','residual_anomaly']].any(axis=1)
    return df

# ---------------------
# Plots
# ---------------------
def plot_time_series(df, y='heart_rate', title='Heart Rate'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df[y], mode='lines', name=y))
    anomalies = df[df['is_anomaly']]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies[y],
                                 mode='markers', name='Anomaly', marker=dict(color='red',size=8)))
    fig.update_layout(title=title)
    return fig

def plot_prophet_forecast(forecast, merged):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='yhat'))
    fig.add_trace(go.Scatter(x=merged['ds'], y=merged['y'], mode='markers', name='Observed'))
    return fig

def plot_clusters(df):
    return px.scatter(df, x='timestamp', y='heart_rate', color='kmeans_label', title='KMeans Clusters')

# ---------------------
# Export
# ---------------------
def export_results_to_csv(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf
