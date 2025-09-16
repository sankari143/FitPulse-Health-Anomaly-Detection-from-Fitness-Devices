import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from data_processing import preprocess

def detect_anomalies(df):
    """Apply anomaly detection methods."""
    # Rolling mean for trend
    df['hr_roll_mean'] = df['heart_rate'].rolling(30).mean()
    df['residual'] = df['heart_rate'] - df['hr_roll_mean']

    # Threshold-based anomalies
    df['anomaly_threshold'] = ((df['heart_rate'] < 40) | (df['heart_rate'] > 150)).astype(int)

    # Residual-based anomalies
    resid_std = df['residual'].std()
    df['anomaly_residual'] = (df['residual'].abs() > 3 * resid_std).astype(int)

    # Clustering anomalies
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['heart_rate', 'steps']].fillna(0))
    labels = DBSCAN(eps=0.8, min_samples=8).fit_predict(X)
    df['anomaly_cluster'] = (labels == -1).astype(int)

    # Final anomaly flag
    df['anomaly'] = df[['anomaly_threshold', 'anomaly_residual', 'anomaly_cluster']].max(axis=1)
    return df

def run_pipeline(csv_path="sample_heart_rate_with_anomalies.csv", output="processed.csv"):
    """Run full anomaly detection pipeline."""
    df = pd.read_csv(csv_path)
    df = preprocess(df)
    df = detect_anomalies(df)
    df.to_csv(output)
    print(f"Pipeline finished. Output saved at {output}")
    return df

if __name__ == "__main__":
    df = run_pipeline()

    # Plot heart rate with anomalies
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df['heart_rate'], label="Heart Rate", color="blue", alpha=0.7)
    plt.scatter(
        df[df['anomaly'] == 1].index,
        df[df['anomaly'] == 1]['heart_rate'],
        color='red', s=10, label="Anomaly"
    )
    plt.title("Heart Rate with Detected Anomalies")
    plt.xlabel("Time")
    plt.ylabel("Heart Rate (bpm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("anomalies_plot.png")

    # 👉 Show the plot in a window (important for IDLE)
    plt.show()

    # 👉 Print summary
    import os
    total_anomalies = df['anomaly'].sum()
    print("\n✅ Pipeline Finished!")
    print("Total anomalies detected:", total_anomalies)
    print("Processed data saved as:", os.path.abspath("processed.csv"))
    print("Anomaly plot saved as:", os.path.abspath("anomalies_plot.png"))


