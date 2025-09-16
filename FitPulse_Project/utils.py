import matplotlib.pyplot as plt

def plot_anomalies(df, save_path="anomalies.png"):
    plt.figure(figsize=(12,4))
    plt.plot(df.index, df['heart_rate'], label="HR", alpha=0.7)
    anomalies = df[df['anomaly']==1]
    plt.scatter(anomalies.index, anomalies['heart_rate'], color="red", s=8, label="Anomaly")
    plt.legend()
    plt.savefig(save_path)
    print(f"Plot saved at {save_path}")
