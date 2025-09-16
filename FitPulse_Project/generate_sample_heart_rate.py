import pandas as pd
import numpy as np

def generate_sample_heart_rate_csv(path="sample_heart_rate_with_anomalies.csv", days=7, seed=42):
    np.random.seed(seed)
    idx = pd.date_range("2025-09-01", periods=days*24*60, freq="1T")
    n = len(idx)
    hr = 60 + (idx.hour >= 7) * 10 + np.random.normal(0, 3, n)
    for _ in range(8):
        start = np.random.randint(0, n-60)
        hr[start:start+30] += np.linspace(20, 5, 30)
    steps = np.random.poisson(0.1, n)
    for _ in range(10):
        start = np.random.randint(0, n-90)
        steps[start:start+60] += np.random.poisson(30, 60)
    df = pd.DataFrame({"timestamp": idx, "heart_rate": hr, "steps": steps})
    df.to_csv(path, index=False)
    print(f"Sample heart rate CSV saved at {path}")

if __name__ == "__main__":
    generate_sample_heart_rate_csv()
