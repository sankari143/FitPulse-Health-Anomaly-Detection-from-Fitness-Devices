import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_df(minutes=1440, seed=42):
    np.random.seed(seed)
    start = datetime.now().replace(second=0, microsecond=0) - timedelta(minutes=minutes)
    timestamps = [start + timedelta(minutes=i) for i in range(minutes)]
    hr = []
    for i in range(minutes):
        base = 65 + np.sin(2*np.pi*(i/1440))*10 + np.random.normal(0,3)
        if 400 < i < 500:  # morning exercise
            base += 40
        if np.random.rand() < 0.002:  # spike anomaly
            base += 80
        hr.append(max(30, base))
    steps = np.random.poisson(0.5, size=minutes)
    return pd.DataFrame({"timestamp":timestamps,"heart_rate":np.round(hr,1),"steps":steps})

if __name__ == "__main__":
    df = generate_sample_df()
    df.to_csv("sample_data.csv", index=False)
    df.to_json("sample_data.json", orient="records", date_format="iso")
    print("Sample CSV and JSON generated.")
