import json
import pandas as pd

def csv_to_json(csv_path="sample_heart_rate_with_anomalies.csv", json_path="anomaly_events_summary.json"):
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"JSON data saved at {json_path}")

if __name__ == "__main__":
    csv_to_json()
