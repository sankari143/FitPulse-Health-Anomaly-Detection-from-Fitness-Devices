import pandas as pd
import numpy as np

def preprocess(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.resample('1T').mean()
    df['heart_rate'] = df['heart_rate'].interpolate(limit=5)
    df['steps'] = df['steps'].fillna(0).astype(int)
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    return df
