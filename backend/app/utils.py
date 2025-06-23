import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from app.config import MODEL_DIR

def scale_features(data, stock: str, mode: str):
    scaler_path = os.path.join(MODEL_DIR, f"{stock.replace('.', '_')}_scaler.pkl")
    if mode == "train" or not os.path.exists(scaler_path):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        scaled_data = scaler.transform(data)
    return scaler, scaled_data

def create_sequences(data, threshold=0.02, future_days=3, short_window=30, long_window=120):
    X_short, X_long, y = [], [], []
    for i in range(len(data) - long_window - future_days):
        X_short.append(data[i + long_window - short_window:i + long_window])
        X_long.append(data[i:i + long_window])
        p0 = data[i + long_window - 1][0]
        future_avg = np.mean([data[i + long_window + j][0] for j in range(future_days)])
        if p0 == 0:
            continue
        delta = (future_avg - p0) / p0
        y.append(1 if delta > threshold else 0)
    min_len = min(len(X_short), len(X_long), len(y))
    return np.array(X_short[:min_len]), np.array(X_long[:min_len]), np.array(y[:min_len])
