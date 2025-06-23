import numpy as np

def print_diagnostics(label, model_path, scaler, last_short, last_long, prediction_prob):
    print("\n========== 🔍 DIAGNOSTICS REPORT: {} ==========".format(label))
    print(f"📂 Model path: {model_path}")
    print("⚖️ Scaler data_min_[:3]:", np.round(scaler.data_min_[:3], 4).tolist())
    print("⚖️ Scaler data_max_[:3]:", np.round(scaler.data_max_[:3], 4).tolist())
    print("⚖️ Scaler data_range_[:3]:", np.round(scaler.data_range_[:3], 4).tolist())
    print("📊 last_short[:5]:", np.round(last_short.flatten()[:5], 4).tolist())
    print("📊 last_long[:5]:", np.round(last_long.flatten()[:5], 4).tolist())
    print(f"🔮 Prediction probability: {round(float(prediction_prob), 4)}")
    print("================================================\n")
