import joblib
import shap
import os

# Suppress TensorFlow logging spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

# ==========================================
# LOAD ALL MODELS AT STARTUP
# ==========================================
print("Loading ML & DL models...")

try:
    # 1. XGBoost Classification Models
    classifier_model = joblib.load("../ml_model/ensemble_model.pkl")
    classifier_scaler = joblib.load("../ml_model/scaler_ensemble.pkl")

    # 2. XGBoost RUL Models
    rul_model = joblib.load("../ml_model/rul_model.pkl")
    rul_scaler = joblib.load("../ml_model/scaler_rul.pkl")

    # 3. SHAP Explainer
    print("Initializing SHAP Explainer...")
    rul_explainer = shap.TreeExplainer(rul_model)

    # 4. LSTM Deep Learning Models
    print("Loading LSTM Time-Series Models...")
    lstm_model = load_model("../dl_model/lstm_model.keras")
    lstm_scaler = joblib.load("../dl_model/scaler_lstm.pkl")
    lstm_encoder = joblib.load("../dl_model/label_encoder.pkl")

    print("All models loaded successfully!")

except Exception as e:
    print(f"Startup Error: Ensure models are in the correct folder. Error: {e}")
    raise
