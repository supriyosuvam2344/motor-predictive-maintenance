from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import deque
import pandas as pd
import numpy as np
import joblib
import shap
import os

# Suppress TensorFlow logging spam in the terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

app = FastAPI(title="Motor Predictive Maintenance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# SERVER-SIDE MEMORY BUFFER
# ==========================================
SEQUENCE_LENGTH = 30
# A deque automatically deletes the oldest reading when a new one arrives
telemetry_buffer = deque(maxlen=SEQUENCE_LENGTH)

# ==========================================
# LOAD ALL MODELS
# ==========================================
try:
    print("Loading models_m...")

    # 1. XGBoost Models (Note: We kept the old .pkl names during training!)
    classifier_model = joblib.load("../models_m/ensemble_model.pkl")
    classifier_scaler = joblib.load("../models_m/scaler_ensemble.pkl")

    rul_model = joblib.load("../models_m/rul_model.pkl")
    rul_scaler = joblib.load("../models_m/scaler_rul.pkl")

    # 2. SHAP Explainer
    print("Initializing SHAP Explainer...")
    rul_explainer = shap.TreeExplainer(rul_model)

    # 3. LSTM Deep Learning Models
    print("Loading LSTM Time-Series Models...")
    lstm_model = load_model("../models_m/lstm_model.keras")
    lstm_scaler = joblib.load("../models_m/scaler_lstm.pkl")
    lstm_encoder = joblib.load("../models_m/label_encoder.pkl")

    print("All ML and Deep Learning models_m loaded successfully!")
except Exception as e:
    print(f"Startup Error: {e}")


class MotorData(BaseModel):
    voltage: float
    current: float
    power: float
    power_factor: float
    frequency: float
    voltage_imbalance: float
    current_imbalance: float
    speed: float
    torque: float
    load_percent: float
    vibration_x: float
    vibration_y: float
    vibration_z: float
    vibration_rms: float
    fft_peak_freq: float
    bearing_temp: float
    winding_temp: float
    ambient_temp: float
    temp_rise: float
    humidity: float
    efficiency: float
    current_spike: float


@app.get("/")
async def root():
    return {"message": "Motor Predictive Maintenance API is running smoothly!"}


@app.post("/predict")
async def predict_motor_health(data: MotorData):
    try:
        feature_names = [
            'VOLTAGE', 'CURRENT', 'POWER', 'POWER_FACTOR', 'FREQUENCY',
            'VOLTAGE_IMBALANCE', 'CURRENT_IMBALANCE', 'SPEED', 'TORQUE',
            'LOAD_PERCENT', 'VIBRATION_X', 'VIBRATION_Y', 'VIBRATION_Z',
            'VIBRATION_RMS', 'FFT_PEAK_FREQ', 'BEARING_TEMP', 'WINDING_TEMP',
            'AMBIENT_TEMP', 'TEMP_RISE', 'HUMIDITY', 'EFFICIENCY', 'CURRENT_SPIKE'
        ]

        input_list = [
            data.voltage, data.current, data.power, data.power_factor, data.frequency,
            data.voltage_imbalance, data.current_imbalance, data.speed, data.torque,
            data.load_percent, data.vibration_x, data.vibration_y, data.vibration_z,
            data.vibration_rms, data.fft_peak_freq, data.bearing_temp, data.winding_temp,
            data.ambient_temp, data.temp_rise, data.humidity, data.efficiency, data.current_spike
        ]

        # Add the current reading to our rolling memory buffer
        telemetry_buffer.append(input_list)

        input_df = pd.DataFrame([input_list], columns=feature_names)

        # ==========================================
        # 1. XGBoost Classification (Instant Snapshot)
        # ==========================================
        class_scaled_data = classifier_scaler.transform(input_df)
        deformity_prediction = classifier_model.predict(class_scaled_data)[0]

        # ==========================================
        # 2. XGBoost RUL Regression & SHAP Analysis
        # ==========================================
        rul_scaled_data = rul_scaler.transform(input_df)
        predicted_hours = rul_model.predict(rul_scaled_data)[0]

        shap_values = rul_explainer.shap_values(rul_scaled_data)
        feature_impacts = []
        for i, name in enumerate(feature_names):
            feature_impacts.append({
                "feature": name,
                "impact": float(shap_values[0][i])
            })

        feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        top_3_impacts = feature_impacts[:3]

        explanations = []
        for item in top_3_impacts:
            direction = "decreased" if item["impact"] < 0 else "increased"
            explanations.append(
                f"{item['feature']} {direction} the estimated lifespan by {abs(item['impact']):.1f} hours."
            )

        # ==========================================
        # 3. LSTM Deep Learning (Time-Series Analysis)
        # ==========================================
        lstm_status = "Waiting for data..."

        # Only run the neural network if we have collected 30 consecutive readings
        if len(telemetry_buffer) == SEQUENCE_LENGTH:
            # Convert memory to a pandas dataframe so the scaler can read it
            buffer_df = pd.DataFrame(list(telemetry_buffer), columns=feature_names)

            # Scale all 30 rows
            buffer_scaled = lstm_scaler.transform(buffer_df)

            # Reshape into the required 3D Tensor: (1 Sample, 30 Timestamps, 22 Features)
            lstm_input = np.expand_dims(buffer_scaled, axis=0)

            # Predict
            lstm_pred_prob = lstm_model.predict(lstm_input, verbose=0)
            lstm_pred_class = np.argmax(lstm_pred_prob, axis=1)

            # Decode the integer back into a readable label
            lstm_decoded_label = lstm_encoder.inverse_transform(lstm_pred_class)[0]

            # Assuming 0 is healthy and 1 is deformity based on your training script
            lstm_status = "Deep Learning Status: Healthy Trend" if lstm_decoded_label == 0 else "Deep Learning Status: DEGRADATION DETECTED"
        else:
            lstm_status = f"LSTM Buffering Sequence: ({len(telemetry_buffer)}/{SEQUENCE_LENGTH})"

        # ==========================================
        # 4. FORMAT THE REPORT
        # ==========================================
        physical_status = "Healthy" if deformity_prediction == 0 else "Deformity Detected"

        if predicted_hours < 500:
            urgency = "CRITICAL: Immediate Maintenance Required"
        elif predicted_hours < 2000:
            urgency = "WARNING: Schedule Maintenance Soon"
        else:
            urgency = "NOMINAL: Normal Operation"

        return {
            "physical_status": physical_status,
            "deformity_code": int(deformity_prediction),
            "estimated_rul_hours": round(float(predicted_hours), 1),
            "maintenance_urgency": urgency,
            "ai_explanations": explanations,
            "lstm_time_series_analysis": lstm_status
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))