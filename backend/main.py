from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Motor Predictive Maintenance API")

# Allow React frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models and Scalers at startup
try:
    print("Loading models...")
    # 1. Classification Models (Deformity Detection)
    classifier_model = joblib.load("ensemble_model.pkl")
    classifier_scaler = joblib.load("scaler_ensemble.pkl")

    # 2. Regression Models (Remaining Useful Life)
    rul_model = joblib.load("rul_model.pkl")
    rul_scaler = joblib.load("scaler_rul.pkl")

    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models. Ensure all 4 .pkl files are in the backend folder. Error: {e}")


# Define the exact 24-feature input structure
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


# Simple homepage to prevent 404 errors in the browser
@app.get("/")
async def root():
    return {"message": "Motor Predictive Maintenance API is running smoothly!"}


@app.post("/predict")
async def predict_motor_health(data: MotorData):
    # noinspection PyShadowingNames
    try:
        # Define the exact column names the model was trained on
        feature_names = [
            'VOLTAGE', 'CURRENT', 'POWER', 'POWER_FACTOR', 'FREQUENCY',
            'VOLTAGE_IMBALANCE', 'CURRENT_IMBALANCE', 'SPEED', 'TORQUE',
            'LOAD_PERCENT', 'VIBRATION_X', 'VIBRATION_Y', 'VIBRATION_Z',
            'VIBRATION_RMS', 'FFT_PEAK_FREQ', 'BEARING_TEMP', 'WINDING_TEMP',
            'AMBIENT_TEMP', 'TEMP_RISE', 'HUMIDITY', 'EFFICIENCY', 'CURRENT_SPIKE'
        ]

        # Extract values from the incoming Pydantic payload
        input_list = [
            data.voltage, data.current, data.power, data.power_factor, data.frequency,
            data.voltage_imbalance, data.current_imbalance, data.speed, data.torque,
            data.load_percent, data.vibration_x, data.vibration_y, data.vibration_z,
            data.vibration_rms, data.fft_peak_freq, data.bearing_temp, data.winding_temp,
            data.ambient_temp, data.temp_rise, data.humidity, data.efficiency, data.current_spike
        ]

        # Convert to Pandas DataFrame to prevent sklearn warnings
        input_df = pd.DataFrame([input_list], columns=feature_names)

        # ==========================================
        # PREDICTION 1: Deformity Classification
        # ==========================================
        class_scaled_data = classifier_scaler.transform(input_df)
        deformity_prediction = classifier_model.predict(class_scaled_data)[0]

        # ==========================================
        # PREDICTION 2: Remaining Useful Life (RUL)
        # ==========================================
        rul_scaled_data = rul_scaler.transform(input_df)
        predicted_hours = rul_model.predict(rul_scaled_data)[0]

        # ==========================================
        # FORMAT THE DIAGNOSTIC REPORT
        # ==========================================

        # Determine current physical status
        physical_status = "Healthy" if deformity_prediction == 0 else "Deformity Detected"

        # Determine RUL urgency
        if predicted_hours < 500:
            urgency = "CRITICAL: Immediate Maintenance Required"
        elif predicted_hours < 2000:
            urgency = "WARNING: Schedule Maintenance Soon"
        else:
            urgency = "NOMINAL: Normal Operation"

        return {
            "physical_status": physical_status,
            "deformity_code": int(deformity_prediction),
            "estimated_rul_hours": round(predicted_hours, 1),
            "maintenance_urgency": urgency
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))