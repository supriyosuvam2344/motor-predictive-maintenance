import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import xgboost as xgb

# ============================================
# DYNAMIC PATH SETUP (Method 1)
# ============================================
# 1. Get the directory where this training script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Point directly to the backend's models folder
BACKEND_MODELS_DIR = os.path.join(BASE_DIR, "..", "backend", "models")

# 3. Automatically create the folder if it doesn't exist!
os.makedirs(BACKEND_MODELS_DIR, exist_ok=True)

# 4. Define the robust read/save paths
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "motor_data.csv")

# Save paths straight into the backend
CLASSIFIER_MODEL_PATH = os.path.join(BACKEND_MODELS_DIR, "ensemble_model.pkl")
CLASSIFIER_SCALER_PATH = os.path.join(BACKEND_MODELS_DIR, "scaler_ensemble.pkl")
RUL_MODEL_PATH = os.path.join(BACKEND_MODELS_DIR, "rul_model.pkl")
RUL_SCALER_PATH = os.path.join(BACKEND_MODELS_DIR, "scaler_rul.pkl")


# ============================================
# SYNTHETIC RUL GENERATOR
# ============================================
def calculate_synthetic_rul(row):
    """Calculates RUL based on Temperature and Vibration degradation rules."""
    BASE_LIFESPAN = 50000.0
    RATED_TEMP = 80.0

    # 1. Calculate Temperature Penalty (Arrhenius Rule)
    temp_diff = max(0, row['Temperature_C'] - RATED_TEMP)
    temp_penalty = 2 ** (temp_diff / 10.0)

    # 2. Calculate Vibration Penalty (ISO 10816 Standard)
    # Convert X, Y, Z to a single RMS magnitude
    vib_rms = np.sqrt(row['Vibration_X_mm_s'] ** 2 + row['Vibration_Y_mm_s'] ** 2 + row['Vibration_Z_mm_s'] ** 2)
    if vib_rms < 2.8:
        vib_penalty = 1.0
    elif vib_rms < 7.1:
        vib_penalty = 1.5
    else:
        vib_penalty = 5.0

    # 3. Calculate Final RUL
    calculated_rul = BASE_LIFESPAN / (temp_penalty * vib_penalty)
    return max(0.0, calculated_rul)


def main():
    # ============================================
    # 1. LOAD & CLEAN DATA
    # ============================================
    print("Loading dataset...")
    try:
        df = pd.read_csv(DATASET_PATH)
        # Clean whitespace and remove rows with missing targets
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=['Anomaly'])
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {DATASET_PATH}")
        return

    # Generate the RUL target column mathematically
    print("Generating synthetic RUL targets...")
    df['RUL_HOURS'] = df.apply(calculate_synthetic_rul, axis=1)

    # Define the 12 features expected by the models
    features = [
        'Temperature_C', 'Bearing_Temp_C', 'Vibration_X_mm_s', 'Vibration_Y_mm_s',
        'Vibration_Z_mm_s', 'Current_A', 'Voltage_V', 'Power_kW', 'Power_Factor',
        'RPM', 'Load_Percent', 'Ambient_Temp_C'
    ]

    # Fill any remaining NaNs with the median
    df[features] = df[features].fillna(df[features].median())

    X = df[features]

    # ============================================
    # 2. TRAIN CLASSIFICATION MODEL (Anomaly Detection)
    # ============================================
    print("\n--- Training Anomaly Classifier ---")
    y_class = df['Anomaly']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42,
                                                                stratify=y_class)

    classifier_scaler = StandardScaler()
    X_train_c_scaled = classifier_scaler.fit_transform(X_train_c)
    X_test_c_scaled = classifier_scaler.transform(X_test_c)

    # Use XGBoost for powerful tabular classification
    classifier_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=99  # Account for the 1% anomaly rate imbalance
    )

    classifier_model.fit(X_train_c_scaled, y_train_c)

    # Evaluate
    y_pred_c = classifier_model.predict(X_test_c_scaled)
    print("Classification Report:")
    print(classification_report(y_test_c, y_pred_c))

    # Save
    joblib.dump(classifier_model, CLASSIFIER_MODEL_PATH)
    joblib.dump(classifier_scaler, CLASSIFIER_SCALER_PATH)
    print(f"Saved Anomaly Model & Scaler to: {BACKEND_MODELS_DIR}")

    # ============================================
    # 3. TRAIN REGRESSION MODEL (RUL Prediction)
    # ============================================
    print("\n--- Training RUL Regressor ---")
    y_reg = df['RUL_HOURS']

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    rul_scaler = StandardScaler()
    X_train_r_scaled = rul_scaler.fit_transform(X_train_r)
    X_test_r_scaled = rul_scaler.transform(X_test_r)

    # Use XGBoost for RUL Regression
    rul_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1
    )

    rul_model.fit(X_train_r_scaled, y_train_r)

    # Evaluate
    y_pred_r = rul_model.predict(X_test_r_scaled)
    mae = mean_absolute_error(y_test_r, y_pred_r)
    r2 = r2_score(y_test_r, y_pred_r)

    print(f"Mean Absolute Error: {mae:.2f} Hours")
    print(f"R-Squared Score: {r2:.4f}")

    # Save
    joblib.dump(rul_model, RUL_MODEL_PATH)
    joblib.dump(rul_scaler, RUL_SCALER_PATH)
    print(f"Saved RUL Model & Scaler to: {BACKEND_MODELS_DIR}")

    print("\n===========================================")
    print("ALL TRAINING COMPLETE & MODELS MOVED TO BACKEND!")
    print("===========================================")


if __name__ == "__main__":
    main()