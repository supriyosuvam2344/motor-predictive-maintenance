import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import joblib

# 1. Define all paths
DATASET_PATH = "../dataset/motor_data.csv"

# Classification Paths
CLASS_MODEL_SAVE_PATH = "../models/ensemble_model.pkl"
CLASS_SCALER_SAVE_PATH = "../models/scaler_ensemble.pkl"

# Regression Paths
RUL_MODEL_SAVE_PATH = "../models/rul_model.pkl"
RUL_SCALER_SAVE_PATH = "../models/scaler_rul.pkl"


def calculate_rul(row):
    """Generates synthetic RUL using thermodynamics and vibration physics."""
    BASE_LIFESPAN = 50000
    RATED_TEMP = 80.0

    # 1. Temperature Penalty
    temp = row.get('Bearing_Temp_C', row.get('Temperature_C', 80.0))
    temp_diff = max(0, temp - RATED_TEMP)
    temp_penalty = 2 ** (temp_diff / 10.0)

    # 2. Vibration Penalty (Calculate approx RMS from X, Y, Z)
    vib_x = row.get('Vibration_X_mm_s', 0)
    vib_y = row.get('Vibration_Y_mm_s', 0)
    vib_z = row.get('Vibration_Z_mm_s', 0)
    vib_rms = np.sqrt(vib_x ** 2 + vib_y ** 2 + vib_z ** 2)

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
    print("Loading dataset...")
    try:
        df = pd.read_csv(DATASET_PATH)
        df.columns = df.columns.str.strip()

        # --- FORCED CLEANING ---
        # 1. Force convert to numeric, turning errors (text/symbols) into NaN
        df['Anomaly'] = pd.to_numeric(df['Anomaly'], errors='coerce')

        # 2. Now drop the NaNs
        df = df.dropna(subset=['Anomaly'])

        # 3. Convert to clean integers
        df['Anomaly'] = df['Anomaly'].astype(int)
        # -----------------------

    except FileNotFoundError:
        print(f"Error: Could not find dataset_m at {DATASET_PATH}")
        return

    # 2. Define the exact features based on the new dataset image
    features = [
        'Temperature_C', 'Bearing_Temp_C', 'Vibration_X_mm_s', 'Vibration_Y_mm_s',
        'Vibration_Z_mm_s', 'Current_A', 'Voltage_V', 'Power_kW', 'Power_Factor',
        'RPM', 'Load_Percent', 'Ambient_Temp_C'
    ]



    # Handle missing RUL_HOURS dynamically
    if 'RUL_HOURS' not in df.columns:
        print("Calculating RUL...")
        df['RUL_HOURS'] = df.apply(calculate_rul, axis=1)

    # Handle classification target (Anomaly)
    target_col = 'Anomaly'
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in dataset.")
        return

    X = df[features]

    print("\n" + "=" * 50)
    print(" TASK 1: TRAINING ANOMALY CLASSIFIER (XGBOOST)")
    print("=" * 50)

    y_class = df[target_col]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

    class_scaler = StandardScaler()
    X_train_c_scaled = class_scaler.fit_transform(X_train_c)
    X_test_c_scaled = class_scaler.transform(X_test_c)

    # Initialize XGBoost Classifier
    xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_clf.fit(X_train_c_scaled, y_train_c)

    print("Evaluating XGBoost Classifier...")
    y_pred_c = xgb_clf.predict(X_test_c_scaled)
    print(f"Accuracy: {accuracy_score(y_test_c, y_pred_c) * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test_c, y_pred_c))

    joblib.dump(xgb_clf, CLASS_MODEL_SAVE_PATH)
    joblib.dump(class_scaler, CLASS_SCALER_SAVE_PATH)
    print(f"Saved: {CLASS_MODEL_SAVE_PATH} & {CLASS_SCALER_SAVE_PATH}")

    print("\n" + "=" * 50)
    print(" TASK 2: TRAINING REMAINING USEFUL LIFE (XGBOOST)")
    print("=" * 50)

    y_reg = df['RUL_HOURS']

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    rul_scaler = StandardScaler()
    X_train_r_scaled = rul_scaler.fit_transform(X_train_r)
    X_test_r_scaled = rul_scaler.transform(X_test_r)

    # Initialize XGBoost Regressor
    xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_reg.fit(X_train_r_scaled, y_train_r)

    print("Evaluating XGBoost Regressor...")
    y_pred_r = xgb_reg.predict(X_test_r_scaled)

    print(f"Mean Absolute Error (MAE): +/- {mean_absolute_error(y_test_r, y_pred_r):.1f} Hours")
    print(f"R-Squared Score: {r2_score(y_test_r, y_pred_r):.2f}")

    joblib.dump(xgb_reg, RUL_MODEL_SAVE_PATH)
    joblib.dump(rul_scaler, RUL_SCALER_SAVE_PATH)
    print(f"Saved: {RUL_MODEL_SAVE_PATH} & {RUL_SCALER_SAVE_PATH}")

    print("\n" + "=" * 50)
    print(" ALL TRAINING COMPLETE! ")
    print("=" * 50)


if __name__ == "__main__":
    main()