import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import joblib

# 1. Define all paths
DATASET_PATH = "../dataset_m/motor_data.csv"

# Classification Paths
CLASS_MODEL_SAVE_PATH = "../models_m/ensemble_model.pkl"
CLASS_SCALER_SAVE_PATH = "../models_m/scaler_ensemble.pkl"

# Regression Paths
RUL_MODEL_SAVE_PATH = "../models_m/rul_model.pkl"
RUL_SCALER_SAVE_PATH = "../models_m/scaler_rul.pkl"


def main():
    print("Loading dataset_m...")
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find dataset_m at {DATASET_PATH}")
        return

    # 2. Define the exact 22 features
    features = [
        'VOLTAGE', 'CURRENT', 'POWER', 'POWER_FACTOR', 'FREQUENCY',
        'VOLTAGE_IMBALANCE', 'CURRENT_IMBALANCE', 'SPEED', 'TORQUE',
        'LOAD_PERCENT', 'VIBRATION_X', 'VIBRATION_Y', 'VIBRATION_Z',
        'VIBRATION_RMS', 'FFT_PEAK_FREQ', 'BEARING_TEMP', 'WINDING_TEMP',
        'AMBIENT_TEMP', 'TEMP_RISE', 'HUMIDITY', 'EFFICIENCY', 'CURRENT_SPIKE'
    ]

    if 'DEFORMITY_LABEL' not in df.columns or 'RUL_HOURS' not in df.columns:
        print("Error: Target columns ('DEFORMITY_LABEL' or 'RUL_HOURS') not found in dataset_m.")
        return

    X = df[features]

    print("\n" + "=" * 50)
    print(" TASK 1: TRAINING DEFORMITY CLASSIFIER (XGBOOST)")
    print("=" * 50)

    y_class = df['DEFORMITY_LABEL']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

    class_scaler = StandardScaler()
    X_train_c_scaled = class_scaler.fit_transform(X_train_c)
    X_test_c_scaled = class_scaler.transform(X_test_c)

    # Initialize XGBoost Classifier (n_jobs=-1 uses all CPU cores for maximum speed)
    xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_clf.fit(X_train_c_scaled, y_train_c)

    print("Evaluating XGBoost Classifier...")
    y_pred_c = xgb_clf.predict(X_test_c_scaled)
    print(f"Accuracy: {accuracy_score(y_test_c, y_pred_c) * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test_c, y_pred_c))

    # We keep the old save names so you don't have to rewrite your FastAPI backend_m!
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