import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

DATASET_PATH = "../dataset/motor_data.csv"
MODEL_SAVE_PATH = "rul_model.pkl"
SCALER_SAVE_PATH = "scaler_rul.pkl"


def main():
    print("Loading dataset for RUL Prediction...")
    df = pd.read_csv(DATASET_PATH)

    features = [
        'VOLTAGE', 'CURRENT', 'POWER', 'POWER_FACTOR', 'FREQUENCY',
        'VOLTAGE_IMBALANCE', 'CURRENT_IMBALANCE', 'SPEED', 'TORQUE',
        'LOAD_PERCENT', 'VIBRATION_X', 'VIBRATION_Y', 'VIBRATION_Z',
        'VIBRATION_RMS', 'FFT_PEAK_FREQ', 'BEARING_TEMP', 'WINDING_TEMP',
        'AMBIENT_TEMP', 'TEMP_RISE', 'HUMIDITY', 'EFFICIENCY', 'CURRENT_SPIKE'
    ]

    # Notice we are targeting the new continuous number column!
    target_column = 'RUL_HOURS'

    X = df[features]
    y = df[target_column]

    print("Splitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the Regressor (Not Classifier)
    print("Training Random Forest Regressor...")
    rul_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rul_model.fit(X_train_scaled, y_train)

    print("\nEvaluating RUL Model...")
    y_pred = rul_model.predict(X_test_scaled)

    # Regression uses different metrics. MAE tells us how many hours off we are on average.
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): +/- {mae:.1f} Hours")
    print(f"R-Squared Score: {r2:.2f} (Closer to 1.0 is better)")

    print("\nSaving RUL model and scaler...")
    joblib.dump(rul_model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Success! Saved {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()