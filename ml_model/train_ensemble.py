import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Define paths
DATASET_PATH = "../dataset/motor_data.csv"
MODEL_SAVE_PATH = "ensemble_model.pkl"
SCALER_SAVE_PATH = "scaler_ensemble.pkl"


def main():
    print("Loading dataset...")
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {DATASET_PATH}")
        return

    # 2. Your exact 24 features
    features = [
        'VOLTAGE', 'CURRENT', 'POWER', 'POWER_FACTOR', 'FREQUENCY',
        'VOLTAGE_IMBALANCE', 'CURRENT_IMBALANCE', 'SPEED', 'TORQUE',
        'LOAD_PERCENT', 'VIBRATION_X', 'VIBRATION_Y', 'VIBRATION_Z',
        'VIBRATION_RMS', 'FFT_PEAK_FREQ', 'BEARING_TEMP', 'WINDING_TEMP',
        'AMBIENT_TEMP', 'TEMP_RISE', 'HUMIDITY', 'EFFICIENCY', 'CURRENT_SPIKE'
    ]
    target_column = 'DEFORMITY_LABEL'

    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found.")
        return

    X = df[features]
    y = df[target_column]

    # 3. Split and Scale (Scaling is strictly required because of the SVM)
    print("Splitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Initialize the base models
    print("Initializing base models...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    # IMPORTANT: probability=True is required for 'soft' voting
    svm = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')

    # 5. Create the Ensemble (Voting Classifier)
    print("Training Ensemble Model (Random Forest + SVM)...")
    # 'soft' voting means it averages the probability scores of both models
    # rather than just taking a hard 1 or 0 vote. This usually yields higher accuracy.
    ensemble_model = VotingClassifier(
        estimators=[('random_forest', rf), ('support_vector', svm)],
        voting='soft'
    )

    ensemble_model.fit(X_train_scaled, y_train)

    # 6. Evaluate the Ensemble
    print("\nEvaluating Ensemble performance...")
    y_pred = ensemble_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Ensemble Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save the final pipeline
    print("\nSaving Ensemble model and scaler...")
    joblib.dump(ensemble_model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)

    print(f"Success! Saved to {MODEL_SAVE_PATH} and {SCALER_SAVE_PATH}")


if __name__ == "__main__":
    main()