import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score
)

import os
# Suppress TensorFlow logging spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

# ============================================
# PATHS
# ============================================
# 1. Get the directory where this training script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Point directly to the backend's models folder
BACKEND_MODELS_DIR = os.path.join(BASE_DIR, "..", "backend", "models")

# 3. Automatically create the folder if it doesn't exist!
os.makedirs(BACKEND_MODELS_DIR, exist_ok=True)

# 4. Define the robust save paths
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "motor_data.csv")
MODEL_SAVE_PATH = os.path.join(BACKEND_MODELS_DIR, "lstm_model.keras")
SCALER_SAVE_PATH = os.path.join(BACKEND_MODELS_DIR, "scaler_lstm.pkl")
ENCODER_PATH = os.path.join(BACKEND_MODELS_DIR, "label_encoder.pkl")

def main():
    # ============================================
    # LOAD DATA
    # ============================================
    print("Loading dataset_m...")
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find dataset_m at {DATASET_PATH}")
        return

    # ============================================
    # CLEANING & STANDARDIZING NAMES
    # ============================================
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    target_col = 'Anomaly'
    time_col = 'Timestamp'
    motor_col = 'Motor_ID'

    if target_col not in df.columns or time_col not in df.columns or motor_col not in df.columns:
        print(f"Error: Could not find {target_col}, {time_col}, or {motor_col} in dataset.")
        return

    df[time_col] = pd.to_datetime(df[time_col], format='mixed', dayfirst=True)
    df = df.sort_values([motor_col, time_col])

    # ============================================
    # LABEL ENCODING
    # ============================================
    label_encoder = LabelEncoder()
    df[target_col] = label_encoder.fit_transform(df[target_col])

    features = [
        'Temperature_C', 'Bearing_Temp_C', 'Vibration_X_mm_s', 'Vibration_Y_mm_s',
        'Vibration_Z_mm_s', 'Current_A', 'Voltage_V', 'Power_kW', 'Power_Factor',
        'RPM', 'Load_Percent', 'Ambient_Temp_C'
    ]

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # ============================================
    # CREATE TIME SEQUENCES
    # ============================================
    SEQUENCE_LENGTH = 30
    X_seq = []
    y_seq = []

    print("Building Time Sequences...")
    for motor_id in df[motor_col].unique():
        motor_df = df[df[motor_col] == motor_id].sort_values(time_col)
        values = motor_df[features].values
        labels = motor_df[target_col].values

        for i in range(SEQUENCE_LENGTH, len(motor_df)):
            X_seq.append(values[i-SEQUENCE_LENGTH:i])
            y_seq.append(labels[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # ============================================
    # SPLIT
    # ============================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.20, random_state=42, stratify=y_seq
    )

    # ============================================
    # MODEL
    # ============================================
    num_classes = len(label_encoder.classes_)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(features))),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # ============================================
    # CALLBACKS
    # ============================================
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # ============================================
    # TRAIN WITH UPDATED CLASS WEIGHT
    # ============================================
    print("Training LSTM with high class sensitivity...")
    # Using 99.0 for the minority class (1) based on ~1% anomaly rate
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        class_weight={0: 1.0, 1: 99.0},
        verbose=1
    )

    # Evaluation and Saving remain the same...
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)
    print(classification_report(y_test, y_pred))

    model.save(MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

if __name__ == "__main__":
    main()