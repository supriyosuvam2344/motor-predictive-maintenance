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

DATASET_PATH = "../dataset/motor_data.csv"
MODEL_SAVE_PATH = "../models/lstm_model.keras"
SCALER_SAVE_PATH = "../models/scaler_lstm.pkl"
ENCODER_PATH = "../models/label_encoder.pkl"


def main():
    # ============================================
    # LOAD DATA
    # ============================================
    print("Loading dataset...")
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {DATASET_PATH}")
        return

    # ============================================
    # CLEANING & STANDARDIZING NAMES
    # ============================================
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Standardize column names based on your new dataset
    target_col = 'Anomaly'
    time_col = 'Timestamp'
    motor_col = 'Motor_ID'

    if target_col not in df.columns or time_col not in df.columns or motor_col not in df.columns:
        print(f"Error: Could not find {target_col}, {time_col}, or {motor_col} in dataset.")
        return

    # ============================================
    # TIMESTAMP
    # ============================================
    # dayfirst=True tells Pandas to treat the first number as the day (DD-MM-YYYY)
    # df[time_col] = pd.to_datetime(df[time_col], dayfirst=True)
    df[time_col] = pd.to_datetime(df[time_col], format='mixed', dayfirst=True)
    df = df.sort_values([motor_col, time_col])



    # ============================================
    # LABEL ENCODING
    # ============================================
    label_encoder = LabelEncoder()
    df[target_col] = label_encoder.fit_transform(df[target_col])

    # ============================================
    # FEATURES (Updated to the 12 new features)
    # ============================================
    features = [
        'Temperature_C', 'Bearing_Temp_C', 'Vibration_X_mm_s', 'Vibration_Y_mm_s',
        'Vibration_Z_mm_s', 'Current_A', 'Voltage_V', 'Power_kW', 'Power_Factor',
        'RPM', 'Load_Percent', 'Ambient_Temp_C'
    ]

    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in dataset: {missing_cols}")
        return

    # ============================================
    # SCALE
    # ============================================
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
            X_seq.append(values[i - SEQUENCE_LENGTH:i])
            y_seq.append(labels[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    print("Sequence Shape:", X_seq.shape)
    # Example: (25000, 30, 12)

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
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # ============================================
    # COMPILE
    # ============================================
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
    # TRAIN
    # ============================================
    print("Training LSTM...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        class_weight = {0: 1.0, 1: 99.0},
        verbose=1
    )

    # ============================================
    # USE TRAINING HISTORY
    # ============================================
    print("\nTraining Metrics:")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")



    # ============================================
    # EVALUATE
    # ============================================
    print("Evaluating...")
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in label_encoder.classes_]))

    # ============================================
    # SAVE
    # ============================================
    model.save(MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    print(f"Model Saved Successfully to {MODEL_SAVE_PATH}")

    # ============================================
    # QUICK INFERENCE TEST
    # ============================================
    print("\nRunning Quick Inference Test...")
    test_model = load_model(MODEL_SAVE_PATH)
    test_encoder = joblib.load(ENCODER_PATH)

    # Create a random sequence matching your new shape (1 sample, 30 timestamps, 12 features)
    sequence = np.random.rand(1, SEQUENCE_LENGTH, len(features))

    prediction = test_model.predict(sequence, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)
    fault = test_encoder.inverse_transform(predicted_class)

    print("Predicted Fault on Random Data:")
    print(fault[0])


if __name__ == "__main__":
    main()