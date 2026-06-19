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

from tensorflow.keras.models import Sequential
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
MODEL_SAVE_PATH = "lstm_model.keras"
SCALER_SAVE_PATH = "scaler_lstm.pkl"
ENCODER_PATH = "label_encoder.pkl"

# ============================================
# LOAD DATA
# ============================================

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

# ============================================
# CLEANING
# ============================================

df.drop_duplicates(inplace=True)
df.fillna(
    df.median(numeric_only=True),
    inplace=True
)

# ============================================
# TIMESTAMP
# ============================================

df['TIMESTAMP'] = pd.to_datetime(
    df['TIMESTAMP']
)

df = df.sort_values(
    ['MOTOR_ID', 'TIMESTAMP']
)

# ============================================
# LABEL ENCODING
# ============================================

label_encoder = LabelEncoder()

df['DEFORMITY_LABEL'] = (
    label_encoder.fit_transform(
        df['DEFORMITY_LABEL']
    )
)

# ============================================
# FEATURES
# ============================================

features = [

    'VOLTAGE',
    'CURRENT',
    'POWER',
    'POWER_FACTOR',
    'FREQUENCY',

    'VOLTAGE_IMBALANCE',
    'CURRENT_IMBALANCE',

    'SPEED',
    'TORQUE',
    'LOAD_PERCENT',

    'VIBRATION_X',
    'VIBRATION_Y',
    'VIBRATION_Z',
    'VIBRATION_RMS',
    'FFT_PEAK_FREQ',

    'BEARING_TEMP',
    'WINDING_TEMP',
    'AMBIENT_TEMP',
    'TEMP_RISE',

    'HUMIDITY',
    'EFFICIENCY',
    'CURRENT_SPIKE'
]

# ============================================
# SCALE
# ============================================

scaler = StandardScaler()

df[features] = scaler.fit_transform(
    df[features]
)

# ============================================
# CREATE TIME SEQUENCES
# ============================================

SEQUENCE_LENGTH = 30

X_seq = []
y_seq = []

for motor_id in df['MOTOR_ID'].unique():

    motor_df = (
        df[df['MOTOR_ID'] == motor_id]
        .sort_values('TIMESTAMP')
    )

    values = (
        motor_df[features]
        .values
    )

    labels = (
        motor_df['DEFORMITY_LABEL']
        .values
    )

    for i in range(
            SEQUENCE_LENGTH,
            len(motor_df)
    ):

        X_seq.append(
            values[
                i-SEQUENCE_LENGTH:i
            ]
        )

        y_seq.append(
            labels[i]
        )

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print("Sequence Shape:")
print(X_seq.shape)

# Example:
# (25000,30,22)

# ============================================
# SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_seq,
    y_seq,
    test_size=0.20,
    random_state=42,
    stratify=y_seq
)

# ============================================
# MODEL
# ============================================

num_classes = len(
    label_encoder.classes_
)

model = Sequential([

    LSTM(
        128,
        return_sequences=True,
        input_shape=(
            SEQUENCE_LENGTH,
            len(features)
        )
    ),

    Dropout(0.3),

    BatchNormalization(),

    LSTM(64),

    Dropout(0.3),

    Dense(
        64,
        activation='relu'
    ),

    Dropout(0.3),

    Dense(
        num_classes,
        activation='softmax'
    )
])

# ============================================
# COMPILE
# ============================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_cross_entropy',
    metrics=['accuracy']
)

# ============================================
# CALLBACKS
# ============================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3
)

# ============================================
# TRAIN
# ============================================

print("Training LSTM...")

history = model.fit(

    X_train,
    y_train,

    epochs=50,
    batch_size=64,

    validation_split=0.2,

    callbacks=[
        early_stop,
        reduce_lr
    ],

    verbose=1
)

# ============================================
# EVALUATE
# ============================================

print("Evaluating...")

y_prob = model.predict(
    X_test
)

y_pred = np.argmax(
    y_prob,
    axis=1
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"Accuracy: {accuracy*100:.2f}%"
)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=
        label_encoder.classes_
    )
)

# ============================================
# SAVE
# ============================================

model.save(
    MODEL_SAVE_PATH
)

joblib.dump(
    scaler,
    SCALER_SAVE_PATH
)

joblib.dump(
    label_encoder,
    ENCODER_PATH
)

print("Model Saved Successfully")

import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model(
    "lstm_model.keras"
)

scaler = joblib.load(
    "scaler_lstm.pkl"
)

encoder = joblib.load(
    "label_encoder.pkl"
)

# shape:
# (1 sample, 30 timestamps, 22 features)

sequence = np.random.rand(
    1,
    30,
    22
)

prediction = model.predict(
    sequence
)

predicted_class = np.argmax(
    prediction,
    axis=1
)

fault = encoder.inverse_transform(
    predicted_class
)

print("Predicted Fault:")
print(fault[0])