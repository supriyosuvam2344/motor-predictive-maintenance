import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed so the data is reproducible
np.random.seed(42)

NUM_SAMPLES = 100000


def generate_motor_data():
    print(f"Generating {NUM_SAMPLES} rows of REALISTIC noisy motor telemetry...")

    # Target Label: 80% Healthy (0), 20% Deformity (1)
    labels = np.random.choice([0, 1], size=NUM_SAMPLES, p=[0.8, 0.2])

    timestamps = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(NUM_SAMPLES)]
    motor_ids = np.random.choice(['MTR-01', 'MTR-02', 'MTR-03', 'MTR-04', 'MTR-05'], size=NUM_SAMPLES)

    # NEW: We multiply the spread (standard deviation) to create a messy "grey area"
    # where healthy and deformed motors look identical to the sensors.
    NOISE_MULTIPLIER = 3.5

    def gen_sensor(h_mean, h_std, d_mean, d_std):
        return np.where(
            labels == 0,
            np.random.normal(h_mean, h_std * NOISE_MULTIPLIER, NUM_SAMPLES),
            np.random.normal(d_mean, d_std * NOISE_MULTIPLIER, NUM_SAMPLES)
        )

    data = {
        'TIMESTAMP': timestamps,
        'MOTOR_ID': motor_ids,

        # Notice we moved the means slightly closer together as well
        'VOLTAGE': gen_sensor(398, 5, 388, 10),
        'CURRENT': gen_sensor(48, 4, 56, 8),
        'POWER': gen_sensor(26, 2, 29, 4),
        'POWER_FACTOR': gen_sensor(0.90, 0.03, 0.86, 0.05),
        'FREQUENCY': gen_sensor(50, 0.2, 49.6, 0.6),
        'VOLTAGE_IMBALANCE': gen_sensor(2.0, 1.0, 4.0, 1.5),
        'CURRENT_IMBALANCE': gen_sensor(2.5, 1.0, 5.0, 2.0),

        'SPEED': gen_sensor(1470, 15, 1435, 25),
        'TORQUE': gen_sensor(155, 10, 170, 15),
        'LOAD_PERCENT': gen_sensor(78, 8, 88, 12),

        'VIBRATION_X': gen_sensor(3.5, 1.5, 6.5, 2.5),
        'VIBRATION_Y': gen_sensor(3.0, 1.2, 5.5, 2.0),
        'VIBRATION_Z': gen_sensor(2.5, 1.0, 4.5, 1.8),
        'VIBRATION_RMS': gen_sensor(4.0, 1.5, 8.0, 3.0),
        'FFT_PEAK_FREQ': gen_sensor(140, 20, 220, 40),

        'BEARING_TEMP': gen_sensor(70, 6, 88, 10),
        'WINDING_TEMP': gen_sensor(85, 8, 105, 12),
        'AMBIENT_TEMP': np.random.normal(25, 5, NUM_SAMPLES),
        'TEMP_RISE': gen_sensor(45, 8, 70, 15),

        'HUMIDITY': np.random.normal(50, 15, NUM_SAMPLES),

        'EFFICIENCY': gen_sensor(90, 4, 84, 6),
        'CURRENT_SPIKE': np.where(labels == 0, np.random.choice([0, 1], p=[0.95, 0.05]),
                                  np.random.choice([0, 1], p=[0.6, 0.4])),

        'DEFORMITY_LABEL': labels,

        # Regression Target (RUL) with added random noise
        'RUL_HOURS': np.where(labels == 0,
                              np.random.uniform(3000, 10000, NUM_SAMPLES),
                              np.random.uniform(0, 2000, NUM_SAMPLES))
    }

    df = pd.DataFrame(data)
    df = df.round(3)

    # NEW: Introduce Label Noise (The "Edge Cases")
    # We will randomly flip 5% of the labels.
    # This guarantees the model will get confused and cannot hit 100% accuracy.
    num_to_flip = int(NUM_SAMPLES * 0.05)
    flip_indices = np.random.choice(df.index, num_to_flip, replace=False)
    df.loc[flip_indices, 'DEFORMITY_LABEL'] = 1 - df.loc[flip_indices, 'DEFORMITY_LABEL']

    output_filename = "motor_data.csv"
    df.to_csv(output_filename, index=False)
    print(f"Success! {output_filename} created. Train your models again to see the new accuracy!")


if __name__ == "__main__":
    generate_motor_data()