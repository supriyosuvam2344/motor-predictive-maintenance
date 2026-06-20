import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed so the data is reproducible
np.random.seed(42)

# Number of rows to generate
NUM_SAMPLES = 100000


def generate_motor_data():
    print(f"Generating {NUM_SAMPLES} rows of synthetic motor telemetry...")

    # Target Label: 80% Healthy (0), 20% Deformity (1)
    labels = np.random.choice([0, 1], size=NUM_SAMPLES, p=[0.8, 0.2])

    # Base setup
    timestamps = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(NUM_SAMPLES)]
    motor_ids = np.random.choice(['MTR-01', 'MTR-02', 'MTR-03', 'MTR-04', 'MTR-05'], size=NUM_SAMPLES)

    # Helper to generate data based on health status
    def gen_sensor(healthy_mean, healthy_std, deformed_mean, deformed_std):
        return np.where(
            labels == 0,
            np.random.normal(healthy_mean, healthy_std, NUM_SAMPLES),
            np.random.normal(deformed_mean, deformed_std, NUM_SAMPLES)
        )

    # Dictionary containing all 24 features matching your schema
    data = {
        'TIMESTAMP': timestamps,
        'MOTOR_ID': motor_ids,

        # Electrical
        'VOLTAGE': gen_sensor(400, 5, 385, 15),
        'CURRENT': gen_sensor(45, 2, 60, 8),
        'POWER': gen_sensor(25, 1, 30, 4),
        'POWER_FACTOR': gen_sensor(0.92, 0.02, 0.85, 0.05),
        'FREQUENCY': gen_sensor(50, 0.1, 49.5, 0.8),
        'VOLTAGE_IMBALANCE': gen_sensor(1.5, 0.5, 4.5, 1.5),
        'CURRENT_IMBALANCE': gen_sensor(2.0, 0.5, 6.0, 2.0),

        # Mechanical
        'SPEED': gen_sensor(1480, 10, 1420, 30),
        'TORQUE': gen_sensor(150, 5, 175, 15),
        'LOAD_PERCENT': gen_sensor(75, 5, 90, 10),

        # Vibration
        'VIBRATION_X': gen_sensor(2.5, 0.5, 8.5, 2.5),
        'VIBRATION_Y': gen_sensor(2.0, 0.4, 7.0, 2.0),
        'VIBRATION_Z': gen_sensor(1.5, 0.3, 5.5, 1.8),
        'VIBRATION_RMS': gen_sensor(3.0, 0.5, 10.5, 3.0),
        'FFT_PEAK_FREQ': gen_sensor(120, 5, 250, 40),

        # Thermal
        'BEARING_TEMP': gen_sensor(65, 3, 95, 10),
        'WINDING_TEMP': gen_sensor(80, 4, 115, 12),
        'AMBIENT_TEMP': np.random.normal(25, 3, NUM_SAMPLES),  # Ambient doesn't change based on motor health
        'TEMP_RISE': gen_sensor(40, 5, 80, 15),

        # Environment
        'HUMIDITY': np.random.normal(50, 10, NUM_SAMPLES),  # Humidity is random weather

        # Derived
        'EFFICIENCY': gen_sensor(92, 2, 82, 5),
        'CURRENT_SPIKE': gen_sensor(0, 0.1, 1, 0.5),  # 0 mostly for healthy, 1 for spikes

        # Target (Classification)
        'DEFORMITY_LABEL': labels,

        # NEW: Target (Regression) - Remaining Useful Life in Hours
        # Healthy motors have 5000 to 10000 hours left. Deformed have 0 to 1000.
        'RUL_HOURS': np.where(labels == 0,
                              np.random.uniform(5000, 10000, NUM_SAMPLES),
                              np.random.uniform(0, 1000, NUM_SAMPLES))
    }

    # Create DataFrame and round the numbers to look like real sensor data
    df = pd.DataFrame(data)
    df = df.round(3)

    # Ensure current spikes are strictly 0 or 1
    df['CURRENT_SPIKE'] = np.where(df['CURRENT_SPIKE'] > 0.5, 1, 0)

    # Save to CSV
    output_filename = "motor_data.csv"
    df.to_csv(output_filename, index=False)
    print(f"Success! {output_filename} has been created with {NUM_SAMPLES} rows.")


if __name__ == "__main__":
    generate_motor_data()