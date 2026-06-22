import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed so the data is reproducible
np.random.seed(42)

NUM_SAMPLES = 300000  # Adjusted for speed, but you can increase this!


def generate_motor_data():
    print(f"Generating {NUM_SAMPLES} rows of REALISTIC motor telemetry based on your dataset...")

    # Target Label: 95% Healthy (0), 5% Anomaly (1)
    labels = np.random.choice([0, 1], size=NUM_SAMPLES, p=[0.99, 0.01])

    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i * 15) for i in range(NUM_SAMPLES)]
    motor_ids = np.random.choice(['M1', 'M2', 'M3', 'M4', 'M5'], size=NUM_SAMPLES)

    # Noise multiplier to create messy "grey areas" where anomalies are hard to spot
    NOISE_MULTIPLIER = 2.0

    def gen_sensor(h_mean, h_std, d_mean, d_std):
        return np.where(
            labels == 0,
            np.random.normal(h_mean, h_std * NOISE_MULTIPLIER, NUM_SAMPLES),
            np.random.normal(d_mean, d_std * NOISE_MULTIPLIER, NUM_SAMPLES)
        )

    anomaly_labels = labels
    maintenance = np.where(anomaly_labels == 1,
                           np.random.choice(['None', 'Repair Needed'], p=[0.2, 0.8], size=NUM_SAMPLES),
                           'None')

    # Building the data using baseline values directly from your image screenshot
    data =  {
        'Timestamp': timestamps,
        'Motor_ID': motor_ids,
        'Temperature': gen_sensor(62.0, 2.5, 85.0, 8.0),
        'Bearing_Temperature': gen_sensor(64.0, 3.0, 90.0, 10.0),
        'Vibration_X': gen_sensor(0.65, 0.15, 3.5, 1.2),
        'Vibration_Y': gen_sensor(0.70, 0.15, 3.8, 1.5),
        'Vibration_Z': gen_sensor(0.75, 0.15, 4.0, 1.8),
        'Current_A': gen_sensor(11.6, 0.5, 18.5, 2.5),
        'Voltage_V': np.random.normal(415, 1.5, NUM_SAMPLES),
        'Power_kW': gen_sensor(4.3, 0.3, 6.5, 0.8),
        'Power_Factor': gen_sensor(0.89, 0.02, 0.82, 0.04),
        'RPM': gen_sensor(1476.0, 2.0, 1440.0, 15.0),
        'Load_Perc': gen_sensor(70.5, 3.0, 92.0, 6.0),
        'Ambient_Temp': np.random.normal(28.0, 3.0, NUM_SAMPLES),
        'Running_Status': np.random.choice([0, 1], p=[0.009, 0.991], size=NUM_SAMPLES),
        'Anomaly': anomaly_labels,
        'Maintenance_Event': maintenance
    }

    df = pd.DataFrame(data)

    # Introduce Label Noise to confuse the AI slightly (Edge Cases)
    # We flip 3% of the labels so the model can't achieve a perfect 100% accuracy
    num_to_flip = int(NUM_SAMPLES * 0.03)
    flip_indices = np.random.choice(df.index, num_to_flip, replace=False)
    df.loc[flip_indices, 'Anomaly'] = 1 - df.loc[flip_indices, 'Anomaly']

    # Round numerical columns for a clean CSV
    numerical_cols = ['Temperature', 'Bearing_Temperature', 'Vibration_X', 'Vibration_Y',
                      'Vibration_Z', 'Current_A', 'Voltage_V', 'Power_kW', 'Power_Factor',
                      'RPM', 'Load_Perc', 'Ambient_Temp']
    df[numerical_cols] = df[numerical_cols].round(4)

    # Save to CSV
    output_filename = "../dataset_m/motor_data1.csv"
    df.to_csv(output_filename, index=False)
    print(f"Success! '{output_filename}' created perfectly matching your new dataset structure.")


if __name__ == "__main__":
    generate_motor_data()