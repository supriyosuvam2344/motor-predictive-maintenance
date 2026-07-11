import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Commented out the fixed seed so that it generates uniquely randomized data every time!
# np.random.seed(42)

NUM_SAMPLES = 600000


def generate_motor_data():
    print(f"Generating {NUM_SAMPLES} rows of motor data")

    labels = np.random.choice([0, 1], size=NUM_SAMPLES, p=[0.99, 0.01])

    # Changed to 6 data points per minute (1 point every 10 seconds)
    timestamps = [datetime(2025, 1, 1) + timedelta(seconds=i * 10) for i in range(NUM_SAMPLES)]

    motor_ids = np.random.choice(['M1', 'M2', 'M3'], size=NUM_SAMPLES)

    # INCREASED NOISE: Makes the range wider and the signal noisier
    NOISE_MULTIPLIER = 5.0

    # SENSORS: Widened the difference between h_mean (healthy) and d_mean (faulty)
    def gen_sensor(h_mean, h_std, d_mean, d_std):
        return np.where(
            labels == 0,
            np.random.normal(h_mean, h_std * NOISE_MULTIPLIER, NUM_SAMPLES),
            np.random.normal(d_mean, d_std * NOISE_MULTIPLIER, NUM_SAMPLES)
        )

    # Data construction with significantly wider variance for faulty states
    data = {
        'Timestamp': timestamps,
        'Motor_ID': motor_ids,
        'Temperature_C': gen_sensor(62.0, 5.0, 110.0, 15.0),  # Range 62 -> 110
        'Bearing_Temp_C': gen_sensor(64.0, 5.0, 120.0, 20.0),  # Range 64 -> 120
        'Vibration_X_mm_s': gen_sensor(0.65, 0.3, 8.0, 3.0),  # Range 0.65 -> 8
        'Vibration_Y_mm_s': gen_sensor(0.70, 0.3, 8.5, 3.0),
        'Vibration_Z_mm_s': gen_sensor(0.75, 0.3, 9.0, 3.0),
        'Current_A': gen_sensor(11.6, 1.0, 35.0, 5.0),  # Range 11 -> 35
        'Voltage_V': np.random.normal(415, 10.0, NUM_SAMPLES),  # Wider voltage drift
        'Power_kW': gen_sensor(4.3, 0.5, 12.0, 2.0),
        'Power_Factor': gen_sensor(0.89, 0.05, 0.60, 0.1),
        'RPM': gen_sensor(1476.0, 5.0, 1300.0, 50.0),  # Range 1476 -> 1300
        'Load_Percent': gen_sensor(70.5, 5.0, 98.0, 10.0),
        'Ambient_Temp_C': np.random.normal(28.0, 8.0, NUM_SAMPLES),
        'Running_Status': np.random.choice([0, 1], p=[0.009, 0.991], size=NUM_SAMPLES),
        'Anomaly': labels,
        'Maintenance_Event': np.where(labels == 1, 'Repair Needed', 'None')
    }

    df = pd.DataFrame(data)

    # Generate a unique timestamp string for the filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the unique file path
    output_filename = f"../dataset/trial_motor_data{current_time}.csv"

    # Rounding and saving
    df.to_csv(output_filename, index=False)
    print(f"Success! Data saved to: {output_filename}")


if __name__ == "__main__":
    generate_motor_data()