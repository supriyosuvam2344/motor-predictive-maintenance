import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_motor_data():
    NUM_SAMPLES_PER_MOTOR = 120000
    MOTORS = ['M1', 'M2', 'M3', 'M4', 'M5']
    START_TIME = datetime(2024, 1, 1)

    all_motor_data = []

    print(f"Generating {len(MOTORS) * NUM_SAMPLES_PER_MOTOR} rows of motor data...")

    for motor_id in MOTORS:
        # 1. Generate timestamps and status for this specific motor
        timestamps = [
            (START_TIME + timedelta(seconds=i * 10)).strftime('%d-%m-%Y %H:%M')
            for i in range(NUM_SAMPLES_PER_MOTOR)
        ]
        status = np.random.choice([0, 1], size=NUM_SAMPLES_PER_MOTOR, p=[0.02, 0.98])
        labels = np.random.choice([0, 1], size=NUM_SAMPLES_PER_MOTOR, p=[0.99, 0.01])

        # 2. Define Helper for Sensor Generation
        def gen_sensor(h_mean, h_std, d_mean, d_std, min_val=0.0):
            # Normal distribution based on health state
            base = np.where(labels == 0,
                            np.random.normal(h_mean, h_std, NUM_SAMPLES_PER_MOTOR),
                            np.random.normal(d_mean, d_std, NUM_SAMPLES_PER_MOTOR))
            # Force 0 if motor is OFF
            active_data = np.where(status == 1, base, 0.0)
            return np.clip(active_data, min_val, None)

        # 3. Build DataFrame for this motor
        df_motor = pd.DataFrame({
        'Timestamp': timestamps,
        'Motor_ID': motor_id,
        'Temperature_C': gen_sensor(62.0, 5.0, 110.0, 15.0),  # Range 62 -> 110
        'Bearing_Temp_C': gen_sensor(64.0, 5.0, 120.0, 20.0),  # Range 64 -> 120
        'Vibration_X_mm_s': gen_sensor(0.65, 0.3, 8.0, 3.0),  # Range 0.65 -> 8
        'Vibration_Y_mm_s': gen_sensor(0.70, 0.3, 8.5, 3.0),
        'Vibration_Z_mm_s': gen_sensor(0.75, 0.3, 9.0, 3.0),
        'Current_A': gen_sensor(11.6, 1.0, 35.0, 5.0),  # Range 11 -> 35
        'Voltage_V': np.random.normal(415, 10.0, NUM_SAMPLES_PER_MOTOR),  # Wider voltage drift
        'Power_kW': gen_sensor(4.3, 0.5, 12.0, 2.0),
        'Power_Factor': gen_sensor(0.89, 0.05, 0.60, 0.1),
        'RPM': gen_sensor(1476.0, 5.0, 1300.0, 50.0),  # Range 1476 -> 1300
        'Load_Percent': gen_sensor(70.5, 5.0, 98.0, 10.0),
        'Ambient_Temp_C': np.random.normal(28.0, 8.0, NUM_SAMPLES_PER_MOTOR),
        'Running_Status': np.random.choice([0, 1], p=[0.009, 0.991], size=NUM_SAMPLES_PER_MOTOR),
        'Anomaly': labels,
        'Maintenance_Event': np.where(labels == 1, 'Repair Needed', 'None')
        })
        all_motor_data.append(df_motor)

    # 4. Combine and Sort
    full_df = pd.concat(all_motor_data, ignore_index=True)
    full_df = full_df.sort_values(by=['Motor_ID', 'Timestamp']).reset_index(drop=True)

    # 5. Save
    output_filename = f"trial_motor_data{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    full_df.to_csv(output_filename, index=False)

    print(f"Success! Data saved to: {output_filename}")
    print(f"Data shape: {full_df.shape}")


if __name__ == "__main__":
    generate_motor_data()