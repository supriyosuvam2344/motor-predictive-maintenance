import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_motor_data():
    NUM_SAMPLES_PER_MOTOR = 120000
    START_TIME = datetime(2025, 1, 1)

    # 1. Use a dynamic timestamp to guarantee a unique filename every single run.
    # This ensures you NEVER rewrite or delete older datasets.
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"trial_motor_data{timestamp_str}.csv"

    # 2. Define unique physical profiles for the 5 motors
    # Each motor has distinctly different healthy (h) and damaged (d) baselines
    motor_profiles = {
        'M1': {'type': 'Perfectly Healthy', 'temp_h': 55.0, 'temp_d': 85.0, 'vib_h': 0.5, 'vib_d': 2.5,
               'p_fault': 0.001},
        'M2': {'type': 'Severe Overheating', 'temp_h': 95.0, 'temp_d': 120.0, 'vib_h': 0.7, 'vib_d': 2.8,
               'p_fault': 0.05},
        'M3': {'type': 'Rotor Imbalance', 'temp_h': 62.0, 'temp_d': 85.0, 'vib_h': 6.5, 'vib_d': 9.5, 'p_fault': 0.04},
        'M4': {'type': 'Critical Bearing', 'temp_h': 92.0, 'temp_d': 115.0, 'vib_h': 7.5, 'vib_d': 11.0,
               'p_fault': 0.08},
        'M5': {'type': 'Moderate Wear', 'temp_h': 72.0, 'temp_d': 95.0, 'vib_h': 3.2, 'vib_d': 5.5, 'p_fault': 0.02}
    }

    print(f"Starting generation of {len(motor_profiles) * NUM_SAMPLES_PER_MOTOR} rows...")
    print(f"New file will be saved as: {output_filename}")

    for motor_id, profile in motor_profiles.items():
        print(f"Generating data for {motor_id} ({profile['type']})...")

        timestamps = [
            (START_TIME + timedelta(seconds=i * 10)).strftime('%d-%m-%Y %H:%M')
            for i in range(NUM_SAMPLES_PER_MOTOR)
        ]
        status = np.random.choice([0, 1], size=NUM_SAMPLES_PER_MOTOR, p=[0.02, 0.98])

        # Inject anomalies based on this specific motor's fault probability
        labels = np.random.choice([0, 1], size=NUM_SAMPLES_PER_MOTOR, p=[1 - profile['p_fault'], profile['p_fault']])

        # Updated logic: min_idle adds a small non-zero baseline when motor is OFF
        def gen_sensor(h_mean, h_std, d_mean, d_std, min_val=0.1, min_idle=0.5):
            # Normal distribution based on health state
            base = np.where(labels == 0,
                            np.random.normal(h_mean, h_std, NUM_SAMPLES_PER_MOTOR),
                            np.random.normal(d_mean, d_std, NUM_SAMPLES_PER_MOTOR))

            # If status == 1, use the calculated base.
            # If status == 0, use a small random standby value instead of 0.0
            standby = np.random.uniform(min_idle * 0.8, min_idle * 1.2, NUM_SAMPLES_PER_MOTOR)
            final_data = np.where(status == 1, base, standby)

            return np.clip(final_data, min_val, None)

        # 3. Build DataFrame feeding in the unique motor profile parameters
        df_motor = pd.DataFrame({
            'Timestamp': timestamps,
            'Motor_ID': motor_id,
            'Temperature_C': gen_sensor(profile['temp_h'], 5.0, profile['temp_d'], 15.0, min_val=20.0, min_idle=25.0),
            'Bearing_Temp_C': gen_sensor(profile['temp_h'] + 5.0, 5.0, profile['temp_d'] + 10.0, 20.0, min_val=20.0,
                                         min_idle=25.0),
            'Vibration_X_mm_s': gen_sensor(profile['vib_h'], 0.3, profile['vib_d'], 3.0, min_val=0.1, min_idle=0.2),
            'Vibration_Y_mm_s': gen_sensor(profile['vib_h'] + 0.05, 0.3, profile['vib_d'] + 0.5, 3.0, min_val=0.1,
                                           min_idle=0.2),
            'Vibration_Z_mm_s': gen_sensor(profile['vib_h'] + 0.1, 0.3, profile['vib_d'] + 1.0, 3.0, min_val=0.1,
                                           min_idle=0.2),
            'Current_A': gen_sensor(11.6, 1.0, 35.0, 5.0, min_val=0.5, min_idle=0.5),
            'Voltage_V': np.random.normal(415, 10.0, NUM_SAMPLES_PER_MOTOR),
            'Power_kW': gen_sensor(4.3, 0.5, 12.0, 2.0, min_val=0.1, min_idle=0.2),
            'Power_Factor': gen_sensor(0.89, 0.05, 0.60, 0.1, min_val=0.1, min_idle=0.2),
            'RPM': gen_sensor(1476.0, 5.0, 1300.0, 50.0, min_val=10.0, min_idle=10.0),
            'Load_Percent': gen_sensor(70.5, 5.0, 98.0, 10.0, min_val=1.0, min_idle=1.0),
            'Ambient_Temp_C': np.random.normal(28.0, 8.0, NUM_SAMPLES_PER_MOTOR),
            'Running_Status': status,
            'Anomaly': labels,
            'Maintenance_Event': np.where(labels == 1, 'Repair Needed', 'None')
        })

        # 4. Safe Appending: Only write the header if the file doesn't exist yet!
        write_header = not os.path.exists(output_filename)
        df_motor.to_csv(output_filename, mode='a', index=False, header=write_header)

    print(f"\nSuccess! Data saved to: {os.path.abspath(output_filename)}")


if __name__ == "__main__":
    generate_motor_data()