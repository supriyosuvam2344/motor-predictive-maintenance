import numpy as np
import random

# Import our loaded models and configurations
from config import (
    MODEL_FEATURES,
    FAULT_MAP,
    MAX_LIFESPAN_HOURS,
    ANOMALY_HEALTH_THRESHOLD,
    get_fault_from_health
)
from model_loader import (
    classifier_model, classifier_scaler,
    rul_model, rul_scaler, rul_explainer,
    lstm_model, lstm_scaler, lstm_encoder
)


def analyze_single_motor(motor_df):
    """
    Takes a dataframe containing a single motor's history and runs all ML predictions.
    """
    motor_features_only = motor_df[MODEL_FEATURES]

    # ==========================================
    # ANALYSIS 1: RUL Trend for the Graph
    # ==========================================
    rul_scaled_all = rul_scaler.transform(motor_features_only)
    all_predicted_ruls = rul_model.predict(rul_scaled_all)

    graph_df = motor_features_only.tail(100).copy()
    graph_ruls = all_predicted_ruls[-100:]

    graph_data = []
    prev_current = None # Used to calculate the Current Spike

    for i in range(len(graph_df)):
        row = graph_df.iloc[i]

        # 1. Calculate Vibration RMS
        vib_x = float(row['Vibration_X_mm_s'])
        vib_y = float(row['Vibration_Y_mm_s'])
        vib_z = float(row['Vibration_Z_mm_s'])
        vib_rms = np.sqrt(vib_x ** 2 + vib_y ** 2 + vib_z ** 2)

        # 2. Calculate Thermal Metrics
        winding_temp = float(row['Temperature_C']) + 15.0
        temp_rise = winding_temp - float(row['Ambient_Temp_C'])

        # --- 3. NEW: CURRENT SPIKE ---
        current_a = float(row['Current_A'])
        if prev_current is None:
            current_spike = 0.0 # First row has no past to compare to
        else:
            current_spike = current_a - prev_current
        prev_current = current_a # Save this row's current for the next loop

        # --- 4. NEW: EFFICIENCY ---
        voltage = float(row['Voltage_V'])
        power_factor = float(row['Power_Factor'])
        output_power_kw = float(row['Power_kW'])

        # Standard 3-Phase Input Power Formula: (sqrt(3) * V * I * PF) / 1000
        input_power_kw = (1.732 * voltage * current_a * power_factor) / 1000.0

        # Protect against dividing by zero if the motor is off
        if input_power_kw > 0:
            efficiency = (output_power_kw / input_power_kw) * 100.0
        else:
            efficiency = 0.0

        # Cap efficiency between 0 and 100% just to keep the graph clean
        efficiency = min(100.0, max(0.0, efficiency))
        # --------------------------------

        # Grab all original features from the CSV
        row_dict = row.to_dict()
        
        # Add all custom AI & Derived features to the payload
        row_dict["time_step"] = i + 1
        row_dict["rul_hours"] = float(graph_ruls[i])
        row_dict["vibration_rms"] = round(float(vib_rms), 3)
        row_dict["WINDING_TEMP"] = round(winding_temp, 2)
        row_dict["TEMP_RISE"] = round(temp_rise, 2)
        row_dict["CURRENT_SPIKE"] = round(current_spike, 3)
        row_dict["EFFICIENCY"] = round(efficiency, 2)

        graph_data.append(row_dict)

    # ==========================================
    # ANALYSIS 2: Current Status (The Latest Row)
    # ==========================================

    print(f"\nREAL COLUMNS ARRIVING: {motor_df.columns.tolist()}\n")

# 1. Strip any invisible Excel spaces from the headers
    motor_df.columns = motor_df.columns.str.strip() 

    # 2. Grab ONLY the exact, real timestamp from the last row
    latest_timestamp = str(motor_df['Timestamp'].iloc[-1]) 

    latest_row = motor_features_only.iloc[-1:]
    current_rul = float(all_predicted_ruls[-1])

    # Calculate Health Percentage
    motor_health_percent = min(100.0, max(0.0, (current_rul / MAX_LIFESPAN_HOURS) * 100))

    latest_scaled = classifier_scaler.transform(latest_row)
    ensemble_prediction = int(classifier_model.predict(latest_scaled)[0])

    if ensemble_prediction == 1:
        # Force a fault if the Ensemble model detects an anomaly
        probs_dict = {2: 0.70, 1: 0.30}  # Assigning high-severity faults
        current_deformity = random.choices(list(probs_dict.keys()), weights=list(probs_dict.values()))[0]
    else:
        # Fallback to RUL-based degradation logic
        probs_dict = get_fault_from_health(motor_health_percent)
        current_deformity = random.choices(list(probs_dict.keys()), weights=list(probs_dict.values()))[0]

    fault_type = FAULT_MAP.get(current_deformity, "Unknown")
    physical_status = "Healthy" if current_deformity == 0 else f"Warning: {fault_type}"

    # Determine Urgency
    if motor_health_percent < 25.0 or ensemble_prediction == 1:
        urgency = "CRITICAL: Immediate Maintenance Required"
    elif motor_health_percent < ANOMALY_HEALTH_THRESHOLD:
        urgency = "WARNING: Schedule Maintenance Soon"
    else:
        urgency = "NOMINAL: Normal Operation"

    # Calculate current RMS for the summary box
    curr_vib_x = float(latest_row['Vibration_X_mm_s'].iloc[0])
    curr_vib_y = float(latest_row['Vibration_Y_mm_s'].iloc[0])
    curr_vib_z = float(latest_row['Vibration_Z_mm_s'].iloc[0])
    current_vib_rms = np.sqrt(curr_vib_x ** 2 + curr_vib_y ** 2 + curr_vib_z ** 2)

    # ==========================================
    # ANALYSIS 3: SHAP Explanations
    # ==========================================
    rul_scaled_latest = rul_scaler.transform(latest_row)
    shap_values = rul_explainer.shap_values(rul_scaled_latest)

    feature_impacts = []
    for i, name in enumerate(MODEL_FEATURES):
        feature_impacts.append({
            "feature": name,
            "impact": float(shap_values[0][i])
        })

    feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
    top_3_impacts = feature_impacts[:3]

    explanations = []
    for item in top_3_impacts:
        direction = "decreased" if item["impact"] < 0 else "increased"
        explanations.append(
            f"{item['feature']} {direction} the estimated lifespan by {abs(item['impact']):.1f} hours."
        )

    # ==========================================
    # ANALYSIS 4: LSTM Time-Series Deep Learning
    # ==========================================
    lstm_status = "Not enough data for Deep Learning trend."
    SEQUENCE_LENGTH = 30

    if len(motor_df) >= SEQUENCE_LENGTH:
        lstm_buffer = motor_features_only.tail(SEQUENCE_LENGTH)
        buffer_scaled = lstm_scaler.transform(lstm_buffer)

        lstm_input = np.expand_dims(buffer_scaled, axis=0)

        lstm_pred_prob = lstm_model.predict(lstm_input, verbose=0)
        lstm_pred_class = np.argmax(lstm_pred_prob, axis=1)
        lstm_decoded_label = lstm_encoder.inverse_transform(lstm_pred_class)[0]

        if lstm_decoded_label == 0:
            lstm_status = "Deep Learning Status: Healthy historical trend."
        else:
            lstm_fault = FAULT_MAP.get(lstm_decoded_label, f"Unknown (Code: {lstm_decoded_label})")
            lstm_status = f"Deep Learning Status: DEGRADATION TREND DETECTED - Probable {lstm_fault}"

    # Return the dictionary for this specific motor
    return {
        "summary": {
            "total_rows_analyzed": len(motor_df),
            "latest_timestamp": latest_timestamp,
            "physical_status": physical_status,
            "fault_code": current_deformity,
            "fault_type": fault_type,
           "current_rul_hours": round(float(current_rul), 2),
            "current_vibration_rms": round(current_vib_rms, 3),  # Added RMS to summary
            "maintenance_urgency": urgency
        },
        "lstm_time_series_analysis": lstm_status,
        "ai_explanations": explanations,
        "graph_data": graph_data
    }
