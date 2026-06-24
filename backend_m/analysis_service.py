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
    for i in range(len(graph_df)):
        row = graph_df.iloc[i]

        # Calculate Vibration RMS on the fly using the X, Y, Z axes
        vib_x = float(row['Vibration_X_mm_s'])
        vib_y = float(row['Vibration_Y_mm_s'])
        vib_z = float(row['Vibration_Z_mm_s'])
        vib_rms = np.sqrt(vib_x ** 2 + vib_y ** 2 + vib_z ** 2)

        graph_data.append({
            "time_step": i + 1,
            "rul_hours": float(graph_ruls[i]),
            "vibration_rms": round(float(vib_rms), 3),  # Added RMS for React Graphs
            "temperature": float(row['Temperature_C']),
            "rpm": float(row['RPM'])
        })

    # ==========================================
    # ANALYSIS 2: Current Status (The Latest Row)
    # ==========================================
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
            "physical_status": physical_status,
            "fault_code": current_deformity,
            "fault_type": fault_type,
            "current_rul_hours": f"{round(current_rul, 1)} hours ({round(current_rul / 24, 1)} days)",
            "current_vibration_rms": round(current_vib_rms, 3),  # Added RMS to summary
            "maintenance_urgency": urgency
        },
        "lstm_time_series_analysis": lstm_status,
        "ai_explanations": explanations,
        "graph data": graph_data
    }