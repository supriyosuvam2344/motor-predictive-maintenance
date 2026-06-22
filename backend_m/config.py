# ==========================================
# CONSTANTS & CONFIGURATIONS
# ==========================================

# The exact 12 features required by your newly trained models
MODEL_FEATURES = [
        'Temperature_C', 'Bearing_Temp_C', 'Vibration_X_mm_s', 'Vibration_Y_mm_s',
        'Vibration_Z_mm_s', 'Current_A', 'Voltage_V', 'Power_kW', 'Power_Factor',
        'RPM', 'Load_Percent', 'Ambient_Temp_C'
    ]

# ==========================================
# HEALTH & ANOMALY THRESHOLDS
# ==========================================
# Assumed maximum lifespan of a brand-new motor in hours
MAX_LIFESPAN_HOURS = 50000.0

# If the motor health percentage drops below this number, trigger an Anomaly (1)
ANOMALY_HEALTH_THRESHOLD = 90.0

# Fault type dictionary for plain-english translations
FAULT_MAP = {
    0: "Healthy",
    1: "Overheating",
    2: "Shaft Misalignment",
    3: "Rotor Imbalance",
    4: "Winding Short Circuit",
    5: "Bearing Failure",
    6: "Residual electrical degradation"
}

def get_fault_from_health(health_percent):
    """Maps a degrading health percentage to a specific fault stage."""
    if health_percent >= 85:
        probs = {
            0: 0.90,  # Healthy
            1: 0.10  # Overheating
        }

    elif health_percent>= 70:
        probs = {
            1: 0.70,  # Overheating
            2: 0.30  # Shaft Misalignment
        }

    elif health_percent>= 55:
        probs = {
            2: 0.60,  # Shaft Misalignment
            3: 0.40  # Rotor Imbalance
        }

    elif health_percent>= 35:
        probs = {
            3: 0.65,  # Rotor Imbalance
            4: 0.35  # Winding Short Circuit
        }

    elif health_percent>= 15:
        probs = {
            4: 0.70,  # Winding Short Circuit
            5: 0.30  # Bearing Failure
        }

    else:
        probs = {
            5: 0.90,  # Bearing Failure
            6: 0.10  # Residual electrical degradation
        }

    return probs