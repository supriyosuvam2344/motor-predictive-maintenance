---
title: Motor Predictive API
emoji: ⚙️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
 

# ⚙️ Motor Predictive Maintenance Platform

An end-to-end, full-stack predictive maintenance system designed to analyze high-frequency industrial motor telemetry.

By leveraging a **hybrid AI architecture combining XGBoost and LSTM**, the platform classifies real-time structural anomalies and predicts the **Remaining Useful Life (RUL)** of critical equipment, helping prevent unplanned downtime and enabling proactive maintenance.

---

## 📂 Project Structure

The project is decoupled into three main architectural components:

```text
motor-predictive-maintenance/
│
├── frontend/
│   └── React.js dashboard for visualizing equipment health,
│       remaining lifespan, and sensor trends
│
├── backend/
│   └── FastAPI server for data ingestion, preprocessing,
│       and real-time AI inference
│
└── ml/
    └── Machine learning pipeline containing XGBoost and LSTM
        training scripts and synthetic industrial data generators
```

### `/frontend`

The React.js dashboard visualizes:

* Equipment health
* Remaining Useful Life (RUL)
* Sensor trends
* Fault predictions
* Motor performance metrics

### `/backend`

The FastAPI server handles:

* Data ingestion
* Sensor data preprocessing
* Real-time inference
* ML model loading
* API communication with the frontend

### `/ml`

The machine learning pipeline contains:

* Synthetic motor data generation
* XGBoost model training
* LSTM time-series training
* Model evaluation
* Model export for backend inference

---

## 🌟 Key Features

### 🔍 Real-Time Anomaly Detection

Uses an ensemble **XGBoost classifier** to identify immediate physical faults from motor sensor data, including:

* Bearing Failure
* Rotor Imbalance
* Misalignment
* Overheating
* Other structural anomalies

---

### ⏳ Remaining Useful Life (RUL) Forecasting

Combines:

* **LSTM time-series networks**
* **XGBoost regression models**

to estimate the remaining operational lifespan of the motor.

---

### 🧠 Explainable AI (XAI)

Integrates **SHAP (SHapley Additive exPlanations)** to quantify how individual sensor metrics influence the estimated Remaining Useful Life.

This helps operators understand:

* Which sensor values are driving predictions
* Why the predicted RUL is increasing or decreasing
* Which motor parameters require immediate attention

---

### 🧪 Synthetic Data Generation

Includes custom Python scripts for generating realistic motor telemetry datasets for:

* Model training
* Model testing
* Fault simulation
* RUL degradation analysis
* Dashboard demonstrations

---

## 🖥️ Frontend Dashboard Highlights

The user interface was built with a focus on translating complex industrial data into actionable, operator-friendly metrics, featuring a custom-built industrial dark theme.

### Multi-Motor Tracking

The application automatically detects multiple motors within a single dataset upload, such as `M1`, `M2`, and `M3`, and provides seamless tabbed navigation to switch between their independent diagnostic reports.

### Intelligent Status Cards

A prominent alert system displays:

* Immediate physical status, such as `WARNING: ROTOR IMBALANCE`
* Color-coded maintenance urgency
* Highly precise RUL countdown in **Days, Hours, and Minutes**

### Integrated SHAP AI Insights

A dedicated UI panel translates complex SHAP values into readable bullet points, explaining exactly which sensor deviations, such as `Vibration_Y_mm_s`, are actively decreasing the motor's lifespan.

### Comprehensive Sensor Telemetry

The dashboard renders a vertically stacked suite of high-contrast line charts tracking every aspect of motor health over time.

#### Electrical Metrics

* Voltage (V)
* Current (A)
* Power (kW)
* Power Factor

#### Vibration Analysis

* 3-Axis Structural Vibration (X, Y, Z)
* Overall Vibration RMS

#### Thermal & Operational Monitoring

* Temperature (°C)
* Bearing Temperature
* RPM
* Load Percentage

---

## 🛠️ Technology Stack

### 1. Frontend: UI & Visualization

* React.js (Vite)
* Tailwind CSS / Custom Dark Theme
* React Charting Library

### 2. Backend: API & Inference

* FastAPI
* Uvicorn
* Python
* Pandas
* NumPy
* Joblib

### 3. Machine Learning & Deep Learning

* XGBoost
* Scikit-learn
* TensorFlow / Keras
* LSTM Time-Series Networks
* SHAP

---

## 🚀 Getting Started

### Prerequisites

Make sure the following tools are installed:

* Python 3.9+
* Node.js
* npm
* Git

---

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/motor-predictive-maintenance.git
cd motor-predictive-maintenance
```

> Replace `yourusername` with your actual GitHub username.

---

### 2. Set Up a Python Virtual Environment

Before running the machine learning scripts or the FastAPI backend, it is recommended to create an isolated Python virtual environment to manage dependencies.

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS and Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Once activated, install the required dependencies:

```bash
pip install -r backend/requirements.txt
```

---

### 3. Machine Learning Setup

This step is optional if pretrained models are already available.

Ensure your virtual environment is activated.

Navigate to the dataset directory and generate fresh synthetic motor data:

```bash
cd dataset
python generate_dataset.py
```

Next, navigate to the ML directory to train the models:


```bash
cd ml
```


Train the XGBoost models:

```bash
python train_models.py
```

Train the LSTM model:

```bash
python train_lstm.py
```

The newly trained `.pkl` and `.keras` model files will automatically be saved to the backend models directory.

---

### 4. Backend Setup

Ensure your virtual environment is activated.

Navigate to the backend directory:

```bash
# If you are currently inside the /ml directory
cd ../backend
```

Or, from the project root:

```bash
cd backend
```

Start the FastAPI development server:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

---

### 5. Frontend Setup

Open a new terminal and navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the Vite development server:

```bash
npm run dev
```

The React application will be available at:

```text
http://localhost:5173
```

---

## 🐳 Deployment Architecture

The project uses a decoupled deployment strategy.

### Frontend: Vercel

The React dashboard is deployed on Vercel for:

* Fast global content delivery
* Continuous deployment
* Automated builds
* Easy frontend hosting

### Backend & ML Inference: Hugging Face Spaces

The FastAPI machine learning backend is:

* Containerized using Docker
* Hosted on Hugging Face Spaces
* Configured to run on port `7860`

### Deployment Steps

1. Create a new Docker Space on Hugging Face.
2. Push the backend code and trained models to the Space.
3. The Dockerfile installs all Python dependencies.
4. The application loads the trained ML models.
5. FastAPI starts and exposes the inference API on port `7860`.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────┐
│       Motor Telemetry       │
│  Vibration | Temp | RPM     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│        FastAPI Backend      │
│  Preprocessing & Validation │
└──────────────┬──────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
┌─────────────┐ ┌─────────────┐
│   XGBoost   │ │    LSTM     │
│ Fault + RUL │ │ Time-Series │
└──────┬──────┘ └──────┬──────┘
       │               │
       └───────┬───────┘
               ▼
┌─────────────────────────────┐
│     Prediction & XAI        │
│  Fault | RUL | SHAP Values  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       React Dashboard       │
│ Charts | Health | Insights  │
└─────────────────────────────┘
```

---

## 🎯 Project Objective

This project was developed as a comprehensive solution for:

* Industrial structural deformity prediction
* Predictive maintenance
* Equipment health monitoring
* Remaining Useful Life estimation
* AI-powered fault diagnosis
* Explainable industrial machine learning

The platform demonstrates the integration of **full-stack development, machine learning, deep learning, explainable AI, real-time APIs, and cloud deployment** in a unified industrial predictive maintenance system.

---

## 📄 Live Dashboard

🚀 **Dashboard:** https://motor-predictive-maintenance.vercel.app

---

## 📌 Future Improvements

* Live IoT sensor integration
* WebSocket-based real-time telemetry streaming
* Automated maintenance scheduling
* Historical motor health comparison
* Multi-factory equipment monitoring
* Cloud database integration
* Advanced alert and notification systems

---

## 📜 License

This project is developed for educational, research, and industrial predictive maintenance experimentation purposes.
