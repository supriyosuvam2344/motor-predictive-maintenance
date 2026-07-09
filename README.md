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

By leveraging a hybrid AI architecture combining **XGBoost** and **LSTM**, the platform classifies real-time structural anomalies and predicts the **Remaining Useful Life (RUL)** of critical equipment, helping prevent unplanned downtime and enabling proactive maintenance.

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

The **React.js dashboard** visualizes:

* Equipment health
* Remaining useful life
* Sensor trends
* Fault predictions
* Motor performance metrics

### `/backend`

The **FastAPI server** handles:

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

### ⏳ Remaining Useful Life (RUL) Forecasting

Combines:

* **LSTM time-series networks**
* **XGBoost regression models**

to estimate the remaining operational lifespan of the motor.

### 🧠 Explainable AI (XAI)

Integrates **SHAP (SHapley Additive exPlanations)** to quantify how individual sensor metrics influence the estimated Remaining Useful Life.

This helps operators understand:

* Which sensor values are driving predictions
* Why the predicted RUL is increasing or decreasing
* Which motor parameters require immediate attention

### 🧪 Synthetic Data Generation

Includes custom Python scripts for generating realistic motor telemetry datasets for:

* Model training
* Model testing
* Fault simulation
* RUL degradation analysis
* Dashboard demonstrations

---

## 🖥️ Frontend Dashboard Highlights

The user interface is designed to translate complex industrial sensor data into actionable, operator-friendly insights.

### 📊 Interactive Data Visualization

Powered by **Recharts**, the dashboard dynamically renders:

#### RUL Degradation Trend

An area chart tracking the predicted remaining lifespan of the motor over time.

#### Vibration vs. Temperature

A dual-axis line chart visualizing correlations between motor vibration and operating temperature.

#### Motor Speed Stability

A step-line chart monitoring RPM fluctuations and speed instability.

#### Vibration Intensity Profile

A bar chart highlighting vibration spikes that may indicate developing structural faults.

### 🛡️ Resilient UI — Demo Mode

The frontend includes an automatic fallback mechanism.

If the live FastAPI backend is unreachable, the dashboard automatically loads high-quality simulated data so the complete interface remains functional for:

* Portfolio demonstrations
* UI testing
* Offline development
* Deployment previews

---

## 🛠️ Technology Stack

### 1. Frontend — UI & Visualization

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| React.js     | Frontend framework             |
| Vite         | Development and build tool     |
| Tailwind CSS | UI styling                     |
| Recharts     | Interactive data visualization |
| Lucide React | Icons                          |

### 2. Backend — API & Inference

| Technology | Purpose               |
| ---------- | --------------------- |
| FastAPI    | REST API framework    |
| Uvicorn    | ASGI server           |
| Python     | Backend development   |
| Pandas     | Data processing       |
| NumPy      | Numerical computation |
| Joblib     | ML model loading      |

### 3. Machine Learning & Deep Learning

| Technology         | Purpose                                 |
| ------------------ | --------------------------------------- |
| XGBoost            | Fault classification and RUL regression |
| Scikit-learn       | Preprocessing and ML utilities          |
| TensorFlow / Keras | LSTM time-series analysis               |
| SHAP               | Model interpretability                  |

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

### 2. Machine Learning Setup

This step is optional if pretrained models are already available.

Navigate to the ML directory:

```bash
cd ml
```

Generate fresh synthetic motor data:

```bash
python generate_motor_data_wide.py
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

### 3. Backend Setup

Navigate to the backend directory:

```bash
cd ../backend
```

Install the required dependencies:

```bash
pip install -r requirements.txt
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

### 4. Frontend Setup

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

### Frontend — Vercel

The React dashboard is deployed on **Vercel** for:

* Fast global content delivery
* Continuous deployment
* Automated builds
* Easy frontend hosting

### Backend & ML Inference — Hugging Face Spaces

The FastAPI machine learning backend is:

* Containerized using Docker
* Hosted on Hugging Face Spaces
* Configured to run on port `7860`

### Deployment Steps

1. Create a new **Docker Space** on Hugging Face.
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

## 📄 Dashboard Link :

https://motor-predictive-maintenance.vercel.app
