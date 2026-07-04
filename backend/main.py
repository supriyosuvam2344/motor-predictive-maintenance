from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

# Import data from our custom modular files
from config import MODEL_FEATURES
from analysis_service import analyze_single_motor

# Note: Simply importing ml_models triggers the script to load them into memory on startup
import model_loader

app = FastAPI(title="Motor Predictive Maintenance API - Modular Version")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Modular Upload API is running. Send a CSV/Excel to /analyze_dataset"}


@app.post("/analyze_dataset")
async def analyze_motor_dataset(file: UploadFile = File(...)):
    try:
        # 1. Read the uploaded file into Pandas
        contents = await file.read()
        filename = file.filename.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV or Excel file.")

        # ==========================================
        # PRE-PROCESSING: Standardize Motor ID Column
        # ==========================================
        # Find the motor id column regardless of case (Motor_ID, motor_id, MOTOR_ID)
        motor_id_col = next((col for col in df.columns if col.upper() == 'MOTOR_ID'), None)

        if motor_id_col:
            # Standardize the name to exactly what the rest of the script expects
            df.rename(columns={motor_id_col: 'MOTOR_ID'}, inplace=True)
        else:
            # If no motor ID is found at all, treat the whole file as one motor
            df['MOTOR_ID'] = 'Motor_1'

        # 2. Validate columns
        missing_cols = [col for col in MODEL_FEATURES if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset is missing required columns expected by the ML models: {missing_cols}. Please map your CSV columns or retrain the models."
            )

        # Filter down to just the features we need + the ID + Timestamp!
        columns_to_keep = MODEL_FEATURES + ['MOTOR_ID']
        
        # Save the Timestamp from being deleted
        if 'Timestamp' in df.columns:
            columns_to_keep.append('Timestamp')
            
        df_clean = df[columns_to_keep].dropna().reset_index(drop=True)

        if len(df_clean) == 0:
            raise HTTPException(status_code=400, detail="Dataset is empty after removing missing values.")

        # ==========================================
        # GROUP BY MOTOR ID AND CALL OUR SERVICE
        # ==========================================
        motors_report = {}
        unique_motors = df_clean['MOTOR_ID'].unique()

        for motor_id in unique_motors:
            motor_df = df_clean[df_clean['MOTOR_ID'] == motor_id].copy()

            # Send the dataframe to our analysis service file and get the results
            motor_report = analyze_single_motor(motor_df)
            motors_report[str(motor_id)] = motor_report

        return {
            "available_motors": [str(m) for m in unique_motors],
            "motor_data": motors_report
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))