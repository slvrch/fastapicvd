import os
import json
import requests
import gdown
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from joblib import load
import pandas as pd
import numpy as np
from supabase_client import supabase
from datetime import datetime, timezone


app = FastAPI()

def download_model_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(url, dest_path, quiet=False)

def load_model_presence():
    model_path = "modeling/model_presence.joblib"
    if not os.path.exists(model_path):
        print("Mengunduh model dari GDrive...")
        file_id = "1nWUhcG4Uyotk_zbvi_LfnchPcCKgAp9f"
        download_model_from_gdrive(file_id, model_path)
    else:
        print("Model presence sudah tersedia.")

    return load(model_path)

# Load model
model_presence = load_model_presence()
model_risk = load("modeling/model_risk.joblib")

# Load encoders
result_target_presence = load("modeling/encoder_presence_target.joblib")
encoder_Hypertension_presence = load("modeling/encoder_presence_Hypertension.joblib")
encoder_ECG_Abnormality_presence = load("modeling/encoder_presence_ECG_Abnormality.joblib")
encoder_Diabetes_presence = load("modeling/encoder_presence_Diabetes.joblib")
encoder_Alcohol_presence = load("modeling/encoder_presence_Alcohol.joblib")
encoder_Previous_Stroke_presence = load("modeling/encoder_presence_Previous_Stroke.joblib")
encoder_Family_History_presence = load("modeling/encoder_presence_Family_History.joblib")
encoder_CVD_Risk_Score_presence = load("modeling/encoder_presence_CVD_Risk_Score.joblib")

result_target_risk = load("modeling/encoder_target_risk.joblib")
encoder_Hypertension_risk = load("modeling/encoder_risk_Hypertension.joblib")
encoder_ECG_Abnormality_risk = load("modeling/encoder_risk_ECG_Abnormality.joblib")
encoder_Diabetes_risk = load("modeling/encoder_risk_Diabetes.joblib")
encoder_Alcohol_risk = load("modeling/encoder_risk_Alcohol.joblib")
encoder_Previous_Stroke_risk = load("modeling/encoder_risk_Previous_Stroke.joblib")
encoder_Family_History_risk = load("modeling/encoder_risk_Family_History.joblib")

# Load scaler
scaler_Insulin_Resistance_presence = load("modeling/scaler_presence_Insulin_Resistance.joblib")
scaler_Pulse_Pressure_presence = load("modeling/scaler_presence_Pulse_Pressure.joblib")
scaler_Diastolic_BP_presence = load("modeling/scaler_presence_Diastolic_BP.joblib")
scaler_Systolic_BP_presence = load("modeling/scaler_presence_Systolic_BP.joblib")
scaler_Resting_HR_presence = load("modeling/scaler_presence_Resting_HR.joblib")

scaler_Insulin_Resistance_risk = load("modeling/scaler_risk_Insulin_Resistance.joblib")
scaler_Pulse_Pressure_risk = load("modeling/scaler_risk_Pulse_Pressure.joblib")
scaler_Diastolic_BP_risk = load("modeling/scaler_risk_Diastolic_BP.joblib")
scaler_Systolic_BP_risk = load("modeling/scaler_risk_Systolic_BP.joblib")
scaler_Resting_HR_risk = load("modeling/scaler_risk_Resting_HR.joblib")

with open("features_order_risk.json", "r") as f:
    features_order_risk = json.load(f)

with open("features_order_presence.json", "r") as f:
    features_order_presence = json.load(f)

class RiskInput(BaseModel):
    Insulin_Resistance: float
    Pulse_Pressure: float
    Diastolic_BP: float
    Systolic_BP: float
    Resting_HR: float
    Hypertension: str
    ECG_Abnormality: str
    Diabetes: str
    Alcohol: str
    Previous_Stroke: str
    Family_History: str

class PresenceInput(RiskInput):
    CVD_Risk_Score: str

class User(BaseModel):
    nama : str
    email: str
    no_tlp: str

class PredictionRecord(BaseModel):
    nama: str
    email: str
    no_tlp: str
    target: str
    hasil_prediksi: str

@app.post("/register", response_model=Dict[str, Any])
def register_user(user: User) -> Dict[str, Any]:
    try:
        result = supabase.table("users").insert({
            "nama": user.nama,
            "email": user.email,
            "no_tlp": user.no_tlp,
            "registered_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        
        if getattr(result, "error", None) is None:
            return {
                "status": "success",
                "user_id": result.data[0]["id"]
            }
        else:
            raise HTTPException(status_code=500, detail="Gagal menyimpan data pengguna.")        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict-risk")
def predict_risk(data: RiskInput):
    try:
        df = pd.DataFrame([data.dict()])
        df_transformed = pd.DataFrame({
            "Insulin_Resistance":scaler_Insulin_Resistance_risk.transform(df[["Insulin_Resistance"]])[:, 0],
            "Pulse_Pressure":scaler_Pulse_Pressure_risk.transform(df[["Pulse_Pressure"]])[:, 0],
            "Diastolic_BP":scaler_Diastolic_BP_risk.transform(df[["Diastolic_BP"]])[:, 0],
            "Systolic_BP":scaler_Systolic_BP_risk.transform(df[["Systolic_BP"]])[:, 0],
            "Resting_HR":scaler_Resting_HR_risk.transform(df[["Resting_HR"]])[:, 0],
            "Hypertension":encoder_Hypertension_risk.transform(df["Hypertension"]),
            "ECG_Abnormality":encoder_ECG_Abnormality_risk.transform(df["ECG_Abnormality"]),
            "Diabetes":encoder_Diabetes_risk.transform(df["Diabetes"]),
            "Alcohol":encoder_Alcohol_risk.transform(df["Alcohol"]),
            "Previous_Stroke":encoder_Previous_Stroke_risk.transform(df["Previous_Stroke"]),
            "Family_History":encoder_Family_History_risk.transform(df["Family_History"])
        })
        df_transformed = df_transformed[features_order_risk]
        
        pred = model_risk.predict(df_transformed)
        result_risk = result_target_risk.inverse_transform(pred)[0]
        return {"prediction_risk": result_risk}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict-presence")
def predict_presence(data: PresenceInput):
    if model_presence is None:
        raise HTTPException(status_code=500, detail="Model presence tidak tersedia di server")
    
    try:
        df = pd.DataFrame([data.dict()])
        df_transformed = pd.DataFrame({
            "Insulin_Resistance":scaler_Insulin_Resistance_presence.transform(df[["Insulin_Resistance"]])[:, 0],
            "Pulse_Pressure":scaler_Pulse_Pressure_presence.transform(df[["Pulse_Pressure"]])[:, 0],
            "Diastolic_BP":scaler_Diastolic_BP_presence.transform(df[["Diastolic_BP"]])[:, 0],
            "Systolic_BP":scaler_Systolic_BP_presence.transform(df[["Systolic_BP"]])[:, 0],
            "Resting_HR":scaler_Resting_HR_presence.transform(df[["Resting_HR"]])[:, 0],
            "Hypertension":encoder_Hypertension_presence.transform(df["Hypertension"]),
            "ECG_Abnormality":encoder_ECG_Abnormality_presence.transform(df["ECG_Abnormality"]),
            "Diabetes":encoder_Diabetes_presence.transform(df["Diabetes"]),
            "Alcohol":encoder_Alcohol_presence.transform(df["Alcohol"]),
            "Previous_Stroke":encoder_Previous_Stroke_presence.transform(df["Previous_Stroke"]),
            "Family_History":encoder_Family_History_presence.transform(df["Family_History"]),
            "CVD_Risk_Score":encoder_CVD_Risk_Score_presence.transform(df["CVD_Risk_Score"])
        })
        df_transformed = df_transformed[features_order_presence]
                    
        pred = model_presence.predict(df_transformed)
        result_presence = result_target_presence.inverse_transform(pred)[0]
        return {"prediction_presence": result_presence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-prediction")
def save_prediction(data: PredictionRecord):
    try:
        payload = {
            "nama": str(data.nama),
            "email": str(data.email),
            "no_tlp": str(data.no_tlp),
            "target": str(data.target),
            "hasil_prediksi": str(data.hasil_prediksi),
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        response = supabase.table("predictions").insert(payload).execute()

        print("Supabase response:", response)

        if getattr(response, "error", None) is None:
            return {"status": "success", "message": "Prediksi berhasil disimpan."}
        else:
            raise HTTPException(status_code=500, detail="Gagal menyimpan ke Supabase")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 