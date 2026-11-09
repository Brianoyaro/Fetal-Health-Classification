import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI application
app = FastAPI(
    title="Fetal Health Classification Service",
    description="API for predicting fetal health status from clinical metrics using a trained XGBoost classifier.",
    version="1.0.0"
)

# Define the file path to our serialized pipeline
MODEL_PATH = "models/xgboost_pipeline.joblib"

# Global variable for model pipeline
model_pipeline = None

@app.on_event("startup")
def load_model():
    global model_pipeline
    if not os.path.exists(MODEL_PATH):
        # fallback check for current directory
        alternative_path = "xgboost_pipeline.joblib"
        if os.path.exists(alternative_path):
            payload = joblib.load(alternative_path)
            model_pipeline = payload["model"]
        else:
            raise RuntimeError(f"Model file not found at {MODEL_PATH} or {alternative_path}")
    else:
        payload = joblib.load(MODEL_PATH)
        model_pipeline = payload["model"]
    print("Model pipeline loaded successfully.")

# Define Pydantic request schema based on notebook features_list
class FetalHealthMetrics(BaseModel):
    prolongued_decelerations: float = Field(..., description="Number of prolonged decelerations per second", ge=0.0)
    accelerations: float = Field(..., description="Number of accelerations per second", ge=0.0)
    uterine_contractions: float = Field(..., description="Number of uterine contractions per second", ge=0.0)
    light_decelerations: float = Field(..., description="Number of light decelerations per second", ge=0.0)
    fetal_movement: float = Field(..., description="Number of fetal movements per second", ge=0.0)
    severe_decelerations: float = Field(..., description="Number of severe decelerations per second", ge=0.0)
    baseline_value: float = Field(..., description="Baseline fetal heart rate (FHR)", ge=0.0)

    class Config:
        schema_extra = {
            "example": {
                "prolongued_decelerations": 0.002,
                "accelerations": 0.001,
                "uterine_contractions": 0.007,
                "light_decelerations": 0.004,
                "fetal_movement": 0.0,
                "severe_decelerations": 0.0,
                "baseline_value": 133.0
            }
        }

class PredictionResponse(BaseModel):
    fetal_health_class: int = Field(..., description="Predicted class label (1: Normal, 2: Suspect, 3: Pathological)")
    class_probabilities: dict = Field(..., description="Probability scores per category")

@app.get("/health", status_code=200)
def health_check():
    '''Verifies if service is running and model is loaded.'''
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(metrics: FetalHealthMetrics):
    '''Accepts input metrics and returns fetal health classification prediction.'''
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not initialized.")
    
    try:
        # Convert metrics input to a dict representation matching expected model interface
        input_dict = [metrics.dict()]
        
        # Predict probability
        probabilities = model_pipeline.predict_proba(input_dict)[0]
        
        # Predict Class
        pred_idx = int(model_pipeline.predict(input_dict)[0])
        
        # XGBoost output is 0-indexed internally [0, 1, 2]. We shift it back to [1, 2, 3]
        actual_class = pred_idx + 1
        
        return {
            "fetal_health_class": actual_class,
            "class_probabilities": {
                "1_Normal": float(probabilities[0]),
                "2_Suspect": float(probabilities[1]),
                "3_Pathological": float(probabilities[2])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
