# predict.py
from fastapi import APIRouter, Depends, HTTPException
from starlette.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
from typing import Optional
from endpoints.auth import get_current_user, User

router = APIRouter()

class PredictionParams(BaseModel):
    model_uri: str  # URI for the model in MLflow
    data_path: str  # Path to the data file

# Define a mapping for predictions to labels
prediction_mapping = {
    0: "Minor Accident",
    1: "Severe Accident"
}

@router.post("/predict", response_class=JSONResponse)
async def predict_gravity(params: PredictionParams, current_user: User = Depends(get_current_user)):
    # Load the model from the specified URI
    try:
        model = mlflow.pyfunc.load_model(params.model_uri)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # Load the data from the specified file path
    try:
        df = pd.read_parquet(params.data_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load data: {e}")
    
    # Prepare data for prediction by removing the target column
    if "target" in df.columns:
        data = df.drop("target", axis=1)
    else:
        data = df  # Assuming all columns are for prediction if 'target' is missing

    # Make predictions
    try:
        predictions = model.predict(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # Map predictions to labels
    labeled_predictions = [
        prediction_mapping.get(int(pred), "Unknown") for pred in predictions
    ]

    # Return the labeled predictions as JSON
    return JSONResponse(content={"predictions": labeled_predictions})