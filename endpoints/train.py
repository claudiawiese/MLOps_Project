# train.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from accident_project.experiment import run_experiment
from endpoints.auth import get_admin_user, User 

router = APIRouter()

class TrainParams(BaseModel):
    data: str
    model_type: str  # e.g., "KNN", "RandomForest", "Ensemble"
    n_neighbors: Optional[int] = 5  # Only relevant for KNN
    n_estimators: Optional[int] = 10 # Only relevant for RandomForest
    max_depth: Optional[int] = None  # Only relevant for RandomForest
    weights: Optional[str] = 'distance'  # Only relevant for KNN
    retrain: Optional[bool] = False  # Optional retrain flag

"""
# Endpoint pour télécharger un nouveau dataset et réentraîner le modèle
@app.get("/upload", response_class=HTMLResponse)
async def get_upload_page(request: Request):
    print('Upload Data')
"""

@router.post("/train")
async def train_model(params: TrainParams, current_user: User = Depends(get_admin_user)):
    try:
        # Call run_experiment directly with parameters from the request
        output = run_experiment(
            data_path=params.data,
            model_type=params.model_type,
            n_neighbors=params.n_neighbors,
            weights=params.weights,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            retrain=params.retrain
        )
    except Exception as e:
        # Return error if the experiment fails
        raise HTTPException(status_code=500, detail=f"Training process failed: {e}")

    return {"message": f"Model {params.model_type} training started successfully with provided parameters.", "output": output}
