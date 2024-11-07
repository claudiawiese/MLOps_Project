import mlflow
import subprocess
from typing import Optional

# Function to set up experiment and call train.py
def run_experiment(data_path: str, model_type: str, n_neighbors: Optional[int] = None,
                   weights: Optional[str] = None, n_estimators: Optional[int] = None,
                   max_depth: Optional[int] = None, retrain: bool = False):
    # Set up experiment in MLflow
    experiment_name = "Accident Experiment"
    mlflow.set_experiment(experiment_name)

    # Prepare the command for train.py based on model type
    command = [
        "python", "accident_project/train.py",
        "--data", data_path,
        "--model_type", model_type
    ]

    # Add model-specific parameters to the command
    if model_type == "KNN":
        if n_neighbors is not None:
            command.extend(["--n_neighbors", str(n_neighbors)])
        if weights is not None:
            command.extend(["--weights", weights])
    elif model_type == "RandomForest":
        if n_estimators is not None:
            command.extend(["--n_estimators", str(n_estimators)])
        if max_depth is not None:
            command.extend(["--max_depth", str(max_depth)])
    elif model_type == "Stacking":
        # Specify ensemble-related flags, if any
        command.append("--stacking")

    # Add retrain flag if specified
    if retrain:
        command.append("--retrain")

    # Run train.py with dynamically constructed command
    subprocess.run(command, check=True)