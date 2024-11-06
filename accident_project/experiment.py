import mlflow
import subprocess

# Set up experiment
experiment_name = "Accident Experiment"
mlflow.set_experiment(experiment_name)

# Define hyperparameter grid
param_grid = {
    "data": 'data/dataset_Cramer.parquet',
    "n_neighbors": 5,
    "weights": 'distance'
}

# Run train.py with different parameters
subprocess.run([
    "python", "accident_project/train.py",
    "--data", param_grid["data"],
    "--n_estimators", str(param_grid["n_neighbors"]),
    "--weights", param_grid["weights"]
])
