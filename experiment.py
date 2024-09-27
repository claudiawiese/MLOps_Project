# Imports librairies
from mlflow import MlflowClient
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Define tracking_uri
client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

# Define experiment name, run name and artifact_path name
accident_experiment = mlflow.set_experiment("Accident_Models")
mlflow.autolog() 
run_name = "first_run"
artifact_path = "rf_accidents"

# Import Database
data = pd.read_csv("fake_data.csv")
X = data.drop(columns=["target_column"]) # drop target_column
X = X.astype('float')
y = data["target_column"] # define target_column
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model

#Define your params
params = {
    "n_estimators": 10,
    "max_depth": 10,
    "random_state": 42,
}
rf = RandomForestRegressor(**params)
rf.fit(X_train, y_train)

# Evaluate your model
y_pred = rf.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)
metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}