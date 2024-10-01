# Imports librairies
from mlflow import MlflowClient
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import sys

# Define tracking_uri
client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

# Define experiment name, run name and artifact_path name
apple_experiment = mlflow.set_experiment("Accident_Models")
run_name = "first_run"
artifact_path = "rf_accidents"

# Import Database
data = pd.read_csv("data/acc_data_encoded_2018-2022.parquet")
target = pd.read_csv("data/acc_target_encoded_2018-2022.parquet")

X = data
y = target
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model

#Define your params
params = {
    k = int(sys.argv[1])
    weights = int(sys.argv[2])
}

#KNN model 
knn = KNeighborsClassifier(**params)
knn.fit(X_train, y_train)

# Evaluate your model
y_pred = knn.predict(X_test)

print(classification_report(y_test, y_pred))
     
print(confusion_matrix(y_test, y_pred))


# Store information in tracking server
with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        sk_model=rf, input_example=X_val, artifact_path=artifact_path
    )
