# Imports librairies
from mlflow import MlflowClient
import mlflow
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
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
    k = 5
    weights = 'distance'
}

#KNN model 
knn = KNeighborsClassifier(**params)
knn.fit(X_train, y_train)

# Evaluate your model
y_pred = knn.predict(X_test)

print(classification_report(y_test, y_pred))
     
print(confusion_matrix(y_test, y_pred))