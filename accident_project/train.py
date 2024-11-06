# Imports librairies
from mlflow import MlflowClient
import mlflow
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import sys

# Manual argument parsing using sys.argv
def get_arg_value(arg_name, default=None):
    if arg_name in sys.argv:
        arg_index = sys.argv.index(arg_name) + 1
        if arg_index < len(sys.argv):
            return sys.argv[arg_index]
    return default

# Retrieve arguments
data_path = get_arg_value("--data")  # Required argument, no default
n_neighbors = int(get_arg_value("--n_neighbors", 5))
weights = get_arg_value("--weights", "uniform")

# Define tracking_uri
client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

# Define experiment name, run name and artifact_path name
apple_experiment = mlflow.set_experiment("Accident Experiment")
run_name = "first_run"
artifact_path = "knn_accidents"

def load_dataset(path):
    # Load the dataset
    df = pd.read_parquet(path)
    target = df["target"]
    data = df.drop("target", axis=1)
    return data, target

# Utility function to split data
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Import Database
X,y = load_dataset(data_path)
X_train, X_test, y_train, y_test = split_data(X,y)


# Train model

#Define your params
params = {
    "n_neighbors": n_neighbors,
    "weights": str(weights)
}

#KNN model 
knn = KNeighborsClassifier(**params)
knn.fit(X_train, y_train)

# Evaluate your model
y_pred = knn.predict(X_test)

classification_report = classification_report(y_test, y_pred)
     
confusion_matrix = confusion_matrix(y_test, y_pred)

# Calculate specific metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")


# Store information in tracking server
with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save the classification report to a text file and log it as an artifact
    with open("classification_report.txt", "w") as f:
        f.write(classification_report)
    mlflow.log_artifact("classification_report.txt")
 
    mlflow.sklearn.log_model(
        sk_model=knn,
        artifact_path="model",
        registered_model_name="KNN_Accident_Model" 
    )
